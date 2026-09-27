//! Proves the binding covers the scheduler, by reading the scheduler.
//!
//! A binding is missing a function and nothing says so: the module
//! imports, every suite passes, and the gap is invisible until someone
//! needs the thing. A test written against the module can never find
//! that, because the module is what is incomplete. So this reads the
//! crate's own source and asks, of every public item, whether the
//! binding exposes it or the census says why not.
//!
//! # What it reads
//!
//! `../src/**/*.rs`, parsed with syn. Every `pub` item, with the module
//! path that reaches it and the cfg gating it.
//!
//! An item declared `pub` inside a module nobody re-exports is not
//! reachable, so the module tree is walked first and an item counts
//! only when every module above it is `pub mod`. Counting unreachable
//! items would make the census a list of work nobody can do.
//!
//! `census.toml`, which names every public item deliberately not bound
//! and why.
//!
//! The built module's own manifest, `target/pwrs/Flynnel/Flynnel.psd1`,
//! whose CmdletsToExport and AliasesToExport are what is bound. Given
//! as the one positional argument. There is no separate descriptor
//! file: the manifest is what the build writes, and it is checked
//! rather than assumed because a tool reading a file that does not
//! exist would report every item uncovered and look like a finding.
//!
//! # What it fails on
//!
//! A public item in neither. A census entry for an item that no longer
//! exists. A reason that is not one of the six. A
//! `takes-a-rust-closure` entry that names no cmdlet reaching the
//! primitive. A `pending-on-main` entry whose commit is already an
//! ancestor of HEAD, so the reminder cannot be forgotten.
//!
//! # It stops rather than skipping
//!
//! A file it cannot read or parse, a module file that is not where the
//! declaration says, a type it cannot name: every one of these exits
//! non-zero naming the thing. None of them is skipped. A completeness
//! tool that quietly drops what it could not read reports a smaller
//! surface and calls it covered, which is the defect it exists to
//! catch, wearing its own uniform.
//!
//! Exit 0 is the only quiet answer, and it prints the counts anyway,
//! because a gate that prints nothing when it passes cannot be told
//! from one that did not run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The reasons a public item may be absent from the binding. Fixed:
/// an entry carrying anything else fails, because a free-text reason
/// is a place for "not yet" to hide.
const REASONS: [&str; 6] = [
    "generic-over-caller-type",
    "takes-a-rust-closure",
    "unsafe-or-structural",
    "internal-to-another-surface",
    "not-in-the-shipped-feature-set",
    "pending-on-main",
];

/// Stop, naming what could not be done. Exit 2 is distinct from the
/// exit 1 a census gap gives, so a broken tool and a real finding are
/// never the same answer to a caller.
fn fail(what: impl std::fmt::Display) -> ! {
    eprintln!("census: FAIL {what}");
    std::process::exit(2)
}

/// One public item found in the crate.
#[derive(Debug, Clone)]
struct Item {
    /// The path a caller would write, as `crate::a::b::Name`.
    path: String,
    /// fn, struct, enum, trait, method, const, type.
    kind: &'static str,
    /// The feature or target this item is gated behind, joined where
    /// there is more than one. Empty for an ungated item.
    cfg: String,
}

/// One entry of census.toml.
#[derive(Debug, Clone, Default)]
struct Entry {
    item: String,
    reason: String,
    reaches: String,
    commit: String,
    line: usize,
}

/// The entry covering `path`: the one naming it exactly, or the
/// longest one naming something it sits inside.
///
/// Longest, so a specific entry beats the family entry above it. That
/// is what lets a reader carve one item out of a subtree and give it
/// its own reason without splitting the family into three hundred
/// lines to do it.
///
/// A prefix only counts on a path boundary. Without that check
/// `crate::sched::plan` would cover `crate::sched::planner`, which
/// shares its text and none of its meaning.
fn covering_entry<'a>(entries: &'a [Entry], path: &str) -> Option<&'a str> {
    let mut best: Option<&'a str> = None;
    for entry in entries {
        let item = entry.item.as_str();
        let covers = path == item
            || (path.len() > item.len()
                && path.starts_with(item)
                && path[item.len()..].starts_with("::"));
        if covers && best.is_none_or(|held| item.len() > held.len()) {
            best = Some(item);
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(parent) = manifest_dir.parent() else {
        fail(format!(
            "{} has no parent, so the scheduler's src cannot be found",
            manifest_dir.display()
        ))
    };
    let crate_src = parent.join("src");
    if !crate_src.is_dir() {
        fail(format!("{} is not a directory", crate_src.display()));
    }
    let census_path = manifest_dir.join("census.toml");
    // The built module's manifest. Optional because the tool is useful
    // before the module has been built, but its absence is stated
    // rather than treated as an empty binding: an empty binding would
    // make every item look uncensused and bury the real gaps in noise.
    let manifest = args.iter().skip(1).find(|a| !a.starts_with('-')).cloned();

    let mut items = Vec::new();
    let mut files = 0usize;
    let reachable = reachable_modules(&crate_src);
    walk(&crate_src, &crate_src, &reachable, &mut items, &mut files);
    items.sort_by(|a, b| a.path.cmp(&b.path));

    println!("census: read {files} file(s) under {}", crate_src.display());
    println!(
        "census: {} module(s) reachable from the crate root",
        reachable.len()
    );
    println!("census: {} public item(s) in them", items.len());

    let (entries, mut failures) = read_census(&census_path);
    println!(
        "census: {} entry(ies) in {}",
        entries.len(),
        census_path.display()
    );

    let bound = match &manifest {
        Some(path) => {
            let set = read_descriptor(Path::new(path));
            println!("census: {} bound name(s) from {path}", set.len());
            Some(set)
        }
        None => {
            println!(
                "census: no module manifest given, so what is bound was not checked. Pass \
                 target/pwrs/Flynnel/Flynnel.psd1 to check it."
            );
            None
        }
    };

    // Whether an entry names something real is checked by the
    // covers-nothing pass further down, and only there.
    //
    // This used to also require the entry's path to be a public item
    // exactly, which is wrong for the subtree entries the file's own
    // header documents. A module is not among the public items this
    // tool collects, so every module-level entry failed the check
    // while covering its contents correctly. Ten of them did, across
    // the ring family and the backend family, and the failure hid
    // among the real gaps because the run fails on those anyway.
    //
    // The covers-nothing pass subsumes it. A misspelt leaf path covers
    // nothing and is caught; a path whose item was deleted covers
    // nothing and is caught; a module prefix that matches its contents
    // is legitimate and is not. One check, and it is the one that asks
    // the question that matters: does this entry describe anything
    // that is here.
    for entry in &entries {
        if !REASONS.contains(&entry.reason.as_str()) {
            failures.push(format!(
                "census.toml:{} gives {} the reason {:?}, which is not one of: {}",
                entry.line,
                entry.item,
                entry.reason,
                REASONS.join(", ")
            ));
        }
        if entry.reason == "takes-a-rust-closure" && entry.reaches.is_empty() {
            failures.push(format!(
                "census.toml:{} says {} takes a Rust closure but names no cmdlet that \
                 reaches it. A closure-taking primitive is reachable by running a declared \
                 kernel, and the entry has to say which.",
                entry.line, entry.item
            ));
        }
        if entry.reason == "pending-on-main" {
            if entry.commit.is_empty() {
                failures.push(format!(
                    "census.toml:{} is pending-on-main and names no commit, so nothing can \
                     tell when it stops being pending.",
                    entry.line
                ));
            } else if is_ancestor(&entry.commit) {
                failures.push(format!(
                    "census.toml:{} holds {} as pending-on-main behind {}, which is already \
                     an ancestor of HEAD. Bind it and delete the entry.",
                    entry.line, entry.item, entry.commit
                ));
            }
        }
    }

    // Every public item is bound or censused.
    //
    // An entry covers the item it names and everything beneath it, so
    // a type's entry covers its methods and a module's entry covers
    // its contents. Exact paths alone cannot express the fact that
    // actually holds here - a whole family is behind a feature this
    // module does not ship - and spelling that as three hundred
    // identical entries would be a worse record of it, not a better
    // one: nobody reads three hundred lines to learn one thing, and
    // the next person to add a function to that family has to
    // remember to add a line saying what every neighboring line
    // already says.
    //
    // How many items each entry covers is printed below, because a
    // prefix that quietly swallows more than its author meant is the
    // way this could go wrong.
    let mut covered: BTreeMap<&str, usize> = BTreeMap::new();
    for entry in &entries {
        covered.insert(entry.item.as_str(), 0);
    }
    if let Some(bound) = &bound {
        for item in &items {
            if let Some(entry) = covering_entry(&entries, &item.path) {
                *covered.entry(entry).or_insert(0) += 1;
                continue;
            }
            if bound.iter().any(|b| binds(b, &item.path)) {
                continue;
            }
            failures.push(format!(
                "{} ({}) is neither bound nor censused{}",
                item.path,
                item.kind,
                if item.cfg.is_empty() {
                    String::new()
                } else {
                    format!(", cfg {}", item.cfg)
                }
            ));
        }
        // An entry that covers nothing names an item that no longer
        // exists, or a prefix that never matched. Either way it is a
        // record of something that is not there, which is the failure
        // this tool exists to prevent, pointed at its own input.
        for entry in &entries {
            if covered.get(entry.item.as_str()).copied().unwrap_or(0) == 0 {
                failures.push(format!(
                    "census.toml:{} covers nothing: {} matches no public item and no subtree",
                    entry.line, entry.item
                ));
            }
        }
    }

    // The summary the release notes are built from, printed whether or
    // not the gate passes.
    if bound.is_some() {
        println!("census: what each entry covers");
        for entry in &entries {
            println!(
                "  {}: {} item(s)",
                entry.item,
                covered.get(entry.item.as_str()).copied().unwrap_or(0)
            );
        }
    }
    let mut by_reason: BTreeMap<&str, usize> = BTreeMap::new();
    for entry in &entries {
        *by_reason.entry(entry.reason.as_str()).or_insert(0) += 1;
    }
    println!("census: entries by reason");
    for reason in REASONS {
        println!("  {reason}: {}", by_reason.get(reason).copied().unwrap_or(0));
    }
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for item in &items {
        *by_kind.entry(item.kind).or_insert(0) += 1;
    }
    println!("census: public items by kind");
    for (kind, count) in &by_kind {
        println!("  {kind}: {count}");
    }

    if failures.is_empty() {
        // Saying PASS here without a manifest would be this tool
        // reporting success for the one question it exists to answer
        // and did not ask. Everything above still ran, so the counts
        // are real and worth printing; what is missing is the
        // comparison against what the module actually binds, and a
        // caller reading an exit code cannot see the line that says
        // so. Exit 2 for the same reason `fail` uses it: the tool was
        // not given what it needs, which is not the same answer as a
        // census gap.
        if bound.is_none() {
            eprintln!(
                "census: NOT CHECKED, no module manifest was given, so nothing above says \
                 whether the binding is complete"
            );
            std::process::exit(2);
        }
        println!("census: PASS");
        return;
    }
    eprintln!("census: FAIL, {} problem(s)", failures.len());
    for failure in &failures {
        eprintln!("  {failure}");
    }
    std::process::exit(1);
}

/// Whether a bound name covers an item path.
///
/// The descriptor carries PowerShell names and the crate carries Rust
/// paths, so the match is on the last segment. Loose on purpose: a
/// false match here hides a gap, so the census entry is the strict
/// record and this is only the shortcut that keeps the obvious cases
/// out of it.
fn binds(bound: &str, path: &str) -> bool {
    // A path with no separator is its own last segment, which is the
    // answer rather than a fallback for one.
    let tail = match path.rfind("::") {
        Some(at) => &path[at + 2..],
        None => path,
    };
    let squashed: String = bound.chars().filter(|c| *c != '-' && *c != '.').collect();
    let pascal = to_pascal(tail);
    squashed.eq_ignore_ascii_case(&pascal) || squashed.ends_with(&pascal)
}

/// A snake_case name as PascalCase, which is how the binding spells it.
fn to_pascal(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut upper = true;
    for c in name.chars() {
        if c == '_' {
            upper = true;
            continue;
        }
        if upper {
            out.extend(c.to_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// Parse one source file, stopping if it cannot be read or parsed.
fn parse(path: &Path) -> syn::File {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => fail(format!("could not read {}: {e}", path.display())),
    };
    match syn::parse_file(&text) {
        Ok(parsed) => parsed,
        Err(e) => fail(format!("could not parse {}: {e}", path.display())),
    }
}

/// The module paths reachable from the crate root through `pub mod`
/// declarations only.
///
/// A `pub fn` inside a private module is not public surface, and
/// listing it would make the census a list of work nobody can do.
fn reachable_modules(src: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    out.insert(String::new());
    // Repeated until nothing new is added, rather than recursed: the
    // file reads stay flat and a cycle a malformed tree could make
    // still terminates.
    loop {
        let before = out.len();
        let known: Vec<String> = out.iter().cloned().collect();
        for parent in known {
            for child in pub_mods_of(src, &parent) {
                let path = if parent.is_empty() {
                    child
                } else {
                    format!("{parent}::{child}")
                };
                out.insert(path);
            }
        }
        if out.len() == before {
            return out;
        }
    }
}

/// The `pub mod` names a module's own file declares that live in
/// files of their own.
///
/// An inline `pub mod name { ... }` has no file and is not returned:
/// its items are collected by the recursion in `collect`, at the
/// deeper path, when the enclosing file is read. Returning it here
/// would send the walk looking for a file that was never meant to
/// exist, which is what `gpu_peer::linalg::cpu` did.
fn pub_mods_of(src: &Path, module: &str) -> Vec<String> {
    let file = module_file(src, module);
    parse(&file)
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(m) if is_public(&m.vis) && m.content.is_none() => {
                Some(m.ident.to_string())
            }
            _ => None,
        })
        .collect()
}

/// Where a module's source lives, as either `a/b.rs` or `a/b/mod.rs`.
///
/// A declared module with neither file is a broken tree rather than an
/// empty module, so it stops here instead of contributing nothing.
fn module_file(src: &Path, module: &str) -> PathBuf {
    if module.is_empty() {
        let lib = src.join("lib.rs");
        if !lib.exists() {
            fail(format!("{} does not exist", lib.display()));
        }
        return lib;
    }
    let rel: PathBuf = module.split("::").collect();
    let flat = src.join(&rel).with_extension("rs");
    if flat.exists() {
        return flat;
    }
    let nested = src.join(&rel).join("mod.rs");
    if nested.exists() {
        return nested;
    }
    fail(format!(
        "module {module} is declared but neither {} nor {} exists",
        flat.display(),
        nested.display()
    ))
}

/// Walk the source tree, collecting public items from every reachable
/// module.
fn walk(
    src: &Path,
    dir: &Path,
    reachable: &BTreeSet<String>,
    out: &mut Vec<Item>,
    files: &mut usize,
) {
    let listing = match std::fs::read_dir(dir) {
        Ok(listing) => listing,
        Err(e) => fail(format!("could not read {}: {e}", dir.display())),
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in listing {
        match entry {
            Ok(entry) => paths.push(entry.path()),
            Err(e) => fail(format!("could not read an entry of {}: {e}", dir.display())),
        }
    }
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(src, &path, reachable, out, files);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        *files += 1;
        let module = module_of(src, &path);
        if !reachable.contains(&module) {
            continue;
        }
        collect(&parse(&path).items, &module, String::new(), out);
    }
}

/// The module path a source file provides.
fn module_of(src: &Path, file: &Path) -> String {
    let rel = match file.strip_prefix(src) {
        Ok(rel) => rel,
        Err(e) => fail(format!(
            "{} is not under {}: {e}",
            file.display(),
            src.display()
        )),
    };
    let mut parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let Some(last) = parts.pop() else {
        fail(format!("{} has no file name", file.display()))
    };
    let stem = last.trim_end_matches(".rs");
    if stem != "mod" && stem != "lib" {
        parts.push(stem.to_string());
    }
    parts.join("::")
}

/// The cfg attributes on an item, as one string.
fn cfg_of(attrs: &[syn::Attribute], where_: &str) -> String {
    let mut out = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("cfg") {
            continue;
        }
        match attr.meta.require_list() {
            Ok(list) => out.push(list.tokens.to_string()),
            Err(e) => fail(format!(
                "a cfg attribute in {where_} has no parenthesized list, so what gates the \
                 item cannot be read: {e}"
            )),
        }
    }
    out.join(" and ")
}

/// Whether an item is `pub` outright, as opposed to `pub(crate)` or
/// private. Only the first is surface a binding could expose.
fn is_public(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

/// Join an inherited cfg with an item's own.
fn join_cfg(outer: &str, own: String) -> String {
    match (outer.is_empty(), own.is_empty()) {
        (true, _) => own,
        (false, true) => outer.to_string(),
        (false, false) => format!("{outer} and {own}"),
    }
}

/// Collect the public items of one module's item list.
fn collect(items: &[syn::Item], module: &str, outer_cfg: String, out: &mut Vec<Item>) {
    let where_ = if module.is_empty() {
        "the crate root".to_string()
    } else {
        format!("module {module}")
    };
    let name = |ident: &syn::Ident| {
        if module.is_empty() {
            format!("crate::{ident}")
        } else {
            format!("crate::{module}::{ident}")
        }
    };
    for item in items {
        match item {
            syn::Item::Fn(f) if is_public(&f.vis) => out.push(Item {
                path: name(&f.sig.ident),
                kind: "fn",
                cfg: join_cfg(&outer_cfg, cfg_of(&f.attrs, &where_)),
            }),
            syn::Item::Struct(s) if is_public(&s.vis) => out.push(Item {
                path: name(&s.ident),
                kind: "struct",
                cfg: join_cfg(&outer_cfg, cfg_of(&s.attrs, &where_)),
            }),
            syn::Item::Enum(e) if is_public(&e.vis) => out.push(Item {
                path: name(&e.ident),
                kind: "enum",
                cfg: join_cfg(&outer_cfg, cfg_of(&e.attrs, &where_)),
            }),
            syn::Item::Trait(t) if is_public(&t.vis) => out.push(Item {
                path: name(&t.ident),
                kind: "trait",
                cfg: join_cfg(&outer_cfg, cfg_of(&t.attrs, &where_)),
            }),
            syn::Item::Const(c) if is_public(&c.vis) => out.push(Item {
                path: name(&c.ident),
                kind: "const",
                cfg: join_cfg(&outer_cfg, cfg_of(&c.attrs, &where_)),
            }),
            syn::Item::Type(t) if is_public(&t.vis) => out.push(Item {
                path: name(&t.ident),
                kind: "type",
                cfg: join_cfg(&outer_cfg, cfg_of(&t.attrs, &where_)),
            }),
            // Inherent methods, which are most of the surface a script
            // reaches. Trait impls are skipped: a trait method is
            // bound through its trait, not once per implementation.
            syn::Item::Impl(i) if i.trait_.is_none() => {
                let owner = type_name(&i.self_ty, &where_);
                let impl_cfg = join_cfg(&outer_cfg, cfg_of(&i.attrs, &where_));
                let base = if module.is_empty() {
                    format!("crate::{owner}")
                } else {
                    format!("crate::{module}::{owner}")
                };
                for sub in &i.items {
                    if let syn::ImplItem::Fn(f) = sub
                        && is_public(&f.vis)
                    {
                        out.push(Item {
                            path: format!("{base}::{}", f.sig.ident),
                            kind: "method",
                            cfg: join_cfg(&impl_cfg, cfg_of(&f.attrs, &where_)),
                        });
                    }
                }
            }
            // An inline `pub mod` inside a file, whose items belong to
            // a deeper path than the file's own.
            syn::Item::Mod(m) if is_public(&m.vis) => {
                if let Some((_, inner)) = &m.content {
                    let deeper = if module.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{module}::{}", m.ident)
                    };
                    let cfg = join_cfg(&outer_cfg, cfg_of(&m.attrs, &where_));
                    collect(inner, &deeper, cfg, out);
                }
            }
            _ => {}
        }
    }
}

/// The bare name of a type in an inherent impl header.
///
/// A self type this cannot name would give every method under it a
/// path no census entry could match, so it stops rather than inventing
/// one.
fn type_name(ty: &syn::Type, where_: &str) -> String {
    let syn::Type::Path(path) = ty else {
        fail(format!(
            "an inherent impl in {where_} is on a type this tool cannot name, so its methods \
             would be censused under a path nobody could write"
        ))
    };
    match path.path.segments.last() {
        Some(segment) => segment.ident.to_string(),
        None => fail(format!("an inherent impl in {where_} has an empty type path")),
    }
}

/// Read census.toml.
///
/// Hand-parsed rather than through a toml crate, on the same reasoning
/// the crate's own target test uses: the shape is a repeated block of
/// `key = "value"` lines, and a dependency to read it would be carried
/// by everything that builds this tool.
fn read_census(path: &Path) -> (Vec<Entry>, Vec<String>) {
    let mut problems = Vec::new();
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Absent is a state to report, not to treat as empty: an
            // empty census makes every unbound item a failure and
            // buries the real ones.
            problems.push(format!(
                "{} does not exist, so nothing is censused and every unbound public item \
                 below is reported",
                path.display()
            ));
            return (Vec::new(), problems);
        }
        Err(e) => fail(format!("could not read {}: {e}", path.display())),
    };
    let mut entries: Vec<Entry> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let number = index + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[entry]]" {
            entries.push(Entry {
                line: number,
                ..Entry::default()
            });
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            problems.push(format!(
                "{}:{number} is neither a comment, an [[entry]] header nor a key = \"value\" \
                 line: {line}",
                path.display()
            ));
            continue;
        };
        let value = value.trim().trim_matches('"').to_string();
        let Some(current) = entries.last_mut() else {
            problems.push(format!(
                "{}:{number} sets {} before any [[entry]] header",
                path.display(),
                key.trim()
            ));
            continue;
        };
        match key.trim() {
            "item" => current.item = value,
            "reason" => current.reason = value,
            "reaches" => current.reaches = value,
            "commit" => current.commit = value,
            "note" => {}
            other => problems.push(format!(
                "{}:{number} sets {other}, which is not one of item, reason, reaches, commit \
                 or note",
                path.display()
            )),
        }
    }
    for entry in &entries {
        if entry.item.is_empty() {
            problems.push(format!(
                "{}:{} is an entry with no item",
                path.display(),
                entry.line
            ));
        }
    }
    (entries, problems)
}

/// The names the built module exports.
///
/// The build writes a PowerShell module manifest, `Flynnel.psd1`,
/// whose CmdletsToExport and AliasesToExport carry every bound name.
/// Pass that file, or the format file beside it for the type names.
///
/// Every quoted string, in either quote style, because the question
/// asked of the file is only whether a name appears in it. A manifest
/// quotes its names with apostrophes and an XML format file with
/// double quotes, and reading only one style would find nothing in the
/// other and report a module that binds nothing.
///
/// A file carrying no names at all stops the run rather than reading
/// as an empty binding, which would make every public item look
/// uncensused at once and bury whatever the real gaps are.
fn read_descriptor(path: &Path) -> BTreeSet<String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => fail(format!(
            "could not read the module manifest {}: {e}",
            path.display()
        )),
    };
    let mut out = BTreeSet::new();
    let mut rest = text.as_str();
    while let Some(open) = rest.find(['"', '\'']) {
        let Some(quote) = rest[open..].chars().next() else { break };
        rest = &rest[open + 1..];
        let Some(close) = rest.find(quote) else { break };
        let name = &rest[..close];
        if !name.is_empty() {
            out.insert(name.to_string());
        }
        rest = &rest[close + 1..];
    }
    if out.is_empty() {
        fail(format!("{} carried no names", path.display()));
    }
    out
}

/// Whether a commit is already an ancestor of HEAD.
///
/// A pending-on-main entry whose commit has landed is a reminder that
/// has expired, and the whole point of the mechanism is that it cannot
/// be forgotten. A git that cannot be run at all stops the tool rather
/// than answering no, because answering no would retire every pending
/// entry's check silently.
fn is_ancestor(commit: &str) -> bool {
    match std::process::Command::new("git")
        .args(["merge-base", "--is-ancestor", commit, "HEAD"])
        .status()
    {
        Ok(status) => status.success(),
        Err(e) => fail(format!(
            "could not run git to ask whether {commit} is an ancestor of HEAD: {e}. Without \
             it a pending-on-main entry cannot be checked, and passing it unchecked is what \
             this entry exists to prevent."
        )),
    }
}
