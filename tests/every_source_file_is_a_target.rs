//! Every file under a target directory is a target cargo knows about.
//!
//! Declaring any target of a kind turns off cargo's discovery for that
//! whole kind. A file it then does not know about is not built, not
//! linted, not run, and fails nothing, so a green gate says nothing
//! about it either way.
//!
//! This manifest declares every example, and a file was added to
//! examples/ without one. Nothing compiled it and the gate passed.
//! Seven older files were in the same state, one of which no longer
//! satisfied clippy.
//!
//! A kind with no declarations at all is discovering its files, so
//! every one of them is already a target and the kind is skipped. That
//! is what tests/ does today, and it is one `[[test]]` block away from
//! not doing it: adding one would strand every other suite here, and
//! the count of suites that ran would fall with nothing failing.

use std::path::Path;

/// Names declared for `kind`, in manifest order.
///
/// A hand parse rather than a toml dependency: the shape read here is
/// two literal lines, and a dev-dependency to read them would be
/// carried by every consumer running this crate's tests.
fn declared_names(manifest: &str, kind: &str) -> Vec<String> {
    let header = format!("[[{kind}]]");
    let mut out = Vec::new();
    let mut in_block = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed == header {
            in_block = true;
            continue;
        }
        if trimmed.starts_with('[') {
            in_block = false;
            continue;
        }
        if !in_block || !trimmed.starts_with("name") {
            continue;
        }
        if let Some(open) = trimmed.find('"')
            && let Some(len) = trimmed[open + 1..].find('"')
        {
            out.push(trimmed[open + 1..open + 1 + len].to_string());
            in_block = false;
        }
    }
    out
}

/// Stems of the `.rs` files directly under `dir`, sorted.
///
/// A directory that is not there has none, which is not a failure: a
/// crate need not have benches. Any other read failure panics with the
/// path and the reason, because returning an empty list for it would
/// mark the kind clean on the strength of a directory nobody could
/// read, which is the outcome this file exists to refuse.
fn source_stems(dir: &Path) -> Vec<String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(err) => panic!("cannot read {}: {err}", dir.display()),
    };
    let mut out = Vec::new();
    for entry in entries {
        let path = match entry {
            Ok(entry) => entry.path(),
            Err(err) => panic!("cannot read an entry under {}: {err}", dir.display()),
        };
        if path.extension().is_some_and(|ext| ext == "rs")
            && let Some(stem) = path.file_stem()
        {
            out.push(stem.to_string_lossy().into_owned());
        }
    }
    out.sort();
    out
}

#[test]
fn every_source_file_is_a_target_cargo_knows_about() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("the manifest sits beside the directories this test reads");

    let mut census = Vec::new();
    let mut missing = Vec::new();

    for (kind, dir) in [
        ("example", "examples"),
        ("bench", "benches"),
        ("test", "tests"),
        ("bin", "src/bin"),
    ] {
        let declared = declared_names(&manifest, kind);
        let files = source_stems(&root.join(dir));

        // Recorded for every kind, including the ones that pass, so a
        // failure message distinguishes a census that found nothing
        // from one that never looked.
        census.push(format!(
            "{kind:<8} files={:<3} declared={:<3}{}",
            files.len(),
            declared.len(),
            if declared.is_empty() { "  (discovering)" } else { "" }
        ));

        if declared.is_empty() {
            continue;
        }
        for file in &files {
            if !declared.iter().any(|name| name == file) {
                missing.push(format!("  {kind} {file}"));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "these files are in a target directory and are not targets, so nothing \
         builds, lints or runs them:\n{}\n\ncensus:\n{}",
        missing.join("\n"),
        census.join("\n")
    );
}
