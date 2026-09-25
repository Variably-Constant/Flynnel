//! The `Flynnel:` drive: a running scheduler, browsed.
//!
//! # Read only, and that is a decision rather than an omission
//!
//! There is no `New-Item`, no `Remove-Item`, no `Set-Content`. A drive
//! that could change the scheduler would be a second way to do what
//! the Set- cmdlets already do, and two ways to write one setting is
//! how they drift apart. The binding framework's own defaults refuse
//! every write with a clear error, so the refusal is what this
//! provider gets by not implementing them.
//!
//! # Every leaf is the object a cmdlet writes
//!
//! Not a second rendering of it. `Flynnel:\host\cpu` answers the same
//! `Flynnel.CpuInfo` that `Get-FlynnelCpuInfo` writes, built by the
//! same function. Two renderings of one reading drift, and a script
//! comparing a drive against a cmdlet would be comparing this module
//! against itself. The suite asserts the equality field by field,
//! which is the check that keeps them together.
//!
//! One consequence a caller meets immediately: a leaf has no `Name`.
//! A `Flynnel.CpuInfo` carries no such property, and adding one here
//! would make the drive's object differ from the cmdlet's, which is
//! the single thing this design must not do. `PSChildName` is the
//! name, supplied by the engine from the item's path, and it is what
//! `Get-ChildItem | ForEach-Object { $_.PSChildName }` reads.
//! Containers do carry `Name`, because their object is this
//! provider's own and has nothing to stay equal to.
//!
//! # A level that cannot be read says so
//!
//! A container that exists and is empty is a different answer from a
//! path that is not there, and a script cannot tell them apart if the
//! second is used for the first. So a reading this host cannot take -
//! the latency table on a machine whose ping-pong sweep could not pin
//! threads - is a leaf that exists and holds nothing.

use pwrs::prelude::*;

/// The drive's tree, which is fixed: the scheduler's shape does not
/// change, only its readings do.
///
/// Held as a path list rather than built per call so enumerating a
/// container is one pass over this table, and so a path that is not
/// here is not there.
const CONTAINERS: [&str; 8] = [
    "host",
    "pool",
    "pool/workers",
    "sites",
    "backends",
    "calibration",
    "trace",
    "peer",
];

/// Every leaf whose path is fixed, by its normalized path.
///
/// `pool/workers` has children too, but how many is a reading rather
/// than a shape, so they are not here; [`FlynnelDrive::children_of`]
/// asks the pool.
const LEAVES: [&str; 10] = [
    "host/topology",
    "host/cpu",
    "host/latency",
    "host/cache",
    "pool/summary",
    "pool/spin",
    "pool/split",
    "calibration/summary",
    "calibration/thresholds",
    "trace/state",
];

/// A provider path in internal form: forward slashes, no leading or
/// trailing separator, and the root is the empty string.
fn norm(path: &str) -> String {
    path.replace('\\', "/").trim_matches('/').to_string()
}

/// The container a path sits in, or the empty root.
fn parent_of(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// A browsable view of the running scheduler.
///
/// One instance per drive. It holds nothing: every answer is read from
/// the scheduler when it is asked for, because a drive that cached a
/// reading would show a pool that has since been started or a
/// calibration that has since been taken.
#[provider(name = "Flynnel")]
#[derive(Default)]
pub struct FlynnelDrive {}

impl FlynnelDrive {
    /// The object at one leaf, or None where the host cannot take that
    /// reading.
    fn leaf_value(path: &str) -> PsResult<Option<PsObject>> {
        // The three levels whose children are readings carry their key
        // in the name, so they are matched before the fixed paths.
        if let Some(name) = path.strip_prefix("pool/workers/") {
            return match Self::worker_by_name(name) {
                Some(w) => Ok(Some(w.into_ps()?)),
                None => Ok(None),
            };
        }
        if let Some(name) = path.strip_prefix("sites/") {
            // Matched by name within each reading of the registry, so a
            // site registered between two readings cannot pair one site's
            // name with another's row.
            if let Some(e) = flynnel::registered_sites()
                .iter()
                .find(|e| Self::located_site_name(e) == name)
            {
                return Ok(Some(crate::observe::call_site_row(e).into_ps()?));
            }
            if let Some(e) = flynnel::registered_keyed_sites()
                .iter()
                .find(|e| Self::keyed_site_name(e) == name)
            {
                return Ok(Some(crate::observe::keyed_site_row(e).into_ps()?));
            }
            return Ok(None);
        }
        if let Some(name) = path.strip_prefix("backends/") {
            let detected = flynnel::backend::detect::detect_all();
            let found = crate::backends::BackendKind::ENUMERABLE
                .iter()
                .find(|k| format!("{k:?}") == name);
            return match found {
                Some(k) => Ok(Some(
                    crate::backends::row_for(k.to_crate(0), &detected).into_ps()?,
                )),
                None => Ok(None),
            };
        }
        Ok(match path {
            "host/topology" => Some(crate::host::topology_snapshot().into_ps()?),
            "host/cpu" => Some(crate::host::cpu_info_row().into_ps()?),
            "host/cache" => Some(crate::host::cache_allocation_row().into_ps()?),
            // The one reading a host can genuinely lack. An empty leaf
            // rather than an absent path, so a script can tell "this
            // host took no measurement" from "no such thing exists".
            "host/latency" => match crate::host::latency_table_row() {
                Some(row) => Some(row.into_ps()?),
                None => None,
            },
            "pool/summary" => Some(crate::pool::pool_snapshot().into_ps()?),
            "pool/spin" => Some(crate::pool::spin_snapshot().into_ps()?),
            "pool/split" => Some(crate::pool::split_snapshot().into_ps()?),
            "calibration/summary" => Some(crate::calibration::calibration_row().into_ps()?),
            "calibration/thresholds" => Some(crate::calibration::threshold_row().into_ps()?),
            "trace/state" => Some(crate::observe::trace_state_row().into_ps()?),
            // Present only while a peer is. The container above stays
            // either way, so a script can tell "no peer is running"
            // from "this module has no peer level".
            "peer/summary" => match crate::gpupeer::running_peer_row() {
                Some(row) => Some(row.into_ps()?),
                None => None,
            },
            _ => None,
        })
    }

    /// The worker a child name spells, or None when no child of that
    /// name is there.
    ///
    /// Matched against the name each row enumerates under rather than
    /// by parsing the segment, so the only names that resolve are the
    /// ones the level lists: "5" finds worker five and "05" finds
    /// nothing, which is what Get-ChildItem showed.
    ///
    /// The external slots a foreign thread pushes through are left out
    /// of the drive. They are real rows and they are not workers, and
    /// a level called `workers` holding some things that are not is
    /// worse than one that omits them; Get-FlynnelWorker takes them
    /// with a switch, which is where a caller who wants them asks.
    fn worker_by_name(name: &str) -> Option<crate::pool::WorkerStat> {
        crate::pool::worker_rows(false)
            .into_iter()
            .find(|w| w.index.to_string() == name)
    }

    /// The object shown for a container, which names it and says what
    /// it holds rather than being empty.
    fn container_value(path: &str) -> PsResult<PsObject> {
        let obj = pwrs::object::new_psobject("Flynnel.DriveContainer");
        let name = if path.is_empty() {
            "Flynnel".to_string()
        } else {
            path.rsplit('/').next().unwrap_or(path).to_string()
        };
        pwrs::object::add_note(&obj, "Name", name.into_ps()?)?;
        pwrs::object::add_note(&obj, "Path", path.replace('/', "\\").into_ps()?)?;
        pwrs::object::add_note(&obj, "IsContainer", true.into_ps()?)?;
        pwrs::object::add_note(
            &obj,
            "ChildCount",
            (Self::children_of(path).len() as i32).into_ps()?,
        )?;
        Ok(obj)
    }

    /// Every child of one container, containers and leaves together,
    /// in one pass over the tree.
    ///
    /// `pool\workers` is the one level whose children are a reading,
    /// and it is built from one call into the pool rather than one per
    /// child: a boundary crossing per worker would make enumerating a
    /// large pool cost more than the answer is worth.
    fn children_of(path: &str) -> Vec<(String, bool)> {
        let mut out = Vec::new();
        for c in CONTAINERS {
            if parent_of(c) == path {
                out.push((c.to_string(), true));
            }
        }
        for l in LEAVES {
            if parent_of(l) == path {
                out.push((l.to_string(), false));
            }
        }
        if path == "pool/workers" {
            for w in crate::pool::worker_rows(false) {
                out.push((format!("pool/workers/{}", w.index), false));
            }
        }
        if path == "sites" {
            for name in Self::site_names() {
                out.push((format!("sites/{name}"), false));
            }
        }
        if path == "backends" {
            for name in Self::backend_names() {
                out.push((format!("backends/{name}"), false));
            }
        }
        // The peer level exists whether or not a peer does. A host
        // with no GPU gets a container that enumerates nothing, not a
        // missing path, because a missing path and a missing device
        // read alike to a script and only one of them is worth
        // retrying.
        if path == "peer" && crate::gpupeer::peer_is_running() {
            out.push(("peer/summary".to_string(), false));
        }
        out
    }

    /// Every call site the scheduler has materialised: those at a source
    /// location, named by it, then those another native library keyed,
    /// named by the key.
    ///
    /// A site appears only once a dispatch has reached it, so an empty
    /// level is a process that has run no work through Flynnel rather
    /// than a level that failed to enumerate.
    fn site_names() -> Vec<String> {
        let mut names: Vec<String> = flynnel::registered_sites()
            .iter()
            .map(Self::located_site_name)
            .collect();
        names.extend(flynnel::registered_keyed_sites().iter().map(Self::keyed_site_name));
        names
    }

    /// A located site's name: its file and line.
    ///
    /// Every separator a path segment cannot carry becomes a hyphen: the
    /// slashes of the file's own path, the colon of a Windows drive
    /// letter, which the crate's absolute source path begins with when
    /// it is built outside the module's directory, and the one a
    /// location puts between file and line. A colon left in the name
    /// would read as a drive separator. The row still holds File, Line
    /// and Column, which is what a script reads; the name only has to
    /// be unique and typeable.
    fn located_site_name(e: &flynnel::RegisteredSite) -> String {
        let file = e.location.file().replace(['\\', '/', ':'], "-");
        format!("{file}-{}", e.location.line())
    }

    /// A keyed site's name: `key-` and the key as sixteen hex digits,
    /// which no located name can be, since those begin with a file path.
    fn keyed_site_name(e: &flynnel::RegisteredKeyedSite) -> String {
        format!("key-{:016x}", e.key)
    }

    /// Every backend kind the taxonomy names, whether or not this host
    /// has it. An absent device is a child that exists and reports
    /// Registered false, never a missing child, for the same reason
    /// Get-FlynnelBackend writes a row for it.
    fn backend_names() -> Vec<String> {
        crate::backends::BackendKind::ENUMERABLE
            .iter()
            .map(|k| format!("{k:?}"))
            .collect()
    }

    fn is_container(path: &str) -> bool {
        path.is_empty() || CONTAINERS.contains(&path)
    }

    fn is_leaf(path: &str) -> bool {
        if LEAVES.contains(&path) {
            return true;
        }
        if path == "peer/summary" {
            return crate::gpupeer::peer_is_running();
        }
        if let Some(name) = path.strip_prefix("pool/workers/") {
            return Self::worker_by_name(name).is_some();
        }
        if let Some(name) = path.strip_prefix("sites/") {
            return Self::site_names().iter().any(|n| n == name);
        }
        if let Some(name) = path.strip_prefix("backends/") {
            return Self::backend_names().iter().any(|n| n == name);
        }
        false
    }
}

impl Provider for FlynnelDrive {
    /// The drive exists as soon as the module is imported, because a
    /// scheduler is always there to browse; there is nothing for a
    /// caller to mount.
    fn default_drives() -> PsResult<Vec<(Drive, FlynnelDrive)>> {
        Ok(vec![(
            Drive {
                name: "Flynnel".to_string(),
                root: String::new(),
            },
            FlynnelDrive::default(),
        )])
    }

    /// A second drive over the same scheduler, for a caller who wants
    /// one under another name. It reads the same process, because
    /// there is only one scheduler to read.
    fn new_drive(name: &str, _root: &str) -> PsResult<(Drive, FlynnelDrive)> {
        Ok((
            Drive {
                name: name.to_string(),
                root: String::new(),
            },
            FlynnelDrive::default(),
        ))
    }

    fn item_exists(&mut self, path: &str) -> PsResult<bool> {
        let p = norm(path);
        Ok(Self::is_container(&p) || Self::is_leaf(&p))
    }

    fn is_item_container(&mut self, path: &str) -> PsResult<bool> {
        Ok(Self::is_container(&norm(path)))
    }

    fn get_item(&mut self, path: &str) -> PsResult<Option<Item>> {
        let p = norm(path);
        if Self::is_container(&p) {
            return Ok(Some(Item::container(
                p.replace('/', "\\"),
                Self::container_value(&p)?,
            )));
        }
        if !Self::is_leaf(&p) {
            return Ok(None);
        }
        // A leaf whose reading this host cannot take is still a leaf.
        // Answering None here would make it indistinguishable from a
        // path that does not exist.
        let value = match Self::leaf_value(&p)? {
            Some(v) => v,
            None => {
                let obj = pwrs::object::new_psobject("Flynnel.DriveUnavailable");
                pwrs::object::add_note(
                    &obj,
                    "Name",
                    p.rsplit('/').next().unwrap_or(&p).to_string().into_ps()?,
                )?;
                pwrs::object::add_note(
                    &obj,
                    "Unavailable",
                    "this host took no such reading".to_string().into_ps()?,
                )?;
                obj
            }
        };
        Ok(Some(Item::leaf(p.replace('/', "\\"), value)))
    }

    fn get_child_items(&mut self, path: &str, recurse: bool) -> PsResult<Vec<Item>> {
        let p = norm(path);
        if !Self::is_container(&p) {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for (child, is_container) in Self::children_of(&p) {
            if is_container {
                out.push(Item::container(
                    child.replace('/', "\\"),
                    Self::container_value(&child)?,
                ));
                if recurse {
                    out.extend(self.get_child_items(&child, true)?);
                }
            } else if let Some(item) = self.get_item(&child)? {
                out.push(item);
            }
        }
        Ok(out)
    }

    fn has_child_items(&mut self, path: &str) -> PsResult<bool> {
        Ok(!Self::children_of(&norm(path)).is_empty())
    }

    /// A leaf's rows. One object for a leaf that answers a single row,
    /// which is every leaf at this level; a leaf whose reading the
    /// host cannot take streams nothing rather than a placeholder,
    /// because Get-Content is asked for rows and there are none.
    fn get_content(&mut self, path: &str) -> PsResult<Vec<PsObject>> {
        let p = norm(path);
        if !Self::is_leaf(&p) {
            return Err(PsError::new(
                ErrorCategory::InvalidOperation,
                "FlynnelDriveNotALeaf",
                format!("{p} is not a leaf, so it has no content to read"),
            ));
        }
        Ok(match Self::leaf_value(&p)? {
            Some(v) => vec![v],
            None => Vec::new(),
        })
    }
}
