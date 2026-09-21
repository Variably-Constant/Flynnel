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
const CONTAINERS: [&str; 1] = ["host"];

/// Every leaf, by its normalized path.
const LEAVES: [&str; 4] = [
    "host/topology",
    "host/cpu",
    "host/latency",
    "host/cache",
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
            _ => None,
        })
    }

    /// The object shown for a container, which names it and says what
    /// it holds rather than being empty.
    fn container_value(path: &str) -> PsResult<PsObject> {
        let obj = pwrs::object::new();
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
        out
    }

    fn is_container(path: &str) -> bool {
        path.is_empty() || CONTAINERS.contains(&path)
    }

    fn is_leaf(path: &str) -> bool {
        LEAVES.contains(&path)
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
                let obj = pwrs::object::new();
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
