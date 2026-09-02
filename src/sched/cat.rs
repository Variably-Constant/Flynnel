//! CPU cache residency through Cache Allocation Technology (resctrl).
//!
//! On Linux, `/sys/fs/resctrl` exposes the CPU's L3 cache-way
//! allocation. You carve a class of service with a capacity bitmask -
//! one bit per L3 way - and bind a process to it, reserving those ways
//! so a noisy neighbor cannot evict the bound process's working set.
//! AMD calls this Platform QoS and supports L3 CAT (plus MBA/SMBA)
//! from Zen2 on; Intel calls it RDT. Neither vendor lets you pin an
//! address the way Intel pseudo-locking or GPU L2 persistence do -
//! this reserves ways, it does not map bytes - but it is the honest
//! CPU-side residency knob, and it is what the scheduler reaches for
//! when a hot job must not share its L3 slice.
//!
//! The lever is capability-gated the same way the accelerator probes
//! are: [`CatCapability::detect`] reports what the running kernel and
//! CPU actually expose, and every operation returns
//! [`CatError::Unsupported`] where resctrl is absent (any non-Linux
//! host, or a Linux host without the feature mounted). Nothing here
//! panics on an unsupported platform.

use std::fmt;

/// resctrl mount point.
const RESCTRL: &str = "/sys/fs/resctrl";

/// What the running host exposes for L3 cache allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatCapability {
    /// resctrl is mounted and L3 allocation is present.
    pub supported: bool,
    /// Number of classes of service (allocatable partitions).
    pub num_closids: u32,
    /// Bits in the capacity bitmask - the number of L3 ways.
    pub cbm_bits: u32,
    /// Fewest contiguous bits a valid mask may set.
    pub min_cbm_bits: u32,
    /// Number of L3 cache domains (one per NUMA/CCD L3 instance).
    pub num_domains: u32,
}

impl CatCapability {
    /// Detect from `/sys/fs/resctrl/info/L3`. Returns an unsupported
    /// capability on any non-Linux host or where resctrl is not
    /// mounted.
    pub fn detect() -> Self {
        let unsupported = Self {
            supported: false,
            num_closids: 0,
            cbm_bits: 0,
            min_cbm_bits: 0,
            num_domains: 0,
        };
        let info = format!("{RESCTRL}/info/L3");
        if !std::path::Path::new(&info).is_dir() {
            return unsupported;
        }
        let read_u32 = |name: &str| -> u32 {
            std::fs::read_to_string(format!("{info}/{name}"))
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(0)
        };
        let cbm_bits = std::fs::read_to_string(format!("{info}/cbm_mask"))
            .ok()
            .and_then(|s| parse_cbm_mask_hex(s.trim()))
            .map(count_ways)
            .unwrap_or(0);
        let num_domains = std::fs::read_to_string(format!("{RESCTRL}/schemata"))
            .ok()
            .map(|s| count_l3_domains(&s))
            .unwrap_or(0);
        Self {
            supported: cbm_bits > 0,
            num_closids: read_u32("num_closids"),
            cbm_bits,
            min_cbm_bits: read_u32("min_cbm_bits").max(1),
            num_domains: num_domains.max(1),
        }
    }
}

/// Errors from the CAT lever.
#[derive(Debug)]
pub enum CatError {
    /// resctrl / L3 CAT is not available on this host.
    Unsupported,
    /// A requested way range does not fit the device's bitmask.
    InvalidRange {
        /// First way requested.
        first_way: u32,
        /// Way count requested.
        num_ways: u32,
        /// Total ways available.
        total_ways: u32,
    },
    /// A resctrl filesystem operation failed.
    Io(std::io::Error),
}

impl fmt::Display for CatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => write!(f, "resctrl / L3 CAT unavailable on this host"),
            Self::InvalidRange { first_way, num_ways, total_ways } => write!(
                f,
                "way range [{first_way}, {}) exceeds {total_ways} L3 ways",
                first_way + num_ways
            ),
            Self::Io(e) => write!(f, "resctrl I/O: {e}"),
        }
    }
}

impl std::error::Error for CatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// A live L3 reservation: a resctrl class of service holding a set of
/// L3 ways, with the current process bound to it. Dropping it removes
/// the class (re-homing the process to the default group).
pub struct L3Reservation {
    dir: std::path::PathBuf,
}

impl L3Reservation {
    /// Reserve `num_ways` contiguous L3 ways starting at `first_way`
    /// for the current process, in a resctrl group named `name`. The
    /// same mask is applied to every L3 domain. Returns
    /// [`CatError::Unsupported`] where resctrl is absent.
    pub fn reserve_ways(name: &str, first_way: u32, num_ways: u32) -> Result<Self, CatError> {
        let cap = CatCapability::detect();
        if !cap.supported {
            return Err(CatError::Unsupported);
        }
        if num_ways == 0 || first_way + num_ways > cap.cbm_bits {
            return Err(CatError::InvalidRange {
                first_way,
                num_ways,
                total_ways: cap.cbm_bits,
            });
        }
        let mask = contiguous_mask(first_way, num_ways);
        let dir = std::path::PathBuf::from(format!("{RESCTRL}/{name}"));
        // Creating the directory allocates a CLOSID.
        std::fs::create_dir_all(&dir).map_err(CatError::Io)?;

        // The new group's schemata is pre-populated with every domain
        // at full mask; rewrite each L3 domain to our mask.
        let current = std::fs::read_to_string(dir.join("schemata")).map_err(CatError::Io)?;
        let line = rewrite_l3_schemata(&current, mask);
        std::fs::write(dir.join("schemata"), line.as_bytes()).map_err(CatError::Io)?;

        // Bind this process.
        let pid = std::process::id();
        std::fs::write(dir.join("tasks"), format!("{pid}\n").as_bytes()).map_err(CatError::Io)?;

        Ok(Self { dir })
    }

    /// The reservation's current L3 schemata, read back from resctrl.
    pub fn schemata(&self) -> Result<String, CatError> {
        std::fs::read_to_string(self.dir.join("schemata")).map_err(CatError::Io)
    }
}

impl Drop for L3Reservation {
    fn drop(&mut self) {
        // rmdir re-homes bound tasks to the default group. Best effort.
        drop(std::fs::remove_dir(&self.dir));
    }
}

// -- pure helpers (unit-tested on every platform) --------------------

/// Contiguous capacity bitmask: `num_ways` bits set starting at
/// `first_way`. This is the CAT bitmask format - one bit per way.
fn contiguous_mask(first_way: u32, num_ways: u32) -> u64 {
    if num_ways == 0 {
        return 0;
    }
    let width = num_ways.min(64);
    let base = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    base << first_way
}

/// Count set bits in a capacity bitmask (the number of ways it holds).
fn count_ways(mask: u64) -> u32 {
    mask.count_ones()
}

/// Parse a hex capacity bitmask like "7fff" (no 0x prefix, as resctrl
/// writes it).
fn parse_cbm_mask_hex(s: &str) -> Option<u64> {
    let t = s.trim().trim_start_matches("0x");
    if t.is_empty() {
        return None;
    }
    u64::from_str_radix(t, 16).ok()
}

/// Count L3 domains in a schemata blob (the `;`-separated
/// `domain=mask` pairs on the `L3:` line).
fn count_l3_domains(schemata: &str) -> u32 {
    for line in schemata.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("L3:") {
            return rest.split(';').filter(|p| p.contains('=')).count() as u32;
        }
    }
    0
}

/// Rewrite every domain on the `L3:` line of a schemata blob to
/// `mask`, preserving domain ids. Non-L3 lines pass through unchanged.
fn rewrite_l3_schemata(schemata: &str, mask: u64) -> String {
    let mut out = String::new();
    for line in schemata.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("L3:") {
            let pairs: Vec<String> = rest
                .split(';')
                .filter_map(|p| p.split('=').next())
                .map(|dom| format!("{}={:x}", dom.trim(), mask))
                .collect();
            out.push_str("L3:");
            out.push_str(&pairs.join(";"));
            out.push('\n');
        } else if !trimmed.is_empty() {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_mask_places_the_right_ways() {
        assert_eq!(contiguous_mask(0, 4), 0x0f);
        assert_eq!(contiguous_mask(4, 4), 0xf0);
        assert_eq!(contiguous_mask(0, 1), 0x1);
        assert_eq!(contiguous_mask(0, 0), 0x0);
        assert_eq!(count_ways(contiguous_mask(2, 6)), 6);
    }

    #[test]
    fn cbm_mask_hex_parses() {
        assert_eq!(parse_cbm_mask_hex("7fff"), Some(0x7fff));
        assert_eq!(parse_cbm_mask_hex("0xff"), Some(0xff));
        assert_eq!(count_ways(0x7fff), 15);
        assert_eq!(parse_cbm_mask_hex(""), None);
    }

    #[test]
    fn domain_counting() {
        assert_eq!(count_l3_domains("L3:0=7fff;1=7fff\nMB:0=100;1=100\n"), 2);
        assert_eq!(count_l3_domains("L3:0=ffff\n"), 1);
        assert_eq!(count_l3_domains("MB:0=100\n"), 0);
    }

    #[test]
    fn schemata_rewrite_preserves_domains() {
        let before = "L3:0=7fff;1=7fff\nMB:0=100;1=100\n";
        let after = rewrite_l3_schemata(before, 0x00f0);
        assert!(after.contains("L3:0=f0;1=f0"), "got: {after}");
        // Non-L3 lines pass through untouched.
        assert!(after.contains("MB:0=100;1=100"));
    }

    #[test]
    fn detect_is_unsupported_off_linux_or_unmounted() {
        // On this dev box (Windows) resctrl does not exist, so detect
        // must report unsupported without panicking. On a Linux host
        // without resctrl mounted, likewise.
        let cap = CatCapability::detect();
        if cfg!(not(target_os = "linux")) {
            assert!(!cap.supported, "resctrl cannot exist off Linux");
            assert_eq!(cap.cbm_bits, 0);
        }
    }

    #[test]
    fn reserve_is_unsupported_when_capability_absent() {
        if !CatCapability::detect().supported {
            assert!(matches!(
                L3Reservation::reserve_ways("flynnel_test", 0, 4),
                Err(CatError::Unsupported)
            ));
        }
    }
}
