//! The cost of a cross-block barrier written inside a user op and run once
//! per generation, at team sizes up to the width of the card.
//!
//! - `gen_wait_max_ns`: the longest any block's thread 0 waited at a
//!   generation barrier in the slot, which is the latency one generation
//!   pays.
//! - `gen0_wait_max_ns`: the same for generation 0 alone, which also
//!   absorbs any skew in when the blocks began the slot.
//! - `wait_sum_ns`: every block's wait over every generation, accumulated
//!   on the device in units of 1024 ns.
//! - `timeouts`: barriers a block left on its deadline. A nonzero count
//!   voids that slot's waits, and the summary leaves such slots out.
//! - `kernel_barrier_max_ns`: the kernel's own once-per-slot barrier, read
//!   through `barrier_wait_max_ns`.
//!
//! Each generation every thread spins `work_us`, plus up to
//! `spread_permille` of it again for the highest rank. The counters live in
//! a resident block (`vram`) or in the host-mapped slot payload (`mapped`).
//! Arms run in a forward pass and a reverse pass; every slot goes to the
//! CSV and the summary prints minimum, median and maximum.
//!
//! A user op runs to completion once it starts. The watchdog that applies
//! to the device is detected at startup, and the generation count is
//! lowered, with a printed note, if a slot whose every barrier expires
//! could run past half of its delay.
//!
//! ```sh
//! cargo run --release --example gpu_peer_generation_barrier -- barrier.csv 16 64 auto
//! ```

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;
use std::time::{Duration, Instant};

use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, ResidentHandle, STATUS_DONE, layout};

/// The probe's opcode. Substituted into the device source for the
/// `PROBE_OP` token, so the two cannot disagree.
const PROBE_OP: u32 = layout::OP_USER_BASE + 50;

/// Argument and result space the op is handed, in bytes.
const ARGS_BYTES: usize = 64;

/// The kernel hands the op its payload advanced past this many bytes of
/// resident parameters, so host offsets into a read-back slot are shifted
/// by it.
const RESIDENT_PREFIX: usize = 8;

/// Argument offsets in the op's payload.
const ARG_GENERATIONS: usize = 0;
const ARG_WORK_US: usize = 4;
const ARG_SPREAD_PERMILLE: usize = 8;
const ARG_PLACE: usize = 12;
const ARG_DEADLINE_US: usize = 16;
/// Rank 0's thread 0 writes the op's own duration here.
const RESULT_TOTAL_NS: usize = 20;
/// Where the counter cells start when they live in the mapped payload.
const MAPPED_CELLS_AT: usize = 32;

/// Counter cells, four bytes each, from the start of wherever they live.
/// Cell 0 is the arrival count the barrier polls.
const CELL_WAIT_MAX: usize = 1;
const CELL_WAIT_SUM_KNS: usize = 2;
const CELL_TIMEOUTS: usize = 3;
const CELL_GEN0_WAIT_MAX: usize = 4;

/// How long a block waits at one generation barrier before leaving it.
/// Forty times the 55 us the kernel's slot barrier measured for a
/// 64-block team in `tests/gpu_peer_team.rs`.
const GEN_DEADLINE_US: u32 = 2_000;

/// Per-generation work in microseconds, and the extra share the highest
/// rank does, in thousandths.
const WORKS_US: [u32; 2] = [0, 250];
const SPREADS_PERMILLE: [u32; 2] = [0, 1_000];

/// Windows' documented TDR defaults when the registry carries no value:
/// level 3 recovers the adapter, after a delay of 2 seconds.
const TDR_DEFAULT_LEVEL: u32 = 3;
const TDR_DEFAULT_DELAY_S: u32 = 2;

/// The device side. Every thread of every block runs it; `__syncthreads`
/// keeps a block's threads together, and the atomic arrival count in cell
/// 0 is the only thing that brings separate blocks together.
const PROBE_SOURCE: &str = r#"
__device__ __forceinline__ unsigned long long probe_now()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)count;
    if (op != PROBE_OPu) return 1u;

    volatile unsigned* args = (volatile unsigned*)payload;
    unsigned gens = args[0];
    unsigned work_us = args[1];
    unsigned spread = args[2];
    unsigned place = args[3];
    unsigned deadline_us = args[4];

    unsigned* cells;
    if (place == 0u) {
        if (block == (unsigned char*)0) return 1u;
        cells = (unsigned*)block;
    } else {
        cells = (unsigned*)(payload + 32);
    }

    unsigned denom = team_size > 1u ? team_size - 1u : 1u;
    unsigned my_us = work_us + ((work_us * spread) / 1000u) * team_rank / denom;
    unsigned long long my_ns = (unsigned long long)my_us * 1000ull;
    unsigned long long deadline_ns = (unsigned long long)deadline_us * 1000ull;
    unsigned long long t_op = probe_now();

    for (unsigned g = 0u; g < gens; g++) {
        if (my_ns != 0ull) {
            unsigned long long t_work = probe_now();
            while (probe_now() - t_work < my_ns) {
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned target = team_size * (g + 1u);
            unsigned long long t_arrive = probe_now();
            atomicAdd(cells, 1u);
            unsigned whole = 1u;
            while (atomicAdd(cells, 0u) < target) {
                if (probe_now() - t_arrive > deadline_ns) { whole = 0u; break; }
            }
            unsigned long long waited = probe_now() - t_arrive;
            unsigned w = (unsigned)(waited > 0xFFFFFFFFull ? 0xFFFFFFFFull : waited);
            atomicMax(cells + 1, w);
            atomicAdd(cells + 2, (unsigned)(waited >> 10));
            if (g == 0u) atomicMax(cells + 4, w);
            if (whole == 0u) atomicAdd(cells + 3, 1u);
        }
        __syncthreads();
    }

    if (team_rank == 0u && threadIdx.x == 0) {
        unsigned long long total = probe_now() - t_op;
        args[5] = (unsigned)(total > 0xFFFFFFFFull ? 0xFFFFFFFFull : total);
    }
    return 0u;
}
"#;

/// An argument that was not supplied takes the default. One that was
/// supplied and does not parse stops the run and says so, so a typo cannot
/// select the default while the reader credits the value they typed.
fn arg<T>(n: usize, default: T) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match env::args().nth(n) {
        None => default,
        Some(text) => match text.parse() {
            Ok(value) => value,
            Err(err) => {
                eprintln!("argument {n} is not a valid value: {text:?} ({err})");
                std::process::exit(2);
            }
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    Vram,
    Mapped,
}

impl Place {
    fn code(self) -> u32 {
        match self {
            Place::Vram => 0,
            Place::Mapped => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Place::Vram => "vram",
            Place::Mapped => "mapped",
        }
    }
}

#[derive(Clone, Copy)]
struct Arm {
    team: u32,
    place: Place,
    work_us: u32,
    spread_permille: u32,
}

/// One slot's readings, as the CSV records them.
struct Slot {
    status: u32,
    gen_wait_max_ns: u32,
    gen0_wait_max_ns: u32,
    wait_sum_ns: u64,
    timeouts: u32,
    op_total_ns: u32,
    host_rtt_us: u128,
}

/// The watchdog that resets the device when one op runs too long.
struct Watchdog {
    /// Its delay, when one applies.
    limit_ns: Option<u64>,
    /// What was read to reach that, printed with the run.
    basis: String,
}

fn put(buf: &mut [u8], at: usize, value: u32) {
    buf[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn get(buf: &[u8], at: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&buf[at..at + 4]);
    u32::from_le_bytes(word)
}

/// A command's standard output, or why there is none.
fn command_output(program: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| format!("{program} did not run: {err}"))?;
    if !out.status.success() {
        return Err(format!("{program} exited {}", out.status));
    }
    String::from_utf8(out.stdout)
        .map_err(|err| format!("{program} printed output that is not UTF-8 ({err})"))
}

/// The driver model and display state nvidia-smi reports for device 0,
/// such as `("WDDM", "Enabled")`.
fn driver_model() -> Result<(String, String), String> {
    let text = command_output(
        "nvidia-smi",
        &[
            "-i",
            "0",
            "--query-gpu=driver_model.current,display_active",
            "--format=csv,noheader",
        ],
    )?;
    let line = text
        .lines()
        .next()
        .ok_or_else(|| "nvidia-smi printed nothing".to_string())?;
    let mut fields = line.split(',').map(|field| field.trim().to_string());
    match (fields.next(), fields.next()) {
        (Some(model), Some(display)) => Ok((model, display)),
        (first, second) => Err(format!(
            "nvidia-smi printed {line:?}, which gave {first:?} and {second:?} rather than two fields"
        )),
    }
}

/// Windows' TDR level and delay in seconds, each taking its documented
/// default when the registry carries no value.
#[cfg(windows)]
fn tdr_settings() -> Result<(u32, u32), String> {
    let text = command_output(
        "reg",
        &["query", r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"],
    )?;
    let mut level = TDR_DEFAULT_LEVEL;
    let mut delay = TDR_DEFAULT_DELAY_S;
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[1] != "REG_DWORD" {
            continue;
        }
        let value = u32::from_str_radix(fields[2].trim_start_matches("0x"), 16)
            .map_err(|err| format!("registry value {line:?} did not parse ({err})"))?;
        if fields[0] == "TdrLevel" {
            level = value;
        } else if fields[0] == "TdrDelay" {
            delay = value;
        }
    }
    Ok((level, delay))
}

/// TDR covers WDDM and MCDM devices unless its level is 0, and does not
/// cover TCC devices. When the settings cannot be read, the documented
/// default delay is taken and the basis says so.
#[cfg(windows)]
fn detect_watchdog() -> Watchdog {
    let (model_text, is_tcc) = match driver_model() {
        Ok((name, display)) => {
            let tcc = name == "TCC";
            (format!("driver model {name}, display {display}"), tcc)
        }
        Err(err) => (format!("driver model unknown ({err})"), false),
    };
    if is_tcc {
        return Watchdog {
            limit_ns: None,
            basis: format!("{model_text}; TCC devices are outside TDR"),
        };
    }
    match tdr_settings() {
        Ok((0, delay)) => Watchdog {
            limit_ns: None,
            basis: format!("{model_text}; TdrLevel 0 disables detection (TdrDelay {delay} s)"),
        },
        Ok((level, delay)) => Watchdog {
            limit_ns: Some(u64::from(delay) * 1_000_000_000),
            basis: format!("{model_text}; TdrLevel {level}, TdrDelay {delay} s"),
        },
        Err(err) => Watchdog {
            limit_ns: Some(u64::from(TDR_DEFAULT_DELAY_S) * 1_000_000_000),
            basis: format!(
                "{model_text}; TDR settings unreadable ({err}), so the documented \
                 {TDR_DEFAULT_DELAY_S} s default is taken (level {TDR_DEFAULT_LEVEL})"
            ),
        },
    }
}

#[cfg(not(windows))]
fn detect_watchdog() -> Watchdog {
    let model_text = match driver_model() {
        Ok((name, display)) => format!("driver model {name}, display {display}"),
        Err(err) => format!("driver model unknown ({err})"),
    };
    Watchdog {
        limit_ns: None,
        basis: format!("{model_text}; no watchdog is known on this platform"),
    }
}

/// The heaviest arm's per-generation work: the largest work share plus the
/// largest spread of it.
fn max_work_us() -> u32 {
    let work = WORKS_US.iter().copied().fold(0, u32::max);
    let spread = SPREADS_PERMILLE.iter().copied().fold(0, u32::max);
    work + work * spread / 1_000
}

/// The longest one slot can run: every generation's barrier expiring after
/// the heaviest arm's work, then the kernel's slot barrier expiring.
fn worst_slot_ns(generations: u32, kernel_deadline_ns: u64) -> u64 {
    let per_generation_ns = u64::from(GEN_DEADLINE_US + max_work_us()) * 1_000;
    u64::from(generations) * per_generation_ns + kernel_deadline_ns
}

/// Powers of two from one block up to twice the card's SM count, so the
/// sweep reaches the card's width and one step past it.
fn auto_teams(sm_count: u32) -> Vec<u32> {
    let cap = sm_count.saturating_mul(2);
    let mut teams = Vec::new();
    let mut team = 1u32;
    while team <= cap {
        teams.push(team);
        match team.checked_mul(2) {
            Some(next) => team = next,
            None => break,
        }
    }
    teams
}

fn parse_teams(text: &str, sm_count: Option<u32>) -> Result<Vec<u32>, String> {
    if text == "auto" {
        return match sm_count {
            Some(count) => Ok(auto_teams(count)),
            None => Err("the device's SM count could not be read, so the width of the card \
                         is unknown; pass the team sizes explicitly, such as 1,2,4,8"
                .to_string()),
        };
    }
    text.split(',')
        .map(|part| {
            let n = part
                .trim()
                .parse::<u32>()
                .map_err(|err| format!("team size {part:?} is not a count ({err})"))?;
            if n == 0 {
                Err("a team of zero blocks runs nothing".to_string())
            } else {
                Ok(n)
            }
        })
        .collect()
}

fn make_peer(team: u32) -> Result<(GpuPeer, ResidentHandle), String> {
    let source = PROBE_SOURCE.replace("PROBE_OP", &PROBE_OP.to_string());
    let mut peer = GpuPeer::init(GpuPeerConfig {
        lanes: 1,
        slots_per_lane: 8,
        vram_block_bytes: 4_096,
        vram_blocks: 4,
        blocks_per_lane: team,
        user_ops_cuda: Some(source),
        ..GpuPeerConfig::default()
    })
    .map_err(|err| format!("peer for team {team} did not initialize: {err}"))?;
    let handle = peer
        .pin_bulk(&[0u8; ARGS_BYTES])
        .map_err(|err| format!("counter block for team {team} was not pinned: {err}"))?;
    Ok((peer, handle))
}

fn run_slot(
    peer: &mut GpuPeer,
    handle: &ResidentHandle,
    arm: Arm,
    generations: u32,
) -> Result<Slot, String> {
    let mut args = [0u8; ARGS_BYTES];
    put(&mut args, ARG_GENERATIONS, generations);
    put(&mut args, ARG_WORK_US, arm.work_us);
    put(&mut args, ARG_SPREAD_PERMILLE, arm.spread_permille);
    put(&mut args, ARG_PLACE, arm.place.code());
    put(&mut args, ARG_DEADLINE_US, GEN_DEADLINE_US);

    if arm.place == Place::Vram {
        peer.write_resident_bulk(handle, &[0u8; ARGS_BYTES])
            .map_err(|err| format!("resetting the counters: {err}"))?;
    }

    let started = Instant::now();
    let ticket = peer
        .submit_user(PROBE_OP, Some(handle), &args)
        .map_err(|err| format!("submit: {err}"))?;
    let status = peer
        .wait_status(ticket, Duration::from_secs(10))
        .map_err(|err| format!("wait: {err}"))?;
    let host_rtt_us = started.elapsed().as_micros();

    let mut result = [0u8; RESIDENT_PREFIX + ARGS_BYTES];
    peer.read_result(ticket, &mut result)
        .map_err(|err| format!("read result: {err}"))?;
    peer.reap(ticket).map_err(|err| format!("reap: {err}"))?;

    let mut cells = [0u8; ARGS_BYTES];
    match arm.place {
        Place::Vram => peer
            .fetch_bulk(handle, &mut cells)
            .map_err(|err| format!("reading the counters: {err}"))?,
        Place::Mapped => {
            let from = RESIDENT_PREFIX + MAPPED_CELLS_AT;
            let len = ARGS_BYTES - MAPPED_CELLS_AT;
            cells[..len].copy_from_slice(&result[from..from + len]);
        }
    }

    Ok(Slot {
        status,
        gen_wait_max_ns: get(&cells, CELL_WAIT_MAX << 2),
        gen0_wait_max_ns: get(&cells, CELL_GEN0_WAIT_MAX << 2),
        wait_sum_ns: u64::from(get(&cells, CELL_WAIT_SUM_KNS << 2)) << 10,
        timeouts: get(&cells, CELL_TIMEOUTS << 2),
        op_total_ns: get(&result, RESIDENT_PREFIX + RESULT_TOTAL_NS),
        host_rtt_us,
    })
}

/// Minimum, median and maximum of a set of readings.
struct Spread {
    text: String,
    median: String,
}

fn spread_of(values: &[u64]) -> Spread {
    if values.is_empty() {
        return Spread { text: "none".to_string(), median: "none".to_string() };
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() >> 1];
    Spread {
        text: format!("{}/{median}/{}", sorted[0], sorted[sorted.len() - 1]),
        median: median.to_string(),
    }
}

fn main() {
    let csv_path: String = arg(1, "gpu_peer_generation_barrier.csv".to_string());
    let reps: u32 = arg(2, 16);
    let requested_generations: u32 = arg(3, 64);
    let teams_text: String = arg(4, "auto".to_string());

    if requested_generations == 0 {
        eprintln!("generations must be at least 1");
        std::process::exit(2);
    }

    let sm_count = flynnel::backend::detect::cuda_sm_count(0);
    let teams = match parse_teams(&teams_text, sm_count) {
        Ok(teams) => teams,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    let watchdog = detect_watchdog();
    let kernel_deadline_ns = GpuPeerConfig::default().barrier_deadline_ns;
    let mut generations = requested_generations;
    if let Some(limit_ns) = watchdog.limit_ns {
        let allowed_ns = limit_ns >> 1;
        while generations > 1 && worst_slot_ns(generations, kernel_deadline_ns) > allowed_ns {
            generations -= 1;
        }
    }

    println!("watchdog: {}", watchdog.basis);
    if generations != requested_generations {
        println!(
            "generations lowered from {requested_generations} to {generations}, so a slot \
             whose every barrier expires stays under half the watchdog delay"
        );
    }
    let sm_text = match sm_count {
        Some(n) => n.to_string(),
        None => "unreadable".to_string(),
    };
    println!(
        "sm_count {sm_text}  teams {teams:?}  reps {reps} (+1 warm)  generations {generations}  \
         gen_deadline_us {GEN_DEADLINE_US}  worst_slot_ns {}",
        worst_slot_ns(generations, kernel_deadline_ns),
    );

    let mut arms = Vec::new();
    for &team in &teams {
        for place in [Place::Vram, Place::Mapped] {
            for work_us in WORKS_US {
                for spread_permille in SPREADS_PERMILLE {
                    if work_us == 0 && spread_permille != 0 {
                        continue;
                    }
                    arms.push(Arm { team, place, work_us, spread_permille });
                }
            }
        }
    }

    let file = match File::create(&csv_path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("cannot create {csv_path}: {err}");
            std::process::exit(2);
        }
    };
    let mut csv = BufWriter::new(file);
    let header = "pass,team,place,work_us,spread_permille,rep,warm,status,gen_wait_max_ns,\
                  gen0_wait_max_ns,wait_sum_ns,timeouts,op_total_ns,host_rtt_us";
    if let Err(err) = writeln!(csv, "{header}") {
        eprintln!("cannot write {csv_path}: {err}");
        std::process::exit(2);
    }

    println!(
        "pass     team  place   work_us  spread  gen_wait_max_ns(min/med/max)  \
         gen0_wait_max_ns(min/med/max)  op_total_ns(med)  timeouts  not_done  errors  \
         kernel_barrier_max_ns  kernel_stalls"
    );

    let mut peers: Vec<Option<(GpuPeer, ResidentHandle)>> = teams.iter().map(|_| None).collect();
    // Teams whose peer could not be made. Their arms are reported once and
    // skipped, and every other team still runs.
    let mut unavailable: Vec<u32> = Vec::new();

    for pass in ["forward", "reverse"] {
        let order: Vec<Arm> = if pass == "forward" {
            arms.clone()
        } else {
            arms.iter().rev().copied().collect()
        };

        for arm in order {
            if unavailable.contains(&arm.team) {
                continue;
            }
            let Some(index) = teams.iter().position(|&t| t == arm.team) else {
                unreachable!("every arm's team comes from the team list");
            };
            if peers[index].is_none() {
                match make_peer(arm.team) {
                    Ok(made) => peers[index] = Some(made),
                    Err(err) => {
                        println!("{pass:<7}  {:>5}  {err}", arm.team);
                        unavailable.push(arm.team);
                        continue;
                    }
                }
            }
            let Some((peer, handle)) = peers[index].as_mut() else {
                unreachable!("the peer for this team was made just above");
            };

            let mut waits = Vec::new();
            let mut gen0_waits = Vec::new();
            let mut totals = Vec::new();
            let mut timeouts = 0u32;
            let mut not_done = 0u32;
            let mut errors = 0u32;

            for rep in 0..=reps {
                let warm = rep == 0;
                let slot = match run_slot(peer, handle, arm, generations) {
                    Ok(slot) => slot,
                    Err(err) => {
                        println!(
                            "{pass:<7}  {:>5}  {:<6}  rep {rep}: {err}",
                            arm.team,
                            arm.place.name()
                        );
                        errors += 1;
                        continue;
                    }
                };
                let row = writeln!(
                    csv,
                    "{pass},{},{},{},{},{rep},{},{},{},{},{},{},{},{}",
                    arm.team,
                    arm.place.name(),
                    arm.work_us,
                    arm.spread_permille,
                    u8::from(warm),
                    slot.status,
                    slot.gen_wait_max_ns,
                    slot.gen0_wait_max_ns,
                    slot.wait_sum_ns,
                    slot.timeouts,
                    slot.op_total_ns,
                    slot.host_rtt_us,
                );
                if let Err(err) = row {
                    eprintln!("cannot write {csv_path}: {err}");
                    std::process::exit(2);
                }

                timeouts += slot.timeouts;
                if slot.status != STATUS_DONE {
                    not_done += 1;
                }
                if !warm && slot.status == STATUS_DONE && slot.timeouts == 0 {
                    waits.push(u64::from(slot.gen_wait_max_ns));
                    gen0_waits.push(u64::from(slot.gen0_wait_max_ns));
                    totals.push(u64::from(slot.op_total_ns));
                }
            }

            let (stalls, stall_depth) = peer.barrier_stalls();
            let wait_spread = spread_of(&waits).text;
            let gen0_spread = spread_of(&gen0_waits).text;
            let total_median = spread_of(&totals).median;
            println!(
                "{pass:<7}  {:>5}  {:<6}  {:>7}  {:>6}  {wait_spread:>28}  {gen0_spread:>29}  \
                 {total_median:>16}  {timeouts:>8}  {not_done:>8}  {errors:>6}  {:>21}  \
                 {stalls:>6} (depth {stall_depth})",
                arm.team,
                arm.place.name(),
                arm.work_us,
                arm.spread_permille,
                peer.barrier_wait_max_ns(),
            );
        }
    }

    if let Err(err) = csv.flush() {
        eprintln!("cannot flush {csv_path}: {err}");
        std::process::exit(2);
    }
    if unavailable.is_empty() {
        println!("done; every team ran every arm");
    } else {
        println!("done; no peer could be made for teams {unavailable:?}");
    }
}
