//! Every rank of a block team runs the op, and its work reaches the
//! caller.
//!
//! `blocks_per_lane > 1` gives each lane a team of blocks: rank 0 owns
//! the descriptor and the ring, and every rank runs the user op so a
//! doorbell op can use more of the device than the single SM one block
//! occupies. Nothing exercised that path - a search for
//! `blocks_per_lane` across the test tree returned no hits at all,
//! while a consumer runs teams in production.
//!
//! The property that matters is not that a team submission completes.
//! It is that every rank's contribution is present when it does, since
//! rank 0 retires the slot on a deadline and a team that fell short
//! would otherwise return a payload with holes in it.
//!
//! Requires a CUDA device and NVRTC.
#![cfg(feature = "gpu-peer")]

use std::time::Duration;

use flynnel::gpu_peer::{
    GpuPeer, GpuPeerConfig, STATUS_DONE, STATUS_ERR, STATUS_TEAM_INCOMPLETE, layout,
};

mod common;

/// op 200 marks the payload byte at its own rank, so a result carries
/// one distinguishable mark per rank that ran; one thread per block
/// writes and the ranks write to disjoint bytes. op 201 holds every rank
/// above 0 past any sane deadline, so rank 0's barrier must expire.
const TEAM_OPS: &str = r#"
extern "C" __device__ unsigned flynnel_user_op(
    unsigned op, unsigned char* block, unsigned count,
    volatile unsigned char* payload,
    unsigned team_rank, unsigned team_size)
{
    (void)block; (void)count; (void)team_size;
    if (op == 200u) {
        if (threadIdx.x == 0) payload[team_rank] = (unsigned char)(team_rank + 1u);
        return 0u;
    }
    if (op == 201u) {
        if (team_rank != 0u && threadIdx.x == 0) {
            unsigned long long t0, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - t0 < 400000000ull);
        }
        return 0u;
    }
    return 1u;
}
"#;

/// Bytes of resident-parameter block the kernel skips before handing a
/// user op its payload pointer, so host-side offsets are shifted by it.
const RESIDENT_PREFIX: usize = 8;

/// Mark bytes the payload carries, one per rank. Wider than the widest
/// team tested, so the bytes above the team are a guard against a rank
/// that does not exist writing one.
const MARKS: usize = 128;

fn team_peer(blocks_per_lane: u32) -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(TEAM_OPS.to_string()),
        blocks_per_lane,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

/// A peer whose barrier deadline is short enough to provoke on purpose,
/// so a timeout is reachable inside a test rather than after the
/// default wait.
fn impatient_team_peer(blocks_per_lane: u32, barrier_deadline_ns: u64) -> GpuPeer {
    GpuPeer::init(GpuPeerConfig {
        user_ops_cuda: Some(TEAM_OPS.to_string()),
        blocks_per_lane,
        barrier_deadline_ns,
        ..GpuPeerConfig::default()
    })
    .expect("a CUDA device and NVRTC are required for this test")
}

/// One submission per team size, each asserting that every rank ran.
///
/// The sizes are the ones consumers configure: 1 is the ungrouped
/// default and the control, 2 is what an index scanner runs, 4 and 8
/// are the range a nearest-neighbour consumer swept, and 64 is what a
/// desktop compositor's field op sets in production. A hole at any rank
/// means the barrier retired a slot the whole team had not finished
/// writing.
#[test]
fn every_rank_of_a_team_contributes_to_the_result() {
    let _device = common::device();
    for team in [1u32, 2, 4, 8, 64] {
        let mut peer = team_peer(team);
        // A request wider than the device runs at its SM count, so the
        // marks expected are those of the team size the peer reports.
        let used = peer.team_size();
        assert!(used >= 1 && used <= team, "team {team}: reported team size {used}");
        // One byte per rank past the resident prefix, with room for the
        // widest team here and a margin above it that must stay zero.
        let mut payload = vec![0u8; RESIDENT_PREFIX + MARKS];

        let t = peer
            .submit_user(layout::OP_USER_BASE + 100, None, &payload)
            .expect("submit the team op");
        let status = peer.wait(t, Duration::from_secs(10)).expect("the team completes");
        assert_eq!(
            status, STATUS_DONE,
            "team {team}: a slot every rank finished must retire DONE, not ERR"
        );

        peer.read_result(t, &mut payload).expect("the result fits the slot");
        peer.reap(t).expect("reap");

        // The kernel hands a user op the payload advanced past the
        // 8-byte resident-parameter block, so a rank writing its own
        // index 0 lands at byte 8 of what the host reads back.
        let marks = &payload[RESIDENT_PREFIX..RESIDENT_PREFIX + MARKS];
        for rank in 0..used as usize {
            assert_eq!(
                marks[rank],
                (rank + 1) as u8,
                "team {team} running {used}: rank {rank} left no mark, so the slot \
                 retired without its contribution. Marks seen: {:?}",
                &marks[..used as usize]
            );
        }
        // Nothing beyond the team wrote, so a mark cannot come from a
        // rank that does not exist.
        assert!(
            marks[used as usize..].iter().all(|&b| b == 0),
            "team {team} running {used}: a byte past the last rank was written: {marks:?}"
        );
    }
}

/// A team that does not assemble reports a lost rank, distinctly from
/// an op that failed.
///
/// Rank 0 retires the slot on a deadline of one whole quantum. Until
/// this test the guard had never been made to fire anywhere in the
/// tree - it was observed once in a consumer's production sweep, where
/// it was indistinguishable from their own op reporting an error,
/// because both were STATUS_ERR.
///
/// The op holds every rank above 0 for 400 ms against a 20 ms quantum,
/// so the barrier must expire. The status must say the team was
/// incomplete rather than that the op failed: the op returned success
/// on every rank that got to return at all.
#[test]
fn a_team_that_does_not_assemble_reports_an_incomplete_team_not_an_error() {
    let _device = common::device();
    let mut peer = impatient_team_peer(4, 20_000_000);
    let payload = vec![0u8; 64];
    let t = peer
        .submit_user(layout::OP_USER_BASE + 101, None, &payload)
        .expect("submit the stalling op");

    let status = peer
        .wait_status(t, Duration::from_secs(20))
        .expect("the slot retires on the deadline rather than hanging");
    assert_eq!(
        status,
        STATUS_TEAM_INCOMPLETE,
        "a barrier that expired must name the lost rank as the cause; \
         STATUS_ERR here would be indistinguishable from the op failing"
    );
    assert_ne!(status, STATUS_ERR, "the op itself returned success on every rank");
    peer.reap(t).expect("a timed-out slot is still reapable");
}

/// A lane keeps serving correctly after one of its teams misses the
/// deadline.
///
/// This is the case a consumer actually lives in: their peer timed out
/// once and then answered 607 more queries on the same lanes. Rank 0
/// resets the arrival counter before retiring a slot it gave up on,
/// while the late ranks are still running and will increment it when
/// they finish. If that stray arrival carried into the next slot's
/// barrier, the lane would be wrong after any timeout rather than only
/// during one - and nothing would say so, because the following slot
/// would report DONE.
///
/// The wait between the two submissions is the stalling op's own
/// duration plus margin, so the late ranks have certainly finished and
/// their arrivals have certainly landed somewhere before the second
/// submission is judged.
#[test]
fn a_lane_serves_correctly_after_one_of_its_teams_times_out() {
    let _device = common::device();
    const TEAM: u32 = 4;
    let mut peer = impatient_team_peer(TEAM, 20_000_000);

    let stalled = peer
        .submit_user_on_lane(layout::OP_USER_BASE + 101, None, &[0u8; 64], 0)
        .expect("submit the stalling op");
    assert_eq!(
        peer.wait_status(stalled, Duration::from_secs(20)).expect("retires"),
        STATUS_TEAM_INCOMPLETE,
        "the first slot must be the timeout this test is recovering from"
    );
    peer.reap(stalled).expect("reap the timed-out slot");

    // The op holds late ranks for 400 ms; wait past that so their
    // arrivals are not still in flight when the next slot is judged.
    std::thread::sleep(Duration::from_millis(700));

    let mut payload = vec![0u8; 64];
    let good = peer
        .submit_user_on_lane(layout::OP_USER_BASE + 100, None, &payload, 0)
        .expect("submit the marking op on the same lane");
    assert_eq!(
        peer.wait_status(good, Duration::from_secs(20)).expect("retires"),
        STATUS_DONE,
        "the lane must recover: a slot after a timeout is not itself a failure"
    );
    peer.read_result(good, &mut payload).expect("read");
    peer.reap(good).expect("reap");

    let marks = &payload[RESIDENT_PREFIX..RESIDENT_PREFIX + 16];
    for rank in 0..TEAM as usize {
        assert_eq!(
            marks[rank],
            (rank + 1) as u8,
            "after a timeout on this lane, rank {rank} is missing from the \
             next slot. Marks seen: {:?}",
            &marks[..TEAM as usize]
        );
    }
}

/// A slot submitted while a previous team is still unwinding never
/// reports success with a rank missing.
///
/// The recovery case above waits for the stalled ranks to finish, which
/// makes it deterministic and leaves the harder case open: a consumer
/// submitting continuously puts the next slot on the lane while the
/// abandoned ranks are still running, and their arrivals land on a
/// counter rank 0 has already zeroed for the new slot.
///
/// The assertion is deliberately one-sided, because the outcome depends
/// on timing this test does not control. A slot may legitimately fail -
/// its own team can miss the deadline while the lane is congested. What
/// it may never do is report DONE while a rank's mark is absent, which
/// is the silent corruption a stray arrival would produce: the barrier
/// satisfied by someone else's count.
///
/// # Which split this reaches, and which it does not
///
/// A rank held inside a slow op is LATE: it has claimed the slot and
/// will arrive at the barrier eventually. That is what the stalling op
/// produces and what these cases cover.
///
/// The split the poller loop actually permits is different. Each rank
/// tests the quantum at the top of its loop, before claiming anything,
/// so a boundary falling between two ranks' checks leaves one of them
/// breaking out without ever joining that slot - gone rather than late,
/// with no arrival to come. Rank 0 then spends its deadline on a rank
/// that already left, and the lane cannot be relaunched until rank 0
/// leaves too, since the host waits for the exit count to reach the
/// team size.
///
/// Reaching that from a test needs control over when ranks evaluate the
/// boundary relative to a submission, which nothing here has. So the
/// deadline is exercised, and the production route to it is not.
#[test]
fn a_slot_behind_an_unwinding_team_never_reports_done_with_a_rank_missing() {
    let _device = common::device();
    const TEAM: u32 = 4;
    let mut peer = impatient_team_peer(TEAM, 20_000_000);

    let stalled = peer
        .submit_user_on_lane(layout::OP_USER_BASE + 101, None, &[0u8; 64], 0)
        .expect("submit the stalling op");
    assert_eq!(
        peer.wait_status(stalled, Duration::from_secs(20)).expect("retires"),
        STATUS_TEAM_INCOMPLETE,
    );
    peer.reap(stalled).expect("reap the timed-out slot");

    // No wait: the abandoned ranks are still spinning out their 400 ms.
    for round in 0..4 {
        let mut payload = vec![0u8; 64];
        let t = peer
            .submit_user_on_lane(layout::OP_USER_BASE + 100, None, &payload, 0)
            .expect("submit while the previous team unwinds");
        let status = peer.wait_status(t, Duration::from_secs(20)).expect("retires");
        peer.read_result(t, &mut payload).expect("read");
        peer.reap(t).expect("reap");

        if status == STATUS_DONE {
            let marks = &payload[RESIDENT_PREFIX..RESIDENT_PREFIX + TEAM as usize];
            for rank in 0..TEAM as usize {
                assert_eq!(
                    marks[rank],
                    (rank + 1) as u8,
                    "round {round}: the slot reported DONE while rank {rank} left no \
                     mark, so its barrier was satisfied by a rank from the previous \
                     team. Marks seen: {marks:?}"
                );
            }
        }
    }
}

/// A barrier expiry is counted, and the ring depth at that moment is
/// recorded.
///
/// The count alone says a team was lost; it does not say why. Two
/// mechanisms produce the same status. A quantum boundary landing
/// between two ranks' clock reads splits whatever the lane was holding,
/// so a lightly fed lane shows a depth of one or two. A lane relaunched
/// late and draining a backlog claims from a full ring and shows a
/// depth near the ring size. Only the depth separates them, and the
/// host cannot recover it after the fact because the ring has moved on
/// by the time a status is read.
///
/// This case produces the first kind deliberately - one submission on
/// an idle lane - so the depth it records must be small. A build where
/// this reported a large depth for a single queued slot would mean the
/// instrument is measuring something other than what it names.
#[test]
fn a_barrier_expiry_records_its_count_and_the_ring_depth() {
    let _device = common::device();
    let mut peer = impatient_team_peer(4, 20_000_000);
    assert_eq!(peer.barrier_stalls(), (0, 0), "a fresh peer has stalled nothing");

    let t = peer
        .submit_user(layout::OP_USER_BASE + 101, None, &[0u8; 64])
        .expect("submit the stalling op");
    assert_eq!(
        peer.wait_status(t, Duration::from_secs(20)).expect("retires"),
        STATUS_TEAM_INCOMPLETE,
    );
    peer.reap(t).expect("reap");

    let (count, depth) = peer.barrier_stalls();
    assert!(count >= 1, "the expiry that produced the status must be counted; saw {count}");
    assert!(
        depth <= 4,
        "one slot on an otherwise idle lane is a shallow ring; a depth of \
         {depth} would mean this is not measuring queue depth"
    );
}

/// What a healthy team costs at the barrier, against the deadline it is
/// given.
///
/// The default deadline rests on this margin: a team that assembles
/// normally must stay orders below it, or shortening it would abandon
/// teams that would have arrived. Sizes 2 through 8 cost single-digit
/// microseconds on the host measured here. 64 is included because a
/// consumer configures it, the cost grows with the team, and the
/// default deadline was 250 ms when that consumer chose the number: a
/// 5 ms deadline is a change under an already-shipped setting,
/// and the margin at the size actually used is the thing that decides
/// whether the change is safe.
///
/// The number is the output and is printed. What is asserted is a ten
/// times margin, which is the property that would break first on
/// hardware slower than this one.
///
/// The 64-block case is what the bound has to survive: 2, 4 and 8 clear
/// any bound by hundreds of times, and 64 measures 55 us against the
/// 5 ms deadline. A 64-block request runs at the device's SM count where
/// that is smaller, so on a 48-SM card this is the 48-block team. It reached 412 us once, on a run where another test
/// binary held the device at the same time, which put it at 82 percent
/// of this bound and turned the assertion into a reading of the
/// neighbour. The device lock in [`common`] is what makes 55 the figure
/// this test sees.
#[test]
fn a_healthy_team_costs_far_less_at_the_barrier_than_its_deadline() {
    let _device = common::device();
    const DEADLINE_NS: u64 = 5_000_000;
    for team in [2u32, 4, 8, 64] {
        let mut peer = impatient_team_peer(team, DEADLINE_NS);
        let mut payload = vec![0u8; RESIDENT_PREFIX + MARKS];

        for _ in 0..32 {
            let t = peer
                .submit_user(layout::OP_USER_BASE + 100, None, &payload)
                .expect("submit");
            assert_eq!(
                peer.wait_status(t, Duration::from_secs(10)).expect("completes"),
                STATUS_DONE
            );
            peer.read_result(t, &mut payload).expect("read");
            peer.reap(t).expect("reap");
        }

        let waited = peer.barrier_wait_max_ns();
        let (stalls, _) = peer.barrier_stalls();
        println!(
            "team {team} running {}: worst healthy barrier wait {waited} ns against a \
             {DEADLINE_NS} ns deadline; {stalls} expiries",
            peer.team_size()
        );

        assert_eq!(stalls, 0, "team {team}: no team should have missed on an idle host");
        assert!(
            (waited as u64) < DEADLINE_NS / 10,
            "team {team}: a healthy team waited {waited} ns, within an order \
             of magnitude of its {DEADLINE_NS} ns deadline. The default rests \
             on that margin being large, and without it the deadline abandons \
             teams that assemble normally"
        );
    }
}

/// A team submission answers the caller rather than paying the
/// deadline.
///
/// Rank 0 retires on a deadline of one whole quantum, and the followers
/// wait for that retirement on the same deadline, so a team that never
/// assembles can cost two quanta before anything reaches the host. The
/// good case must sit far below that.
///
/// Measured device-side rather than by wall clock. An earlier version
/// timed the host round trip and asserted under 100 ms, which failed
/// whenever another test in this binary held the device - the tests run
/// concurrently, and a round trip is charged for whatever else is
/// queued. The barrier wait is the quantity the assertion was always
/// about, and it is unaffected by who else is on the machine.
#[test]
fn a_team_submission_does_not_approach_its_barrier_deadline() {
    let _device = common::device();
    let mut peer = team_peer(4);
    let t = peer
        .submit_user(layout::OP_USER_BASE + 100, None, &[0u8; 64])
        .expect("submit");
    let status = peer.wait(t, Duration::from_secs(10)).expect("completes");
    peer.reap(t).expect("reap");

    assert_eq!(status, STATUS_DONE);
    let waited = peer.barrier_wait_max_ns();
    let (stalls, _) = peer.barrier_stalls();
    assert_eq!(stalls, 0, "the team assembled, so nothing should have expired");
    assert!(
        (waited as u64) < 5_000_000 / 10,
        "a team that assembles must sit far below the default barrier \
         deadline; rank 0 waited {waited} ns"
    );
}

/// A team requested wider than the device runs at the device's SM count,
/// and every rank of the team it runs still contributes.
#[test]
fn a_team_wider_than_the_device_runs_at_its_sm_count() {
    let _device = common::device();
    let Some(sm) = flynnel::backend::detect::cuda_sm_count(0) else {
        panic!("this test needs the device's SM count, and the driver reported none");
    };
    let mut peer = team_peer(sm * 2);
    assert_eq!(
        peer.team_size(),
        sm,
        "a request of {} blocks on a device with {sm} SMs",
        sm * 2
    );

    let marks_len = (sm as usize + 1).max(MARKS);
    let mut payload = vec![0u8; RESIDENT_PREFIX + marks_len];
    let t = peer
        .submit_user(layout::OP_USER_BASE + 100, None, &payload)
        .expect("submit the team op");
    assert_eq!(
        peer.wait_status(t, Duration::from_secs(10)).expect("completes"),
        STATUS_DONE
    );
    peer.read_result(t, &mut payload).expect("the result fits the slot");
    peer.reap(t).expect("reap");

    let marks = &payload[RESIDENT_PREFIX..];
    for (rank, &mark) in marks.iter().enumerate().take(sm as usize) {
        assert_eq!(mark, (rank + 1) as u8, "rank {rank} of the clamped team left no mark");
    }
    assert!(
        marks[sm as usize..].iter().all(|&b| b == 0),
        "a byte past the clamped team's last rank was written"
    );
}
