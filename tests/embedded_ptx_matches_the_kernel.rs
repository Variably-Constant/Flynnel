//! The shipped PTX still matches the CUDA source it was generated
//! from.
//!
//! `kernels/gpu_peer.ptx` is a build artifact checked into the repo so
//! that consumers need no CUDA toolkit: with no `user_ops_cuda`, which
//! is the default, the peer loads that PTX rather than compiling
//! anything. Regenerating it after editing `kernels/gpu_peer.cu`
//! requires nvcc, so it is a manual step, and a manual step gets
//! missed.
//!
//! Missing it fails in a way that hides. Every test that registers a
//! user op goes through NVRTC, which compiles the current source, so
//! those keep passing; only the default path breaks, and it breaks as
//! a timeout on the doorbell rather than as a launch error, because a
//! poller launched with the wrong argument list simply never does
//! anything the host can see. That is what happened when
//! `barrier_deadline_ns` was added: the whole GPU suite passed except
//! five tests in one file, all reporting `Timeout`, and nothing said
//! the artifact was stale.
//!
//! So this compares the poller's parameter list on both sides. It
//! needs no device and no toolkit - both files are text, and both are
//! in the repository.

/// Parameter width, which is all PTX records: every pointer and every
/// 64-bit scalar is `.u64`, every 32-bit scalar `.u32`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Width {
    B32,
    B64,
}

const ENTRY: &str = "flynnel_peer_poller";

/// The parameter widths the CUDA source declares, in order.
fn widths_from_cuda(src: &str) -> Vec<Width> {
    let head = format!("void {ENTRY}(");
    let start = src.find(&head).expect("the poller must be defined in the CUDA source") + head.len();
    let body = &src[start..];
    let end = body.find(')').expect("the parameter list must close");

    // Several of these parameters carry a trailing `// ...` comment.
    // A comment runs to the end of its line, and the comma that ends
    // the parameter comes before it, so the comment text would lead the
    // next comma-separated chunk. Drop the comments line by line first.
    let decls: String = body[..end]
        .lines()
        .map(|line| line.split("//").next().unwrap_or("").trim_end())
        .collect::<Vec<_>>()
        .join(" ");

    decls
        .split(',')
        .map(|raw| {
            let decl = raw.trim();
            assert!(!decl.is_empty(), "empty parameter in the CUDA declaration");
            if decl.contains('*') || decl.starts_with("u64") {
                Width::B64
            } else if decl.starts_with("u32") {
                Width::B32
            } else {
                panic!("unrecognised parameter type in the CUDA declaration: {decl:?}");
            }
        })
        .collect()
}

/// The parameter widths the PTX entry declares, in order.
fn widths_from_ptx(ptx: &str) -> Vec<Width> {
    let head = format!(".visible .entry {ENTRY}(");
    let start = ptx.find(&head).expect("the poller entry must be present in the PTX") + head.len();
    let body = &ptx[start..];
    let end = body.find(')').expect("the parameter list must close");

    body[..end]
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if !line.starts_with(".param") {
                return None;
            }
            if line.contains(".u64") || line.contains(".b64") {
                Some(Width::B64)
            } else if line.contains(".u32") || line.contains(".b32") {
                Some(Width::B32)
            } else {
                panic!("unrecognised parameter width in the PTX entry: {line:?}");
            }
        })
        .collect()
}

/// The two parameter lists agree in length and in order.
///
/// Order matters as much as length: swapping two adjacent parameters of
/// different widths leaves the count alone and still hands the kernel
/// its arguments in the wrong slots.
#[test]
fn the_shipped_ptx_declares_the_same_poller_parameters_as_the_cuda_source() {
    let cuda = widths_from_cuda(include_str!("../kernels/gpu_peer.cu"));
    let ptx = widths_from_ptx(include_str!("../kernels/gpu_peer.ptx"));

    assert!(!cuda.is_empty(), "the parser found no parameters in the CUDA source");
    assert_eq!(
        cuda.len(),
        ptx.len(),
        "kernels/gpu_peer.ptx is stale: the CUDA source declares {} parameters \
         for {ENTRY} and the shipped PTX declares {}. Regenerate it with\n  \
         nvcc -ptx -arch=compute_75 gpu_peer.cu -o gpu_peer.ptx\nfrom the \
         kernels directory. Until then the default no-NVRTC path launches the \
         poller with the wrong argument list, which surfaces as every doorbell \
         operation timing out while every user-op test keeps passing.\n  \
         cuda: {cuda:?}\n  ptx:  {ptx:?}",
        cuda.len(),
        ptx.len()
    );
    assert_eq!(
        cuda, ptx,
        "kernels/gpu_peer.ptx has the right number of parameters for {ENTRY} \
         but not the right widths in the right order, so the kernel reads its \
         arguments from the wrong slots. Regenerate it with\n  \
         nvcc -ptx -arch=compute_75 gpu_peer.cu -o gpu_peer.ptx"
    );
}
