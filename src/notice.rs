//! Where the library's notes go.
//!
//! A note is a line for whoever runs the program: a calibration
//! outcome, a lever value that could not be read, a device that would
//! not start. With no sink installed each note goes to stderr as one
//! line. A program that embeds Flynnel and keeps its console for itself
//! installs a sink with [`set_sink`]; while one is installed, every note
//! goes to that function, which may log it or drop it.
//!
//! ```
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! static NOTES: AtomicUsize = AtomicUsize::new(0);
//!
//! fn count_note(_note: &str) {
//!     NOTES.fetch_add(1, Ordering::Relaxed);
//! }
//!
//! flynnel::notice::set_sink(Some(count_note));
//! flynnel::notice::set_sink(None);
//! ```
//!
//! Installing is one atomic store and delivering a note is one atomic
//! load, so neither takes a lock and a sink can be set from any thread
//! at any time. A note delivered while the sink is being replaced goes
//! to one of the two sinks, never to neither. The `FLYNNEL_TRACE` dump
//! is event data rather than a note and goes to stderr whatever sink is
//! installed.

use core::sync::atomic::{AtomicPtr, Ordering};

/// A function that receives each note, one line without its newline.
pub type NoticeSink = fn(&str);

/// The installed sink as a data pointer, null when none is installed.
static SINK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Send each later note to `sink`, or to stderr with `None`.
pub fn set_sink(sink: Option<NoticeSink>) {
    let raw = match sink {
        Some(f) => f as *mut (),
        None => core::ptr::null_mut(),
    };
    SINK.store(raw, Ordering::Release);
}

/// Deliver one note to the installed sink, or to stderr when none is.
///
/// A note that is a single literal reaches the sink without being
/// copied; one with arguments is formatted into a string first.
pub(crate) fn emit(args: core::fmt::Arguments<'_>) {
    let raw = SINK.load(Ordering::Acquire);
    if raw.is_null() {
        eprintln!("{args}");
        return;
    }
    // SAFETY: SINK holds either null, handled above, or a pointer that
    // set_sink made from a NoticeSink, and a function pointer cast to a
    // data pointer and transmuted to its own function type is that
    // same function.
    let sink = unsafe { core::mem::transmute::<*mut (), NoticeSink>(raw) };
    match args.as_str() {
        Some(literal) => sink(literal),
        None => sink(&args.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Every note the test sink received. Notes from tests running
    /// beside this one can land here as well while it is installed, so
    /// the test looks for its own notes rather than counting them.
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());

    fn record(note: &str) {
        SEEN.lock().expect("the note log").push(note.to_string());
    }

    #[test]
    fn an_installed_sink_takes_every_note_and_none_leaves_them_on_stderr() {
        set_sink(Some(record));
        notice!("a note with {} in it", 42);
        notice!("a note that is one literal");
        set_sink(None);
        notice!("a note for stderr, which the sink never sees");
        let seen = SEEN.lock().expect("the note log");
        assert!(seen.iter().any(|n| n == "a note with 42 in it"));
        assert!(seen.iter().any(|n| n == "a note that is one literal"));
        assert!(!seen.iter().any(|n| n.starts_with("a note for stderr")));
    }
}
