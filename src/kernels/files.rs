//! The file kernels: hashing with BLAKE3, the manifest check, the literal
//! search, and line and byte counts, one block per file.
//!
//! A file that cannot be read is a row of the answer carrying its
//! refusal, in the path's own position, so a run over a thousand files
//! reports the nine it could not open rather than stopping at the first
//! or answering 991 rows with nothing to say which are missing.

use super::{BlockError, Job, OneCopy, Refusal, Slots, Tracker};

/// A line of a file that matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileMatch {
    /// The file the line came from.
    pub path: String,
    /// The line's position, counting from one.
    pub line_number: u64,
    /// The line, without its terminator.
    pub line: String,
}

/// What one file measured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileMeasure {
    /// The file this describes.
    pub path: String,
    /// How many lines it holds. A final line with no terminator counts.
    /// Zero from [`file_byte`], which counts no lines.
    pub lines: u64,
    /// How many bytes it holds.
    pub bytes: u64,
}

/// A file read whole, or the failure text when it cannot be.
fn slurp(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| e.to_string())
}

/// A line with its terminating carriage return removed, when it has one.
pub(crate) fn without_cr(line: &[u8]) -> &[u8] {
    match line.strip_suffix(b"\r") {
        Some(trimmed) => trimmed,
        None => line,
    }
}

/// One when a buffer's last byte leaves a line open, which makes that
/// final unterminated line one more line; zero otherwise.
pub(crate) fn unterminated_tail(bytes: &[u8]) -> u64 {
    match bytes.last() {
        None => 0,
        Some(&b'\n') => 0,
        Some(_last) => 1,
    }
}

/// Whether `hay` holds `needle`.
fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Every line of a buffer holding `needle`, with its one-based line
/// number. `-IgnoreCase` folds the ASCII range and nothing else.
fn matching_lines(hay: &[u8], needle: &[u8], ignore_case: bool) -> Vec<(u64, String)> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return out;
    }
    let folded_needle = if ignore_case {
        needle.to_ascii_lowercase()
    } else {
        needle.to_vec()
    };
    for (idx, raw) in hay.split(|&b| b == b'\n').enumerate() {
        let line = without_cr(raw);
        let hit = if ignore_case {
            contains(&line.to_ascii_lowercase(), &folded_needle)
        } else {
            contains(line, &folded_needle)
        };
        if hit {
            out.push((idx as u64 + 1, String::from_utf8_lossy(line).into_owned()));
        }
    }
    out
}

/// A job of one phase with one block per path, each block answering one
/// row: what the file gave, or the refusal naming it.
struct PerFile<'a, R> {
    paths: &'a [String],
    rows: Slots<Result<R, Refusal>>,
    done: Vec<Result<R, Refusal>>,
    tracker: Tracker,
}

impl<'a, R: Send + Sync> PerFile<'a, R> {
    fn new(paths: &'a [String]) -> Self {
        let n = paths.len();
        Self {
            paths,
            rows: Slots::new(n),
            done: Vec::new(),
            tracker: Tracker::new((n > 0).then_some(n)),
        }
    }

    /// Run block `block` of `phase`: `work` over its file, as one copy at
    /// every level, since a file's time goes to reading it.
    fn run(
        &self,
        phase: usize,
        block: usize,
        work: impl Fn(&str) -> Result<R, String>,
    ) -> Result<(), BlockError> {
        self.tracker.run(
            phase,
            block,
            &OneCopy(|_phase, block| {
                let path = &self.paths[block];
                let row = work(path).map_err(|detail| Refusal::unreadable(path, detail));
                self.rows.put(block, row);
            }),
        )
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        self.done = self.rows.take_all()?;
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<Vec<Result<R, Refusal>>, Refusal> {
        self.tracker.check_finished()?;
        Ok(self.done)
    }
}

/// [`search_file`]'s job: one block per file.
pub struct SearchFileJob<'a> {
    inner: PerFile<'a, Vec<FileMatch>>,
    needle: Vec<u8>,
    ignore_case: bool,
}

/// Search each file in `paths` for the literal `pattern`, matching bytes
/// ordinally, with `ignore_case` folding the ASCII range and nothing
/// else. Each file's row is its matching lines, or its refusal.
pub fn search_file<'a>(
    pattern: &str,
    paths: &'a [String],
    ignore_case: bool,
) -> Result<SearchFileJob<'a>, Refusal> {
    if pattern.is_empty() {
        return Err(Refusal::argument("Pattern must not be empty"));
    }
    Ok(SearchFileJob {
        inner: PerFile::new(paths),
        needle: pattern.as_bytes().to_vec(),
        ignore_case,
    })
}

impl Job for SearchFileJob<'_> {
    type Answer = Vec<Result<Vec<FileMatch>, Refusal>>;

    fn blocks(&self, phase: usize) -> Option<usize> {
        self.inner.tracker.blocks(phase)
    }

    fn state(&self, phase: usize, block: usize) -> Option<super::BlockState> {
        self.inner.tracker.state(phase, block)
    }

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.inner.run(phase, block, |path| {
            slurp(path).map(|bytes| {
                matching_lines(&bytes, &self.needle, self.ignore_case)
                    .into_iter()
                    .map(|(line_number, line)| FileMatch {
                        path: path.to_string(),
                        line_number,
                        line,
                    })
                    .collect()
            })
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.inner.end_phase(phase)
    }

    fn finish(self) -> Result<Self::Answer, Refusal> {
        self.inner.finish()
    }
}

/// [`file_line`]'s job: one block per file.
pub struct FileLineJob<'a> {
    inner: PerFile<'a, FileMeasure>,
}

/// Count the lines and bytes of each file in `paths`. A final line with
/// no terminator counts, so a one-line file with no newline has one line.
pub fn file_line(paths: &[String]) -> FileLineJob<'_> {
    FileLineJob {
        inner: PerFile::new(paths),
    }
}

impl Job for FileLineJob<'_> {
    type Answer = Vec<Result<FileMeasure, Refusal>>;

    fn blocks(&self, phase: usize) -> Option<usize> {
        self.inner.tracker.blocks(phase)
    }

    fn state(&self, phase: usize, block: usize) -> Option<super::BlockState> {
        self.inner.tracker.state(phase, block)
    }

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.inner.run(phase, block, |path| {
            slurp(path).map(|bytes| {
                let newlines = bytes.iter().filter(|&&b| b == b'\n').count() as u64;
                FileMeasure {
                    path: path.to_string(),
                    lines: newlines + unterminated_tail(&bytes),
                    bytes: bytes.len() as u64,
                }
            })
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.inner.end_phase(phase)
    }

    fn finish(self) -> Result<Self::Answer, Refusal> {
        self.inner.finish()
    }
}

/// [`file_byte`]'s job: one block per file.
pub struct FileByteJob<'a> {
    inner: PerFile<'a, FileMeasure>,
}

/// The byte length of each file in `paths`, from its metadata rather
/// than its contents, so it answers for a file too large to hold. The
/// line count is zero because no line was counted.
pub fn file_byte(paths: &[String]) -> FileByteJob<'_> {
    FileByteJob {
        inner: PerFile::new(paths),
    }
}

impl Job for FileByteJob<'_> {
    type Answer = Vec<Result<FileMeasure, Refusal>>;

    fn blocks(&self, phase: usize) -> Option<usize> {
        self.inner.tracker.blocks(phase)
    }

    fn state(&self, phase: usize, block: usize) -> Option<super::BlockState> {
        self.inner.tracker.state(phase, block)
    }

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.inner.run(phase, block, |path| {
            std::fs::metadata(path)
                .map(|m| FileMeasure {
                    path: path.to_string(),
                    lines: 0,
                    bytes: m.len(),
                })
                .map_err(|e| e.to_string())
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.inner.end_phase(phase)
    }

    fn finish(self) -> Result<Self::Answer, Refusal> {
        self.inner.finish()
    }
}

#[cfg(feature = "verify-chain")]
pub use hashing::{
    FileHash, FileHashCheckJob, FileHashJob, HashCheck, HashChecked, file_hash, file_hash_check,
    file_hash_streamed,
};

#[cfg(feature = "verify-chain")]
mod hashing {
    use blake3::hazmat::{
        ChainingValue, HasherExt, Mode, left_subtree_len, merge_subtrees_non_root,
        merge_subtrees_root,
    };

    use super::super::{BlockError, BlockState, Job, OneCopy, Refusal, Slots, Tracker};

    /// A file's BLAKE3 root.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct FileHash {
        /// The file this describes.
        pub path: String,
        /// The BLAKE3 root, lower-case hex.
        pub hash: String,
        /// How many bytes were read.
        pub bytes: u64,
    }

    /// One file checked against a manifest.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct HashCheck {
        /// The file this describes.
        pub path: String,
        /// What the manifest said it should be, lower-case hex.
        pub expected: String,
        /// What it hashed to. `None` when the file could not be read.
        pub actual: Option<String>,
        /// Whether the two agree. False for a file that could not be
        /// read, so a filter on it cannot pass an unreadable file.
        pub is_match: bool,
    }

    /// A [`HashCheck`] row with the refusal behind it when the file
    /// could not be read. A manifest check whose failure mode is silence
    /// cannot be used to decide anything, so an unreadable file is a row
    /// and a refusal, never an absent row.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct HashChecked {
        /// The row.
        pub check: HashCheck,
        /// Why the file could not be read, when it could not.
        pub refusal: Option<Refusal>,
    }

    /// How much of a file one streamed read takes.
    const FILE_CHUNK: usize = 1 << 20;

    /// The largest file read whole in order to hash its subtrees on
    /// several blocks. Above it the file streams through one hasher.
    /// Both paths give the standard root, so this bounds memory and speed
    /// and never what the answer is.
    const MAX_IN_MEMORY: u64 = 1 << 30;

    /// The largest span one block hashes as a single subtree. Below it
    /// one hasher runs the span, which is where BLAKE3's own SIMD
    /// parallelism does the work.
    const SUBTREE_LEAF: usize = 1 << 20;

    /// Lower-case hex for a 32-byte root.
    fn hex32(bytes: &[u8; 32]) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(64);
        for b in bytes {
            s.push(DIGITS[(b >> 4) as usize] as char);
            s.push(DIGITS[(b & 0x0F) as usize] as char);
        }
        s
    }

    /// The BLAKE3 root of a file read a chunk at a time.
    fn hash_file_streaming(path: &str) -> Result<(String, u64), String> {
        use std::io::Read;
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(file);
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0u8; FILE_CHUNK];
        let mut total = 0u64;
        loop {
            let got = reader.read(&mut buf).map_err(|e| e.to_string())?;
            if got == 0 {
                break;
            }
            hasher.update(&buf[..got]);
            total += got as u64;
        }
        Ok((hex32(hasher.finalize().as_bytes()), total))
    }

    /// The subtree spans of a buffer, left to right, by BLAKE3's own
    /// split rule, which is the only split that yields valid subtrees.
    /// It depends on the length alone.
    fn plan_subtrees(start: usize, len: usize, out: &mut Vec<(usize, usize)>) {
        if len <= SUBTREE_LEAF || len <= blake3::CHUNK_LEN {
            out.push((start, len));
            return;
        }
        let left = left_subtree_len(len as u64) as usize;
        plan_subtrees(start, left, out);
        plan_subtrees(start + left, len - left, out);
    }

    /// The subtree chaining values combined back up the tree, walking the
    /// split that produced them so each lands where it belongs.
    fn fold_subtrees(len: usize, cvs: &[ChainingValue], next: &mut usize) -> ChainingValue {
        if len <= SUBTREE_LEAF || len <= blake3::CHUNK_LEN {
            let cv = cvs[*next];
            *next += 1;
            return cv;
        }
        let left = left_subtree_len(len as u64) as usize;
        let l = fold_subtrees(left, cvs, next);
        let r = fold_subtrees(len - left, cvs, next);
        merge_subtrees_non_root(&l, &r, Mode::Hash)
    }

    /// The BLAKE3 root of one file streamed through one hasher, as a
    /// [`file_hash`] row, for a caller that runs its own reads, such as on
    /// an IO pool. The root is the one [`file_hash`] answers for the same
    /// file.
    pub fn file_hash_streamed(path: &str) -> Result<FileHash, Refusal> {
        match hash_file_streaming(path) {
            Ok((hash, bytes)) => Ok(FileHash {
                path: path.to_string(),
                hash,
                bytes,
            }),
            Err(detail) => Err(Refusal::unreadable(path, detail)),
        }
    }

    /// What reading the one file of a one-file hash gave.
    enum Loaded {
        /// The file is hashed already: small enough for one hasher, too
        /// large to hold, or unreadable.
        Answered(Result<(String, u64), String>),
        /// The bytes, to be hashed as subtrees in the next phase.
        Held(Vec<u8>),
    }

    /// [`file_hash`]'s job.
    ///
    /// Several files: one phase with a block per file, each streamed
    /// through one hasher. One file: a phase with one block that reads it,
    /// then a phase with a block per BLAKE3 subtree span, folded into the
    /// standard root, which is the file's own tree rather than a scheme of
    /// this crate's.
    pub struct FileHashJob<'a> {
        paths: &'a [String],
        streamed: Slots<Result<(String, u64), String>>,
        read: Slots<Loaded>,
        bytes: Vec<u8>,
        spans: Vec<(usize, usize)>,
        /// How many of `spans` belong to the root's left half.
        left_count: usize,
        cvs: Slots<ChainingValue>,
        answer: Vec<Result<(String, u64), String>>,
        tracker: Tracker,
    }

    /// Hash each file in `paths` with BLAKE3. Each row is the file's root,
    /// or its refusal.
    pub fn file_hash(paths: &[String]) -> FileHashJob<'_> {
        let n = paths.len();
        let one = n == 1;
        FileHashJob {
            paths,
            streamed: Slots::new(if one { 0 } else { n }),
            read: Slots::new(if one { 1 } else { 0 }),
            bytes: Vec::new(),
            spans: Vec::new(),
            left_count: 0,
            cvs: Slots::new(0),
            answer: Vec::new(),
            tracker: Tracker::new((n > 0).then_some(n)),
        }
    }

    impl FileHashJob<'_> {
        fn one_file(&self) -> bool {
            self.paths.len() == 1
        }

        /// The file's contents, or its root when it needs no split.
        fn read_one(path: &str) -> Loaded {
            let size = match std::fs::metadata(path) {
                Ok(m) => m.len(),
                Err(e) => return Loaded::Answered(Err(e.to_string())),
            };
            if size > MAX_IN_MEMORY {
                return Loaded::Answered(hash_file_streaming(path));
            }
            match std::fs::read(path) {
                Err(e) => Loaded::Answered(Err(e.to_string())),
                Ok(bytes) if bytes.len() <= SUBTREE_LEAF => {
                    let len = bytes.len() as u64;
                    Loaded::Answered(Ok((hex32(blake3::hash(&bytes).as_bytes()), len)))
                }
                Ok(bytes) => Loaded::Held(bytes),
            }
        }

        /// The rows, each refusal naming its path.
        fn rows(&self) -> Vec<Result<FileHash, Refusal>> {
            self.answer
                .iter()
                .zip(self.paths)
                .map(|(row, path)| match row {
                    Ok((hash, bytes)) => Ok(FileHash {
                        path: path.clone(),
                        hash: hash.clone(),
                        bytes: *bytes,
                    }),
                    Err(detail) => Err(Refusal::unreadable(path, detail)),
                })
                .collect()
        }
    }

    impl Job for FileHashJob<'_> {
        type Answer = Vec<Result<FileHash, Refusal>>;

        fn blocks(&self, phase: usize) -> Option<usize> {
            self.tracker.blocks(phase)
        }

        fn state(&self, phase: usize, block: usize) -> Option<BlockState> {
            self.tracker.state(phase, block)
        }

        fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
            // One copy at every level: BLAKE3 picks its own instructions.
            self.tracker.run(
                phase,
                block,
                &OneCopy(|phase, block| {
                    if !self.one_file() {
                        self.streamed
                            .put(block, hash_file_streaming(&self.paths[block]));
                    } else if phase == 0 {
                        self.read.put(block, Self::read_one(&self.paths[0]));
                    } else {
                        let (start, len) = self.spans[block];
                        let cv = blake3::Hasher::new()
                            .set_input_offset(start as u64)
                            .update(&self.bytes[start..start + len])
                            .finalize_non_root();
                        self.cvs.put(block, cv);
                    }
                }),
            )
        }

        fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
            self.tracker.check_ended(phase)?;
            if !self.one_file() {
                self.answer = self.streamed.take_all()?;
                self.tracker.advance(None);
                return Ok(());
            }
            if phase == 0 {
                let Some(read) = self.read.take_all()?.pop() else {
                    return Err(Refusal::internal("the read of the one file left no result"));
                };
                match read {
                    Loaded::Answered(row) => {
                        self.answer = vec![row];
                        self.tracker.advance(None);
                    }
                    Loaded::Held(bytes) => {
                        // The root split is the one merge that is
                        // root-flagged, so it is taken at the end and the
                        // two halves are folded as ordinary subtrees.
                        let n = bytes.len();
                        let left = left_subtree_len(n as u64) as usize;
                        let mut spans = Vec::new();
                        plan_subtrees(0, left, &mut spans);
                        self.left_count = spans.len();
                        plan_subtrees(left, n - left, &mut spans);
                        self.cvs = Slots::new(spans.len());
                        self.tracker.advance(Some(spans.len()));
                        self.spans = spans;
                        self.bytes = bytes;
                    }
                }
                return Ok(());
            }
            let cvs = self.cvs.take_all()?;
            let n = self.bytes.len();
            let left = left_subtree_len(n as u64) as usize;
            let mut from_left = 0usize;
            let lcv = fold_subtrees(left, &cvs[..self.left_count], &mut from_left);
            let mut from_right = 0usize;
            let rcv = fold_subtrees(n - left, &cvs[self.left_count..], &mut from_right);
            let root = hex32(merge_subtrees_root(&lcv, &rcv, Mode::Hash).as_bytes());
            self.answer = vec![Ok((root, n as u64))];
            self.bytes = Vec::new();
            self.tracker.advance(None);
            Ok(())
        }

        fn finish(self) -> Result<Self::Answer, Refusal> {
            self.tracker.check_finished()?;
            Ok(self.rows())
        }
    }

    /// [`file_hash_check`]'s job: [`file_hash`] over the paths, each root
    /// compared with the manifest's.
    pub struct FileHashCheckJob<'a> {
        hashing: FileHashJob<'a>,
        manifest: &'a [String],
    }

    /// Check each file in `paths` against the root `manifest` gives for
    /// it, in the same order. Case in the manifest does not matter.
    pub fn file_hash_check<'a>(
        paths: &'a [String],
        manifest: &'a [String],
    ) -> Result<FileHashCheckJob<'a>, Refusal> {
        if paths.len() != manifest.len() {
            return Err(Refusal::argument(format!(
                "Path has {} entr(ies) and Manifest has {}; each file needs exactly one expected \
                 root",
                paths.len(),
                manifest.len()
            )));
        }
        Ok(FileHashCheckJob {
            hashing: file_hash(paths),
            manifest,
        })
    }

    impl Job for FileHashCheckJob<'_> {
        type Answer = Vec<HashChecked>;

        fn blocks(&self, phase: usize) -> Option<usize> {
            self.hashing.blocks(phase)
        }

        fn state(&self, phase: usize, block: usize) -> Option<BlockState> {
            self.hashing.state(phase, block)
        }

        fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
            self.hashing.run(phase, block)
        }

        fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
            self.hashing.end_phase(phase)
        }

        fn finish(self) -> Result<Self::Answer, Refusal> {
            let manifest = self.manifest;
            let paths = self.hashing.paths;
            let rows = self.hashing.finish()?;
            Ok(rows
                .into_iter()
                .zip(manifest)
                .zip(paths)
                .map(|((row, expected), path)| {
                    let expected = expected.to_ascii_lowercase();
                    match row {
                        Ok(found) => {
                            let is_match = found.hash == expected;
                            HashChecked {
                                check: HashCheck {
                                    path: path.clone(),
                                    expected,
                                    actual: Some(found.hash),
                                    is_match,
                                },
                                refusal: None,
                            }
                        }
                        Err(refusal) => HashChecked {
                            check: HashCheck {
                                path: path.clone(),
                                expected,
                                actual: None,
                                is_match: false,
                            },
                            refusal: Some(refusal),
                        },
                    }
                })
                .collect())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::super::{drive_serial, simd};
        use super::*;

        fn scratch(name: &str, bytes: &[u8]) -> String {
            let dir = std::env::temp_dir().join(format!(
                "flynnel-kernels-hash-{}-{name}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("a scratch directory");
            let path = dir.join("input.bin");
            std::fs::write(&path, bytes).expect("the scratch file is written");
            path.to_string_lossy().into_owned()
        }

        #[test]
        fn one_large_file_hashes_to_the_standard_root_through_its_subtrees() {
            let bytes: Vec<u8> = (0..5_000_000u32).map(|i| (i * 31 % 251) as u8).collect();
            let path = scratch("large", &bytes);
            let paths = vec![path];
            let rows = drive_serial(file_hash(&paths)).expect("finishes");
            let row = rows[0].as_ref().expect("the file reads");
            assert_eq!(row.hash, blake3::hash(&bytes).to_hex().to_string());
            assert_eq!(row.bytes, 5_000_000);
        }

        #[test]
        fn an_unreadable_file_is_a_row_naming_it() {
            let good = scratch("good", b"abc");
            let paths = vec![good, "no-such-file-for-flynnel-kernels".to_string()];
            let rows = drive_serial(file_hash(&paths)).expect("finishes");
            assert_eq!(
                rows[0].as_ref().expect("the good file reads").hash,
                blake3::hash(b"abc").to_hex().to_string()
            );
            let refused = rows[1].as_ref().expect_err("the missing file is refused");
            assert!(
                refused
                    .message
                    .starts_with("no-such-file-for-flynnel-kernels could not be read: ")
            );
            let manifest = vec![
                blake3::hash(b"abc").to_hex().to_uppercase(),
                "00".to_string(),
            ];
            let checked = drive_serial(file_hash_check(&paths, &manifest).expect("lengths"))
                .expect("finishes");
            assert!(checked[0].check.is_match);
            assert!(!checked[1].check.is_match);
            assert_eq!(checked[1].check.actual, None);
            assert!(checked[1].refusal.is_some());
        }

        #[test]
        fn a_hash_answers_the_same_at_every_level() {
            let bytes: Vec<u8> = (0..3_000_000u32).map(|i| (i * 31 % 251) as u8).collect();
            let paths = vec![scratch("levels", &bytes)];
            simd::assert_every_level_agrees("hash", || {
                drive_serial(file_hash(&paths)).expect("finishes")
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{drive_serial, simd};
    use super::*;

    fn scratch(name: &str, bytes: &[u8]) -> String {
        let dir = std::env::temp_dir().join(format!(
            "flynnel-kernels-files-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("input.txt");
        std::fs::write(&path, bytes).expect("the scratch file is written");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn lines_bytes_and_matches_answer_per_file_in_path_order() {
        let a = scratch("a", b"one\r\nTwo panic\nthree");
        let b = scratch("b", b"panic\n");
        let paths = vec![a.clone(), "missing-file-for-flynnel".to_string(), b.clone()];

        let lines = drive_serial(file_line(&paths)).expect("finishes");
        assert_eq!(lines[0].as_ref().expect("a reads").lines, 3);
        assert_eq!(lines[2].as_ref().expect("b reads").lines, 1);
        assert!(lines[1].is_err());

        let bytes = drive_serial(file_byte(&paths)).expect("finishes");
        assert_eq!(bytes[0].as_ref().expect("a reads").bytes, 20);
        assert_eq!(bytes[0].as_ref().expect("a reads").lines, 0);

        let found =
            drive_serial(search_file("PANIC", &paths, true).expect("a pattern")).expect("finishes");
        let in_a = found[0].as_ref().expect("a reads");
        assert_eq!(in_a.len(), 1);
        assert_eq!(in_a[0].line_number, 2);
        assert_eq!(in_a[0].line, "Two panic");
        assert_eq!(found[2].as_ref().expect("b reads").len(), 1);
        let exact = drive_serial(search_file("PANIC", &paths, false).expect("a pattern"))
            .expect("finishes");
        assert!(exact[0].as_ref().expect("a reads").is_empty());
        assert_eq!(
            search_file("", &paths, false)
                .err()
                .map(|r| r.message)
                .as_deref(),
            Some("Pattern must not be empty")
        );
    }

    #[test]
    fn every_kernel_answers_the_baseline_s_bits_at_every_level() {
        let mut text = String::new();
        let mut i = 0u64;
        while text.len() < 400_000 {
            text.push_str(&format!("line {i} with panic and PANIC in it\r\n"));
            i += 1;
        }
        let a = scratch("levels-a", text.as_bytes());
        let b = scratch("levels-b", b"panic\n");
        let paths = vec![a, "missing-file-for-flynnel".to_string(), b];
        simd::assert_every_level_agrees("lines", || {
            drive_serial(file_line(&paths)).expect("finishes")
        });
        simd::assert_every_level_agrees("bytes", || {
            drive_serial(file_byte(&paths)).expect("finishes")
        });
        for ignore_case in [true, false] {
            simd::assert_every_level_agrees(&format!("search ignore_case={ignore_case}"), || {
                drive_serial(search_file("PANIC", &paths, ignore_case).expect("a pattern"))
                    .expect("finishes")
            });
        }
    }
}
