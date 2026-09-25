//! The text kernels over one large string: the literal search, the
//! counts, the split and the rewrite.
//!
//! A block searches its own span extended by the pattern's length less
//! one, so a match across a block boundary is found by the block to its
//! left, and a match belongs to the block its first byte falls in, which
//! keeps the overlap from reporting one twice. Hits are gathered block by
//! block and so arrive in ascending order with no sort.

use std::ops::Range;

use super::files::{unterminated_tail, without_cr};
use super::{BlockError, Blocks, Job, MAX_BLOCKS, Refusal, Slots, Tracker, tracked};

/// The least number of bytes a text block holds, where the text has that
/// many.
const TEXT_BLOCK_MIN: usize = 64 * 1024;

/// A place in a string where a pattern was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextMatch {
    /// The byte offset of the match, counting from zero.
    pub index: u64,
    /// The line the match is on, counting from one.
    pub line_number: u64,
    /// The whole line the match is on, without its terminator.
    pub line: String,
}

/// What a text measurement answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextMeasure {
    /// How many bytes the text holds.
    pub bytes: u64,
    /// How many lines it holds.
    pub lines: u64,
    /// How many whitespace-separated words it holds.
    pub words: u64,
    /// How many times the pattern occurs, counting non-overlapping
    /// occurrences. `None` when no pattern was given.
    pub matches: Option<u64>,
}

/// What [`update_text`] does to the text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextTransform {
    /// Replace every non-overlapping occurrence of the pattern with the
    /// replacement.
    #[default]
    Replace,
    /// Upper-case, by the Unicode mapping.
    ToUpper,
    /// Lower-case, by the Unicode mapping.
    ToLower,
}

impl TextTransform {
    /// Every transform, in declaration order.
    pub const ALL: [Self; 3] = [Self::Replace, Self::ToUpper, Self::ToLower];

    /// Every transform's name, in the order of [`Self::ALL`].
    pub const NAMES: [&'static str; 3] = ["Replace", "ToUpper", "ToLower"];
}

/// Every offset in `span` at which `needle` starts in `hay`, searched
/// past the span's end by the needle's length less one.
fn hits_in(hay: &[u8], needle: &[u8], span: Range<usize>) -> Vec<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return Vec::new();
    }
    let stop = (span.end + needle.len() - 1).min(hay.len());
    if span.start >= stop || stop - span.start < needle.len() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for (offset, w) in hay[span.start..stop].windows(needle.len()).enumerate() {
        if w == needle && span.start + offset < span.end {
            hits.push(span.start + offset);
        }
    }
    hits
}

/// Every newline offset in `span`.
fn newlines_in(hay: &[u8], span: Range<usize>) -> Vec<usize> {
    let start = span.start;
    hay[span]
        .iter()
        .enumerate()
        .filter(|(_offset, b)| **b == b'\n')
        .map(|(offset, _newline)| start + offset)
        .collect()
}

/// The one-based line number of a byte offset, given the newline
/// positions in ascending order.
fn line_of(newlines: &[usize], at: usize) -> u64 {
    newlines.partition_point(|&p| p < at) as u64 + 1
}

/// The whole line containing a byte offset, without its terminator.
fn line_at(hay: &[u8], newlines: &[usize], at: usize) -> String {
    let idx = newlines.partition_point(|&p| p < at);
    let start = if idx == 0 { 0 } else { newlines[idx - 1] + 1 };
    // No newline at or after the offset means the line runs to the end
    // of the text.
    let end = match newlines.get(idx) {
        Some(&p) => p,
        None => hay.len(),
    };
    String::from_utf8_lossy(without_cr(&hay[start..end])).into_owned()
}

/// How many of the found offsets survive a left-to-right non-overlapping
/// walk.
fn non_overlapping(hits: &[usize], width: usize) -> usize {
    let mut kept = 0usize;
    let mut next_free = 0usize;
    for &at in hits {
        if at >= next_free {
            kept += 1;
            next_free = at + width;
        }
    }
    kept
}

/// The text cut into blocks, and the first phase's block count, or
/// `None` for empty text.
fn text_blocks(n: usize) -> (Blocks, Option<usize>) {
    let blocks = Blocks::new(n, TEXT_BLOCK_MIN, MAX_BLOCKS);
    let opening = (blocks.count() > 0).then_some(blocks.count());
    (blocks, opening)
}

/// Every block's hit list, concatenated in block order.
fn concat(parts: Vec<Vec<usize>>) -> Vec<usize> {
    parts.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------

/// [`search_text`]'s job: one phase finding the pattern and the newlines
/// in each block.
pub struct SearchTextJob<'a> {
    hay: &'a [u8],
    needle: Vec<u8>,
    blocks: Blocks,
    found: Slots<(Vec<usize>, Vec<usize>)>,
    hits: Vec<usize>,
    newlines: Vec<usize>,
    tracker: Tracker,
}

/// Every occurrence of the literal `pattern` in `text`, each with its
/// line.
pub fn search_text<'a>(text: &'a str, pattern: &str) -> Result<SearchTextJob<'a>, Refusal> {
    if pattern.is_empty() {
        return Err(Refusal::argument("Pattern must not be empty"));
    }
    let hay = text.as_bytes();
    let (blocks, opening) = text_blocks(hay.len());
    Ok(SearchTextJob {
        hay,
        needle: pattern.as_bytes().to_vec(),
        found: Slots::new(blocks.count()),
        hits: Vec::new(),
        newlines: Vec::new(),
        tracker: Tracker::new(opening),
        blocks,
    })
}

impl Job for SearchTextJob<'_> {
    type Answer = Vec<TextMatch>;

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let span = self.blocks.range(block);
            let hits = hits_in(self.hay, &self.needle, span.clone());
            let newlines = newlines_in(self.hay, span);
            self.found.put(block, (hits, newlines));
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        let (hits, newlines): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
            self.found.take_all()?.into_iter().unzip();
        self.hits = concat(hits);
        self.newlines = concat(newlines);
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<Vec<TextMatch>, Refusal> {
        self.tracker.check_finished()?;
        Ok(self
            .hits
            .iter()
            .map(|&at| TextMatch {
                index: at as u64,
                line_number: line_of(&self.newlines, at),
                line: line_at(self.hay, &self.newlines, at),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------
// Counts
// ---------------------------------------------------------------------

/// [`text_count`]'s job: one phase counting lines and words and finding
/// the pattern in each block.
pub struct TextCountJob<'a> {
    hay: &'a [u8],
    needle: Option<Vec<u8>>,
    blocks: Blocks,
    counts: Slots<(u64, u64, Vec<usize>)>,
    totals: (u64, u64),
    hits: Vec<usize>,
    tracker: Tracker,
}

/// The bytes, lines and words of `text`, and the non-overlapping
/// occurrences of `pattern` when one is given.
pub fn text_count<'a>(text: &'a str, pattern: Option<&str>) -> Result<TextCountJob<'a>, Refusal> {
    let needle = match pattern {
        None => None,
        Some("") => return Err(Refusal::argument("Pattern must not be empty")),
        Some(p) => Some(p.as_bytes().to_vec()),
    };
    let hay = text.as_bytes();
    let (blocks, opening) = text_blocks(hay.len());
    Ok(TextCountJob {
        hay,
        needle,
        counts: Slots::new(blocks.count()),
        totals: (0, 0),
        hits: Vec::new(),
        tracker: Tracker::new(opening),
        blocks,
    })
}

impl Job for TextCountJob<'_> {
    type Answer = TextMeasure;

    tracked!();

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let span = self.blocks.range(block);
            let (lo, hi) = (span.start, span.end);
            // A word is counted where it starts, so a block needs to know
            // whether the byte before its span was whitespace; reading
            // one byte to the left makes the per-block counts add up to
            // the serial answer.
            let mut lines = 0u64;
            let mut words = 0u64;
            let mut prev_space = lo == 0 || self.hay[lo - 1].is_ascii_whitespace();
            for &b in &self.hay[lo..hi] {
                if b == b'\n' {
                    lines += 1;
                }
                let space = b.is_ascii_whitespace();
                if prev_space && !space {
                    words += 1;
                }
                prev_space = space;
            }
            let hits = match &self.needle {
                Some(needle) => hits_in(self.hay, needle, span),
                None => Vec::new(),
            };
            self.counts.put(block, (lines, words, hits));
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        let mut lines = 0u64;
        let mut words = 0u64;
        let mut hits = Vec::new();
        for (l, w, h) in self.counts.take_all()? {
            lines += l;
            words += w;
            hits.extend(h);
        }
        self.totals = (lines, words);
        self.hits = hits;
        self.tracker.advance(None);
        Ok(())
    }

    fn finish(self) -> Result<TextMeasure, Refusal> {
        self.tracker.check_finished()?;
        Ok(TextMeasure {
            bytes: self.hay.len() as u64,
            lines: self.totals.0 + unterminated_tail(self.hay),
            words: self.totals.1,
            matches: self
                .needle
                .as_ref()
                .map(|needle| non_overlapping(&self.hits, needle.len()) as u64),
        })
    }
}

// ---------------------------------------------------------------------
// Split and replace
// ---------------------------------------------------------------------

/// The job behind [`split_text`] and [`update_text`]'s Replace: one phase
/// finding the pattern in each block, the answer built once from the
/// hits.
struct Find<'a> {
    hay: &'a [u8],
    needle: Vec<u8>,
    blocks: Blocks,
    found: Slots<Vec<usize>>,
    hits: Vec<usize>,
    tracker: Tracker,
}

impl<'a> Find<'a> {
    fn new(text: &'a str, needle: &str) -> Self {
        let hay = text.as_bytes();
        let (blocks, opening) = text_blocks(hay.len());
        Self {
            hay,
            needle: needle.as_bytes().to_vec(),
            found: Slots::new(blocks.count()),
            hits: Vec::new(),
            tracker: Tracker::new(opening),
            blocks,
        }
    }

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.tracker.run(phase, block, || {
            let hits = hits_in(self.hay, &self.needle, self.blocks.range(block));
            self.found.put(block, hits);
        })
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.tracker.check_ended(phase)?;
        self.hits = concat(self.found.take_all()?);
        self.tracker.advance(None);
        Ok(())
    }

    /// The text between the non-overlapping hits, left to right.
    fn pieces(&self) -> Vec<&'a [u8]> {
        let width = self.needle.len();
        let mut out = Vec::with_capacity(self.hits.len() + 1);
        let mut cursor = 0usize;
        for &at in &self.hits {
            if at < cursor {
                continue;
            }
            out.push(&self.hay[cursor..at]);
            cursor = at + width;
        }
        out.push(&self.hay[cursor..]);
        out
    }
}

/// [`split_text`]'s job.
pub struct SplitTextJob<'a> {
    find: Find<'a>,
    no_empty: bool,
}

/// `text` split on every non-overlapping occurrence of the literal
/// `separator`, left to right; `no_empty` drops the empty pieces two
/// adjacent separators make.
pub fn split_text<'a>(
    text: &'a str,
    separator: &str,
    no_empty: bool,
) -> Result<SplitTextJob<'a>, Refusal> {
    if separator.is_empty() {
        return Err(Refusal::argument("Separator must not be empty"));
    }
    Ok(SplitTextJob {
        find: Find::new(text, separator),
        no_empty,
    })
}

impl Job for SplitTextJob<'_> {
    type Answer = Vec<String>;

    fn blocks(&self, phase: usize) -> Option<usize> {
        self.find.tracker.blocks(phase)
    }

    fn state(&self, phase: usize, block: usize) -> Option<super::BlockState> {
        self.find.tracker.state(phase, block)
    }

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        self.find.run(phase, block)
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        self.find.end_phase(phase)
    }

    fn finish(self) -> Result<Vec<String>, Refusal> {
        self.find.tracker.check_finished()?;
        let mut out: Vec<String> = self
            .find
            .pieces()
            .into_iter()
            .map(|piece| String::from_utf8_lossy(piece).into_owned())
            .collect();
        if self.no_empty {
            out.retain(|s| !s.is_empty());
        }
        Ok(out)
    }
}

/// Split points for a case mapping: a nominal split every block, each
/// moved forward to the first point a piece may end.
///
/// For ToUpper that is any char boundary. For ToLower it is just after an
/// ASCII whitespace byte: std lowercases Σ to σ or ς by the letters on
/// both sides of it within the string it is given (Unicode's Final_Sigma),
/// so a piece boundary beside Σ would change the answer. ASCII whitespace
/// is neither cased nor case-ignorable, so no Σ decision reads across it.
/// Text with no whitespace past a point is not split past it.
fn piece_bounds(text: &str, lower: bool) -> Vec<usize> {
    let n = text.len();
    let bytes = text.as_bytes();
    let blocks = Blocks::new(n, TEXT_BLOCK_MIN, MAX_BLOCKS);
    let mut bounds = vec![0usize];
    let mut last = 0usize;
    for b in 1..blocks.count() {
        let nominal = blocks.range(b).start.max(last + 1);
        if nominal >= n {
            break;
        }
        let at = if lower {
            match bytes[nominal - 1..]
                .iter()
                .position(|c| c.is_ascii_whitespace())
            {
                Some(offset) => nominal + offset,
                None => break,
            }
        } else {
            let mut at = nominal;
            while at < n && !text.is_char_boundary(at) {
                at += 1;
            }
            at
        };
        if at > last && at < n {
            bounds.push(at);
            last = at;
        }
    }
    bounds.push(n);
    bounds
}

/// What [`update_text`] rewrites with.
enum Rewrite<'a> {
    Replace {
        find: Find<'a>,
        replacement: String,
    },
    Case {
        text: &'a str,
        upper: bool,
        bounds: Vec<usize>,
        pieces: Slots<String>,
        done: Vec<String>,
        tracker: Tracker,
    },
}

/// [`update_text`]'s job.
pub struct UpdateTextJob<'a> {
    rewrite: Rewrite<'a>,
    capacity: usize,
}

/// `text` rewritten by `op`. Replace swaps every non-overlapping
/// occurrence of `pattern` for `replacement`, and an absent replacement
/// deletes the pattern; the case transforms map each piece of the text
/// and join the pieces.
pub fn update_text<'a>(
    text: &'a str,
    op: TextTransform,
    pattern: Option<&str>,
    replacement: Option<&str>,
) -> Result<UpdateTextJob<'a>, Refusal> {
    let rewrite = match op {
        TextTransform::Replace => {
            let Some(pattern) = pattern else {
                return Err(Refusal::argument("Replace needs Pattern"));
            };
            if pattern.is_empty() {
                return Err(Refusal::argument("Pattern must not be empty"));
            }
            // Deleting the pattern is what an absent replacement asks for.
            let replacement = match replacement {
                Some(r) => r.to_string(),
                None => String::new(),
            };
            Rewrite::Replace {
                find: Find::new(text, pattern),
                replacement,
            }
        }
        TextTransform::ToUpper | TextTransform::ToLower => {
            let upper = op == TextTransform::ToUpper;
            let bounds = if text.is_empty() {
                Vec::new()
            } else {
                piece_bounds(text, !upper)
            };
            let n_pieces = bounds.len().saturating_sub(1);
            Rewrite::Case {
                text,
                upper,
                pieces: Slots::new(n_pieces),
                done: Vec::new(),
                tracker: Tracker::new((n_pieces > 0).then_some(n_pieces)),
                bounds,
            }
        }
    };
    Ok(UpdateTextJob {
        rewrite,
        capacity: text.len(),
    })
}

impl Job for UpdateTextJob<'_> {
    type Answer = String;

    fn blocks(&self, phase: usize) -> Option<usize> {
        match &self.rewrite {
            Rewrite::Replace { find, .. } => find.tracker.blocks(phase),
            Rewrite::Case { tracker, .. } => tracker.blocks(phase),
        }
    }

    fn state(&self, phase: usize, block: usize) -> Option<super::BlockState> {
        match &self.rewrite {
            Rewrite::Replace { find, .. } => find.tracker.state(phase, block),
            Rewrite::Case { tracker, .. } => tracker.state(phase, block),
        }
    }

    fn run(&self, phase: usize, block: usize) -> Result<(), BlockError> {
        match &self.rewrite {
            Rewrite::Replace { find, .. } => find.run(phase, block),
            Rewrite::Case {
                text,
                upper,
                bounds,
                pieces,
                tracker,
                ..
            } => tracker.run(phase, block, || {
                let piece = &text[bounds[block]..bounds[block + 1]];
                let mapped = if *upper {
                    piece.to_uppercase()
                } else {
                    piece.to_lowercase()
                };
                pieces.put(block, mapped);
            }),
        }
    }

    fn end_phase(&mut self, phase: usize) -> Result<(), Refusal> {
        match &mut self.rewrite {
            Rewrite::Replace { find, .. } => find.end_phase(phase),
            Rewrite::Case {
                pieces,
                done,
                tracker,
                ..
            } => {
                tracker.check_ended(phase)?;
                *done = pieces.take_all()?;
                tracker.advance(None);
                Ok(())
            }
        }
    }

    fn finish(self) -> Result<String, Refusal> {
        match self.rewrite {
            Rewrite::Replace { find, replacement } => {
                find.tracker.check_finished()?;
                let pieces = find.pieces();
                let mut out = String::with_capacity(self.capacity);
                let last = pieces.len().saturating_sub(1);
                for (i, piece) in pieces.into_iter().enumerate() {
                    out.push_str(&String::from_utf8_lossy(piece));
                    if i < last {
                        out.push_str(&replacement);
                    }
                }
                Ok(out)
            }
            Rewrite::Case { done, tracker, .. } => {
                tracker.check_finished()?;
                Ok(done.concat())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{drive, drive_serial};
    use super::*;

    /// Run every phase's blocks in reverse order.
    fn drive_reversed<J: Job>(job: J) -> Result<J::Answer, Refusal> {
        drive(job, |n, body| {
            for b in (0..n).rev() {
                body(b)?;
            }
            Ok(())
        })
    }

    /// A text long enough to be cut into several blocks, with matches on
    /// and around the block boundaries.
    fn long_text() -> String {
        let mut s = String::new();
        let mut i = 0u64;
        while s.len() < 700_000 {
            s.push_str(&format!("line {i} has ERROR and ERRORERROR\r\n"));
            i += 1;
        }
        s.push_str("tail without newline ERROR");
        s
    }

    #[test]
    fn a_search_finds_what_a_serial_scan_finds_with_the_right_lines() {
        let text = long_text();
        let got =
            drive_reversed(search_text(&text, "ERROR").expect("a pattern")).expect("finishes");
        let want: Vec<usize> = text.match_indices("ERROR").map(|(at, _found)| at).collect();
        assert_eq!(
            got.iter().map(|m| m.index as usize).collect::<Vec<_>>(),
            want
        );
        assert_eq!(got[0].line_number, 1);
        assert_eq!(got[0].line, "line 0 has ERROR and ERRORERROR");
        let last = got.last().expect("matches");
        assert_eq!(last.line, "tail without newline ERROR");
        assert_eq!(
            search_text(&text, "").err().map(|r| r.message).as_deref(),
            Some("Pattern must not be empty")
        );
    }

    #[test]
    fn counts_agree_with_the_serial_walk() {
        let text = long_text();
        let got = drive_reversed(text_count(&text, Some("ERRORERROR")).expect("a pattern"))
            .expect("finishes");
        assert_eq!(got.bytes, text.len() as u64);
        assert_eq!(got.lines, text.lines().count() as u64);
        assert_eq!(got.words, text.split_ascii_whitespace().count() as u64);
        assert_eq!(got.matches, Some(text.matches("ERRORERROR").count() as u64));
        let none = drive_serial(text_count("", None).expect("no pattern")).expect("finishes");
        assert_eq!(
            none,
            TextMeasure {
                bytes: 0,
                lines: 0,
                words: 0,
                matches: None
            }
        );
    }

    #[test]
    fn a_split_and_a_replace_agree_with_std() {
        let text = long_text();
        let parts = drive_reversed(split_text(&text, "ERROR", false).expect("a separator"))
            .expect("finishes");
        assert_eq!(parts, text.split("ERROR").collect::<Vec<_>>());
        let kept =
            drive_serial(split_text("a,,b", ",", true).expect("a separator")).expect("finishes");
        assert_eq!(kept, vec!["a", "b"]);
        assert_eq!(
            drive_serial(split_text("", ",", false).expect("a separator")).expect("finishes"),
            vec![String::new()]
        );
        let replaced = drive_reversed(
            update_text(&text, TextTransform::Replace, Some("ERROR"), Some("ok"))
                .expect("a pattern"),
        )
        .expect("finishes");
        assert_eq!(replaced, text.replace("ERROR", "ok"));
        let deleted = drive_serial(
            update_text("aXbXc", TextTransform::Replace, Some("X"), None).expect("a pattern"),
        )
        .expect("finishes");
        assert_eq!(deleted, "abc");
        assert_eq!(
            update_text("a", TextTransform::Replace, None, None)
                .err()
                .map(|r| r.message)
                .as_deref(),
            Some("Replace needs Pattern")
        );
    }

    #[test]
    fn lowercasing_splits_where_no_final_sigma_can_be_moved() {
        // Σ lowercases to ς at the end of a word and to σ inside one, by
        // the letters around it. A piece boundary between Σ and the letter
        // after it would read the Σ as word-final.
        let mut text = String::new();
        while text.len() < 600_000 {
            text.push_str("ΟΔΟΣΣΟΦΙΑΣ ΑΣΑ ΣΑΣ\n");
        }
        let lowered = drive_reversed(
            update_text(&text, TextTransform::ToLower, None, None).expect("no operand"),
        )
        .expect("finishes");
        assert_eq!(lowered, text.to_lowercase());
        let raised = drive_reversed(
            update_text(&lowered, TextTransform::ToUpper, None, None).expect("no operand"),
        )
        .expect("finishes");
        assert_eq!(raised, lowered.to_uppercase());
        for &at in &piece_bounds(&text, true)[1..] {
            if at < text.len() {
                assert!(
                    text.as_bytes()[at - 1].is_ascii_whitespace(),
                    "split at {at}"
                );
            }
        }
    }

    #[test]
    fn text_with_no_whitespace_lowercases_whole() {
        let text = "ΣΑ".repeat(200_000);
        assert_eq!(piece_bounds(&text, true), vec![0, text.len()]);
        let lowered = drive_serial(
            update_text(&text, TextTransform::ToLower, None, None).expect("no operand"),
        )
        .expect("finishes");
        assert_eq!(lowered, text.to_lowercase());
    }

    #[test]
    fn the_transform_names_follow_the_declaration_order() {
        for (op, name) in TextTransform::ALL.iter().zip(TextTransform::NAMES) {
            assert_eq!(format!("{op:?}"), name);
        }
    }
}
