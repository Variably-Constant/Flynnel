//! What each declared kernel takes and answers, as the module's cmdlet
//! for it declares its parameters, so a library offering the same kernels
//! takes its parameters from the same source as the cmdlets.
//!
//! The module checks every descriptor against its cmdlet's own metadata
//! and fails its tests when the two disagree, so a parameter added to a
//! cmdlet and not here, or here and not there, cannot ship.

use super::{MapOp, ReduceOp, TextTransform, ZipOp};

/// What a kernel takes as its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputKind {
    /// One array of doubles, which the kernel works on as its own copy.
    Doubles,
    /// One typed double array, changed in place.
    DoublesInPlace,
    /// Two arrays of doubles of one length.
    TwoDoubles,
    /// A list of file paths.
    Paths,
    /// A list of file paths and a list of expected roots of the same
    /// length.
    PathsAndManifest,
    /// One string.
    Text,
}

/// What a kernel answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnswerKind {
    /// An array of doubles.
    Doubles,
    /// Nothing: the input array was changed in place.
    Nothing,
    /// One double.
    Double,
    /// One `Flynnel.Reduction`.
    Reduction,
    /// A `Flynnel.HistogramBin` row per bin, or one `Flynnel.Histogram`
    /// when asked for the whole.
    Histogram,
    /// A `Flynnel.FileHash` row per readable file.
    FileHashes,
    /// A `Flynnel.HashCheck` row per file.
    HashChecks,
    /// A `Flynnel.FileMatch` row per matching line.
    FileMatches,
    /// A `Flynnel.FileMeasure` row per readable file.
    FileMeasures,
    /// A `Flynnel.TextMatch` row per match.
    TextMatches,
    /// One `Flynnel.TextMeasure`.
    TextMeasure,
    /// An array of strings.
    Strings,
    /// One string.
    Text,
}

/// A parameter's type, as the cmdlet declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamType {
    /// An array of doubles.
    Doubles,
    /// A double.
    Double,
    /// An unsigned 32-bit count.
    Count,
    /// A switch.
    Switch,
    /// A string.
    Text,
    /// An array of strings.
    Texts,
    /// Any object; the kernel checks what it needs.
    Object,
    /// A `Flynnel.JobPlan`.
    Plan,
    /// One name out of a fixed set.
    Choice {
        /// The module's enum type for it.
        clr: &'static str,
        /// Every name it takes, in declaration order.
        names: &'static [&'static str],
    },
}

impl ParamType {
    /// The CLR type the module's cmdlet declares for it.
    pub const fn clr(&self) -> &'static str {
        match *self {
            Self::Doubles => "double[]",
            Self::Double => "double",
            Self::Count => "uint",
            Self::Switch => "SwitchParameter",
            Self::Text => "string",
            Self::Texts => "string[]",
            Self::Object => "object",
            Self::Plan => "Flynnel.JobPlan",
            Self::Choice { clr, .. } => clr,
        }
    }
}

/// One parameter of a kernel's cmdlet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamDescriptor {
    /// The parameter's name as PowerShell spells it.
    pub name: &'static str,
    /// Its type.
    pub ty: ParamType,
    /// Whether the cmdlet requires it.
    pub mandatory: bool,
    /// Its position, when it binds by position.
    pub position: Option<u32>,
    /// Whether it takes pipeline input.
    pub from_pipeline: bool,
}

/// One declared kernel: its name, the cmdlet that runs it, and what it
/// takes and answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelDescriptor {
    /// The kernel's name.
    pub kernel: &'static str,
    /// The module's cmdlet for it.
    pub cmdlet: &'static str,
    /// What it takes.
    pub input: InputKind,
    /// What it answers.
    pub answer: AnswerKind,
    /// The cmdlet's parameters, in the order it declares them.
    pub params: &'static [ParamDescriptor],
}

/// A mandatory parameter at `position`.
const fn at(name: &'static str, ty: ParamType, position: u32) -> ParamDescriptor {
    ParamDescriptor {
        name,
        ty,
        mandatory: true,
        position: Some(position),
        from_pipeline: false,
    }
}

/// A mandatory parameter at `position` that takes pipeline input.
const fn piped(name: &'static str, ty: ParamType, position: u32) -> ParamDescriptor {
    ParamDescriptor {
        name,
        ty,
        mandatory: true,
        position: Some(position),
        from_pipeline: true,
    }
}

/// An optional parameter bound by name.
const fn named(name: &'static str, ty: ParamType) -> ParamDescriptor {
    ParamDescriptor {
        name,
        ty,
        mandatory: false,
        position: None,
        from_pipeline: false,
    }
}

const MAP_OP: ParamType = ParamType::Choice {
    clr: "Flynnel.MapOp",
    names: &MapOp::NAMES,
};
const ZIP_OP: ParamType = ParamType::Choice {
    clr: "Flynnel.ZipOp",
    names: &ZipOp::NAMES,
};
const REDUCE_OP: ParamType = ParamType::Choice {
    clr: "Flynnel.ReduceOp",
    names: &ReduceOp::NAMES,
};
const TEXT_TRANSFORM: ParamType = ParamType::Choice {
    clr: "Flynnel.TextTransform",
    names: &TextTransform::NAMES,
};

const PLAN: ParamDescriptor = named("Plan", ParamType::Plan);

const MAP: KernelDescriptor = KernelDescriptor {
    kernel: "Map",
    cmdlet: "Invoke-FlynnelMap",
    input: InputKind::Doubles,
    answer: AnswerKind::Doubles,
    params: &[
        piped("InputObject", ParamType::Doubles, 0),
        at("Operation", MAP_OP, 1),
        named("Min", ParamType::Double),
        named("Max", ParamType::Double),
        named("Factor", ParamType::Double),
        named("Addend", ParamType::Double),
        PLAN,
    ],
};

const MAP_IN_PLACE: KernelDescriptor = KernelDescriptor {
    kernel: "MapInPlace",
    cmdlet: "Update-FlynnelArray",
    input: InputKind::DoublesInPlace,
    answer: AnswerKind::Nothing,
    params: &[
        piped("InputObject", ParamType::Object, 0),
        at("Operation", MAP_OP, 1),
        named("Min", ParamType::Double),
        named("Max", ParamType::Double),
        named("Factor", ParamType::Double),
        named("Addend", ParamType::Double),
        PLAN,
    ],
};

const ZIP: KernelDescriptor = KernelDescriptor {
    kernel: "Zip",
    cmdlet: "Invoke-FlynnelZip",
    input: InputKind::TwoDoubles,
    answer: AnswerKind::Doubles,
    params: &[
        at("Left", ParamType::Doubles, 0),
        at("Right", ParamType::Doubles, 1),
        at("Operation", ZIP_OP, 2),
        PLAN,
    ],
};

const REDUCE: KernelDescriptor = KernelDescriptor {
    kernel: "Reduce",
    cmdlet: "Measure-FlynnelReduce",
    input: InputKind::Doubles,
    answer: AnswerKind::Reduction,
    params: &[
        piped("InputObject", ParamType::Doubles, 0),
        at("Operation", REDUCE_OP, 1),
        named("Min", ParamType::Double),
        named("Max", ParamType::Double),
        PLAN,
    ],
};

const PREFIX_SUM: KernelDescriptor = KernelDescriptor {
    kernel: "PrefixSum",
    cmdlet: "Get-FlynnelPrefixSum",
    input: InputKind::Doubles,
    answer: AnswerKind::Doubles,
    params: &[piped("InputObject", ParamType::Doubles, 0), PLAN],
};

const HISTOGRAM: KernelDescriptor = KernelDescriptor {
    kernel: "Histogram",
    cmdlet: "Get-FlynnelHistogram",
    input: InputKind::Doubles,
    answer: AnswerKind::Histogram,
    params: &[
        piped("InputObject", ParamType::Doubles, 0),
        at("Bins", ParamType::Count, 1),
        named("Min", ParamType::Double),
        named("Max", ParamType::Double),
        named("AsArray", ParamType::Switch),
        PLAN,
    ],
};

const DOT_PRODUCT: KernelDescriptor = KernelDescriptor {
    kernel: "DotProduct",
    cmdlet: "Get-FlynnelDotProduct",
    input: InputKind::TwoDoubles,
    answer: AnswerKind::Double,
    params: &[
        at("Left", ParamType::Doubles, 0),
        at("Right", ParamType::Doubles, 1),
        PLAN,
    ],
};

const SORT: KernelDescriptor = KernelDescriptor {
    kernel: "Sort",
    cmdlet: "Invoke-FlynnelSort",
    input: InputKind::Doubles,
    answer: AnswerKind::Doubles,
    params: &[
        piped("InputObject", ParamType::Doubles, 0),
        named("Descending", ParamType::Switch),
        PLAN,
    ],
};

#[cfg(feature = "verify-chain")]
const FILE_HASH: KernelDescriptor = KernelDescriptor {
    kernel: "FileHash",
    cmdlet: "Measure-FlynnelFileHash",
    input: InputKind::Paths,
    answer: AnswerKind::FileHashes,
    params: &[
        piped("Path", ParamType::Texts, 0),
        PLAN,
        named("UseIoPool", ParamType::Switch),
    ],
};

#[cfg(feature = "verify-chain")]
const FILE_HASH_CHECK: KernelDescriptor = KernelDescriptor {
    kernel: "FileHashCheck",
    cmdlet: "Test-FlynnelFileHash",
    input: InputKind::PathsAndManifest,
    answer: AnswerKind::HashChecks,
    params: &[
        at("Path", ParamType::Texts, 0),
        at("Manifest", ParamType::Texts, 1),
        PLAN,
    ],
};

const SEARCH_FILE: KernelDescriptor = KernelDescriptor {
    kernel: "SearchFile",
    cmdlet: "Search-FlynnelFile",
    input: InputKind::Paths,
    answer: AnswerKind::FileMatches,
    params: &[
        at("Pattern", ParamType::Text, 0),
        piped("Path", ParamType::Texts, 1),
        named("IgnoreCase", ParamType::Switch),
        PLAN,
    ],
};

const FILE_LINE: KernelDescriptor = KernelDescriptor {
    kernel: "FileLine",
    cmdlet: "Measure-FlynnelFileLine",
    input: InputKind::Paths,
    answer: AnswerKind::FileMeasures,
    params: &[piped("Path", ParamType::Texts, 0), PLAN],
};

const FILE_BYTE: KernelDescriptor = KernelDescriptor {
    kernel: "FileByte",
    cmdlet: "Measure-FlynnelFileByte",
    input: InputKind::Paths,
    answer: AnswerKind::FileMeasures,
    params: &[piped("Path", ParamType::Texts, 0), PLAN],
};

const SEARCH_TEXT: KernelDescriptor = KernelDescriptor {
    kernel: "SearchText",
    cmdlet: "Search-FlynnelText",
    input: InputKind::Text,
    answer: AnswerKind::TextMatches,
    params: &[
        piped("Text", ParamType::Text, 0),
        at("Pattern", ParamType::Text, 1),
        PLAN,
    ],
};

const TEXT_COUNT: KernelDescriptor = KernelDescriptor {
    kernel: "TextCount",
    cmdlet: "Measure-FlynnelTextCount",
    input: InputKind::Text,
    answer: AnswerKind::TextMeasure,
    params: &[
        piped("Text", ParamType::Text, 0),
        named("Pattern", ParamType::Text),
        PLAN,
    ],
};

const SPLIT_TEXT: KernelDescriptor = KernelDescriptor {
    kernel: "SplitText",
    cmdlet: "Split-FlynnelText",
    input: InputKind::Text,
    answer: AnswerKind::Strings,
    params: &[
        piped("Text", ParamType::Text, 0),
        at("Separator", ParamType::Text, 1),
        named("NoEmpty", ParamType::Switch),
        PLAN,
    ],
};

const UPDATE_TEXT: KernelDescriptor = KernelDescriptor {
    kernel: "UpdateText",
    cmdlet: "Update-FlynnelText",
    input: InputKind::Text,
    answer: AnswerKind::Text,
    params: &[
        piped("Text", ParamType::Text, 0),
        at("Operation", TEXT_TRANSFORM, 1),
        named("Pattern", ParamType::Text),
        named("Replacement", ParamType::Text),
        PLAN,
    ],
};

/// Every declared kernel this build carries, in the module's order.
#[cfg(feature = "verify-chain")]
pub static DESCRIPTORS: &[KernelDescriptor] = &[
    MAP,
    MAP_IN_PLACE,
    ZIP,
    REDUCE,
    PREFIX_SUM,
    HISTOGRAM,
    DOT_PRODUCT,
    SORT,
    FILE_HASH,
    FILE_HASH_CHECK,
    SEARCH_FILE,
    FILE_LINE,
    FILE_BYTE,
    SEARCH_TEXT,
    TEXT_COUNT,
    SPLIT_TEXT,
    UPDATE_TEXT,
];

/// Every declared kernel this build carries, in the module's order. The
/// two hashing kernels need the `verify-chain` feature.
#[cfg(not(feature = "verify-chain"))]
pub static DESCRIPTORS: &[KernelDescriptor] = &[
    MAP,
    MAP_IN_PLACE,
    ZIP,
    REDUCE,
    PREFIX_SUM,
    HISTOGRAM,
    DOT_PRODUCT,
    SORT,
    SEARCH_FILE,
    FILE_LINE,
    FILE_BYTE,
    SEARCH_TEXT,
    TEXT_COUNT,
    SPLIT_TEXT,
    UPDATE_TEXT,
];

/// The descriptor of the kernel named `kernel`, or of the kernel whose
/// cmdlet is named `kernel`, compared without regard to ASCII case.
pub fn descriptor_for(kernel: &str) -> Option<&'static KernelDescriptor> {
    DESCRIPTORS
        .iter()
        .find(|d| d.kernel.eq_ignore_ascii_case(kernel) || d.cmdlet.eq_ignore_ascii_case(kernel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_is_described_once_and_found_by_either_name() {
        let mut names: Vec<&str> = DESCRIPTORS.iter().map(|d| d.kernel).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), DESCRIPTORS.len());
        #[cfg(feature = "verify-chain")]
        assert_eq!(DESCRIPTORS.len(), 17);
        assert_eq!(
            descriptor_for("map").map(|d| d.cmdlet),
            Some("Invoke-FlynnelMap")
        );
        assert_eq!(
            descriptor_for("Update-FlynnelText").map(|d| d.kernel),
            Some("UpdateText")
        );
        assert!(descriptor_for("NoSuchKernel").is_none());
        for d in DESCRIPTORS {
            let positions: Vec<u32> = d.params.iter().filter_map(|p| p.position).collect();
            let expected: Vec<u32> = (0..positions.len() as u32).collect();
            assert_eq!(
                positions, expected,
                "{} numbers its positions from zero",
                d.kernel
            );
        }
    }
}
