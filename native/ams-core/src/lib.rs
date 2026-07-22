//! Allocation-bounded native primitives for AMS.
//!
//! The crate intentionally starts without third-party dependencies. Callers own every
//! scratch buffer used by the execution path, making resource admission inspectable.

#![forbid(unsafe_code)]

mod checked;
mod error;
mod identity;
mod reader;
mod ternary;

pub use error::{AmsError, ErrorCode};
pub use identity::{
    IdentityDType, IdentityLinearPlan, IdentityScratch, IdentityScratchRequirements,
    stream_linear_identity,
};
pub use reader::{FileRangeReader, RangeReader, SliceReader};
pub use ternary::{
    TernaryConfig, TernaryLinearPlan, TernaryScratch, TernaryScratchRequirements,
    decode_ternary_group, stream_linear_ternary,
};
