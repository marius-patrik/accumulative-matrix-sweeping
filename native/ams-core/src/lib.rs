//! Allocation-bounded native primitives for AMS.
//!
//! The crate intentionally starts without third-party dependencies. Callers own every
//! scratch buffer used by the execution path, making resource admission inspectable.

#![forbid(unsafe_code)]

mod checked;
mod error;
mod glm;
mod identity;
mod reader;
mod ternary;

pub use error::{AmsError, ErrorCode};
pub use glm::{
    DsaTopKPlan, GlmRouterPlan, GlmRouterScratch, glm_dsa_topk, glm_layer_norm, glm_rms_norm,
    glm_rope_half_split, glm_rope_interleaved, glm_route_experts, glm_silu, glm_softmax,
};
pub use identity::{
    IdentityDType, IdentityLinearPlan, IdentityScratch, IdentityScratchRequirements,
    stream_linear_identity,
};
pub use reader::{FileRangeReader, RangeReader, SliceReader};
pub use ternary::{
    TernaryConfig, TernaryLinearPlan, TernaryScratch, TernaryScratchRequirements,
    decode_ternary_group, stream_linear_ternary,
};
