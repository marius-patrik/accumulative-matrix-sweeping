//! Allocation-bounded native primitives for AMS.
//!
//! The crate intentionally starts without third-party dependencies. Callers own every
//! scratch buffer used by the execution path, making resource admission inspectable.

#![forbid(unsafe_code)]

mod checked;
mod error;
mod full_attention;
mod gated_mlp;
mod glm;
mod identity;
mod linear;
mod reader;
mod sparse_attention;
mod sparse_moe;
mod streamed_dsa;
mod ternary;

pub use error::{AmsError, ErrorCode};
pub use full_attention::{
    FullAttentionPlan, FullAttentionReaders, FullAttentionScratch,
    FullAttentionScratchRequirements, FullAttentionShape, FullKvLayout, glm_full_attention,
};
pub use gated_mlp::{
    GatedMlpPlan, GatedMlpReaders, GatedMlpScratch, GatedMlpScratchRequirements, glm_gated_mlp,
};
pub use glm::{
    DsaTopKPlan, GlmRouterPlan, GlmRouterScratch, glm_dsa_topk, glm_layer_norm, glm_rms_norm,
    glm_rope_half_split, glm_rope_interleaved, glm_route_experts, glm_silu, glm_softmax,
};
pub use identity::{
    IdentityDType, IdentityLinearPlan, IdentityScratch, IdentityScratchRequirements,
    read_identity_vector, stream_linear_identity,
};
pub use linear::{LinearPlan, LinearScratch, LinearScratchRequirements, stream_linear};
pub use reader::{FileRangeReader, RangeReader, SliceReader};
pub use sparse_attention::{
    SparseAttentionPlan, SparseAttentionReaders, SparseAttentionScratch,
    SparseAttentionScratchRequirements, SparseAttentionShape, SparseKvLayout, glm_sparse_attention,
};
pub use sparse_moe::{
    SparseMoeBindings, SparseMoePlan, SparseMoeScratch, SparseMoeScratchRequirements,
    glm_sparse_moe,
};
pub use streamed_dsa::{
    StreamedDsaLayout, StreamedDsaScratch, StreamedDsaScratchRequirements, StreamedDsaShape,
    StreamedDsaTopKPlan, glm_streamed_dsa_topk,
};
pub use ternary::{
    TernaryConfig, TernaryLinearPlan, TernaryScratch, TernaryScratchRequirements,
    decode_ternary_group, stream_linear_ternary,
};
