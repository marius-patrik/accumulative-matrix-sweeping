use crate::checked::{add, mul};
use crate::{
    AmsError, ErrorCode, GatedMlpPlan, GatedMlpReaders, GatedMlpScratch,
    GatedMlpScratchRequirements, GlmRouterPlan, GlmRouterScratch, LinearPlan, RangeReader,
    glm_gated_mlp, glm_route_experts, stream_linear,
};

/// Immutable native plan for one GLM routed-plus-shared sparse MLP token.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMoePlan {
    router_linear: LinearPlan,
    router: GlmRouterPlan,
    shared: GatedMlpPlan,
    hidden_elements: usize,
    scratch: SparseMoeScratchRequirements,
}

impl SparseMoePlan {
    /// Validate router, routed-expert, and shared-expert shapes and resource bounds.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for incomplete/inconsistent expert inventory or checked overflow.
    #[allow(clippy::too_many_lines)] // Validation and high-water derivation form one proof.
    pub fn new(
        router_linear: LinearPlan,
        router: GlmRouterPlan,
        expert_plans: &[GatedMlpPlan],
        shared: GatedMlpPlan,
    ) -> Result<Self, AmsError> {
        if router_linear.rows() != router.expert_count()
            || expert_plans.len() != router.expert_count()
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "sparse MoE router and expert inventory differ",
            ));
        }
        let hidden_elements = router_linear.columns();
        if shared.input_elements() != hidden_elements
            || shared.output_elements() != hidden_elements
            || expert_plans.iter().any(|plan| {
                plan.input_elements() != hidden_elements
                    || plan.output_elements() != hidden_elements
            })
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "sparse MoE expert hidden dimensions differ",
            ));
        }
        let mut mlp = shared.scratch();
        for expert in expert_plans {
            mlp = mlp.union(expert.scratch())?;
        }
        let linear = mlp.linear.union(router_linear.scratch())?;
        let intermediate_bytes = mul(
            mlp.intermediate_elements,
            size_of::<f64>(),
            "sparse MoE intermediate bytes overflow",
        )?;
        mlp = GatedMlpScratchRequirements {
            linear,
            intermediate_elements: mlp.intermediate_elements,
            total_bytes: add(
                linear.total_bytes,
                intermediate_bytes,
                "sparse MoE MLP scratch bytes overflow",
            )?,
        };
        let router_logits_elements = router.expert_count();
        let selected_expert_elements = router.selected_count();
        let expert_buffer_elements = mul(
            hidden_elements,
            2,
            "sparse MoE expert buffer elements overflow",
        )?;
        let router_logits_bytes = mul(
            router_logits_elements,
            size_of::<f64>(),
            "sparse MoE router logits bytes overflow",
        )?;
        let selected_bytes = add(
            mul(
                selected_expert_elements,
                size_of::<usize>(),
                "sparse MoE selected index bytes overflow",
            )?,
            mul(
                selected_expert_elements,
                size_of::<f64>(),
                "sparse MoE selected weight bytes overflow",
            )?,
            "sparse MoE selected bytes overflow",
        )?;
        let expert_buffer_bytes = mul(
            expert_buffer_elements,
            size_of::<f64>(),
            "sparse MoE expert buffer bytes overflow",
        )?;
        let total_bytes = add(
            add(
                add(
                    add(
                        mlp.total_bytes,
                        router.scratch_bytes(),
                        "sparse MoE MLP and routing scratch overflow",
                    )?,
                    router_logits_bytes,
                    "sparse MoE router logits total overflow",
                )?,
                selected_bytes,
                "sparse MoE selected total overflow",
            )?,
            expert_buffer_bytes,
            "sparse MoE expert buffer total overflow",
        )?;
        Ok(Self {
            router_linear,
            router,
            shared,
            hidden_elements,
            scratch: SparseMoeScratchRequirements {
                mlp,
                router_logits_elements,
                router_probability_elements: router.expert_count(),
                router_corrected_elements: router.expert_count(),
                router_group_score_elements: router.group_count(),
                router_selected_group_elements: router.top_groups(),
                selected_expert_elements,
                expert_buffer_elements,
                total_bytes,
            },
        })
    }

    /// Exact logical caller-owned scratch required by this plan.
    #[must_use]
    pub const fn scratch(&self) -> SparseMoeScratchRequirements {
        self.scratch
    }

    /// Logical hidden width consumed and produced by the routed-plus-shared MLP.
    #[must_use]
    pub const fn hidden_elements(&self) -> usize {
        self.hidden_elements
    }

    /// Number of routed experts whose plans and readers must be bound.
    #[must_use]
    pub const fn expert_count(&self) -> usize {
        self.router.expert_count()
    }

    fn inventory_admits(&self, bindings: &SparseMoeBindings<'_, '_>) -> bool {
        bindings.expert_plans.len() == self.router.expert_count()
            && bindings.expert_readers.len() == self.router.expert_count()
            && bindings
                .expert_plans
                .iter()
                .all(|expert| binding_is_admitted(self, expert))
    }

    pub(crate) fn bindings_admit(&self, bindings: &SparseMoeBindings<'_, '_>) -> bool {
        self.inventory_admits(bindings)
            && self.router_linear.reader_end() <= bindings.router.len()
            && self.shared.readers_admit(bindings.shared_readers)
            && bindings
                .expert_plans
                .iter()
                .zip(bindings.expert_readers)
                .all(|(expert, readers)| expert.readers_admit(readers))
    }
}

/// Caller-owned scratch regions and high-water for one sparse-MoE token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseMoeScratchRequirements {
    /// Reusable router/MLP linear scratch and gated intermediates.
    pub mlp: GatedMlpScratchRequirements,
    /// Router-logit FP64 elements.
    pub router_logits_elements: usize,
    /// Unbiased sigmoid-probability FP64 elements.
    pub router_probability_elements: usize,
    /// Bias-corrected selection-score FP64 elements.
    pub router_corrected_elements: usize,
    /// Group-score FP64 elements.
    pub router_group_score_elements: usize,
    /// Retained group indices.
    pub router_selected_group_elements: usize,
    /// Selected expert indices and weights in each corresponding output region.
    pub selected_expert_elements: usize,
    /// Combined expert-output and transactional-accumulator FP64 elements.
    pub expert_buffer_elements: usize,
    /// Sum of all simultaneously resident scratch and local bytes.
    pub total_bytes: usize,
}

/// Storage bindings for the router, complete routed inventory, and shared expert.
pub struct SparseMoeBindings<'reader, 'slice> {
    router: &'reader dyn RangeReader,
    expert_plans: &'slice [GatedMlpPlan],
    expert_readers: &'slice [GatedMlpReaders<'reader>],
    shared_readers: &'slice GatedMlpReaders<'reader>,
}

impl<'reader, 'slice> SparseMoeBindings<'reader, 'slice> {
    /// Bind the exact expert inventory without opening or reading unselected experts.
    #[must_use]
    pub const fn new(
        router: &'reader dyn RangeReader,
        expert_plans: &'slice [GatedMlpPlan],
        expert_readers: &'slice [GatedMlpReaders<'reader>],
        shared_readers: &'slice GatedMlpReaders<'reader>,
    ) -> Self {
        Self {
            router,
            expert_plans,
            expert_readers,
            shared_readers,
        }
    }
}

/// Caller-owned scratch for allocation-free sparse-MoE execution.
pub struct SparseMoeScratch<'a> {
    mlp: GatedMlpScratch<'a>,
    router_logits: &'a mut [f64],
    routing: GlmRouterScratch<'a>,
    expert_indices: &'a mut [usize],
    expert_weights: &'a mut [f64],
    expert_output: &'a mut [f64],
    accumulator: &'a mut [f64],
}

impl<'a> SparseMoeScratch<'a> {
    /// Group every preallocated sparse-MoE scratch region under one borrow.
    #[must_use]
    pub const fn new(
        mlp: GatedMlpScratch<'a>,
        router_logits: &'a mut [f64],
        routing: GlmRouterScratch<'a>,
        expert_indices: &'a mut [usize],
        expert_weights: &'a mut [f64],
        expert_output: &'a mut [f64],
        accumulator: &'a mut [f64],
    ) -> Self {
        Self {
            mlp,
            router_logits,
            routing,
            expert_indices,
            expert_weights,
            expert_output,
            accumulator,
        }
    }

    pub(crate) const fn admits(&self, plan: &SparseMoePlan) -> bool {
        let requirement = plan.scratch;
        let hidden_buffer = requirement.expert_buffer_elements / 2;
        self.mlp.admits(requirement.mlp)
            && self.routing.admits(plan.router)
            && self.router_logits.len() >= requirement.router_logits_elements
            && self.expert_indices.len() >= requirement.selected_expert_elements
            && self.expert_weights.len() >= requirement.selected_expert_elements
            && self.expert_output.len() >= hidden_buffer
            && self.accumulator.len() >= hidden_buffer
    }
}

const fn binding_is_admitted(plan: &SparseMoePlan, expert: &GatedMlpPlan) -> bool {
    expert.input_elements() == plan.hidden_elements
        && expert.output_elements() == plan.hidden_elements
        && plan.scratch.mlp.admits(expert.scratch())
}

/// Execute one GLM routed-plus-shared sparse MLP token from bounded range readers.
///
/// Only selected routed experts are read. Output remains untouched until every routed
/// and shared expert completes successfully.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
#[allow(clippy::too_many_lines)] // Preflight and transactional execution share one resource proof.
pub fn glm_sparse_moe(
    plan: &SparseMoePlan,
    bindings: &SparseMoeBindings<'_, '_>,
    input: &[f64],
    correction_bias: &[f64],
    scratch: &mut SparseMoeScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    if input.len() != plan.hidden_elements
        || output.len() != plan.hidden_elements
        || correction_bias.len() != plan.router.expert_count()
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "sparse MoE input, output, or correction-bias dimensions differ",
        ));
    }
    if input.iter().any(|value| !value.is_finite())
        || correction_bias.iter().any(|value| !value.is_finite())
    {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "sparse MoE input or correction bias is non-finite",
        ));
    }
    if !plan.inventory_admits(bindings) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "sparse MoE execution bindings differ from the admitted inventory",
        ));
    }
    if !plan.bindings_admit(bindings) {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "sparse MoE execution binding range exceeds its storage object",
        ));
    }
    let requirement = plan.scratch;
    let hidden_buffer = requirement.expert_buffer_elements / 2;
    if !scratch.admits(plan) {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "sparse MoE scratch is smaller than the admitted plan",
        ));
    }
    let router_logits = &mut scratch.router_logits[..requirement.router_logits_elements];
    stream_linear(
        bindings.router,
        plan.router_linear,
        input,
        None,
        scratch.mlp.linear_mut(),
        router_logits,
    )?;
    let expert_indices = &mut scratch.expert_indices[..requirement.selected_expert_elements];
    let expert_weights = &mut scratch.expert_weights[..requirement.selected_expert_elements];
    glm_route_experts(
        plan.router,
        router_logits,
        correction_bias,
        &mut scratch.routing,
        expert_indices,
        expert_weights,
    )?;
    let expert_output = &mut scratch.expert_output[..hidden_buffer];
    let accumulator = &mut scratch.accumulator[..hidden_buffer];
    accumulator.fill(0.0);
    for (&expert_index, &expert_weight) in expert_indices.iter().zip(expert_weights.iter()) {
        let expert_plan = bindings.expert_plans.get(expert_index).ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "router selected an expert outside the admitted inventory",
            )
        })?;
        let expert_readers = bindings.expert_readers.get(expert_index).ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "selected expert reader is absent after inventory validation",
            )
        })?;
        glm_gated_mlp(
            *expert_plan,
            expert_readers,
            input,
            &mut scratch.mlp,
            expert_output,
        )?;
        for (destination, value) in accumulator.iter_mut().zip(expert_output.iter()) {
            *destination += expert_weight * value;
            if !destination.is_finite() {
                return Err(AmsError::new(
                    ErrorCode::NumericFailure,
                    "sparse MoE routed accumulation is non-finite",
                ));
            }
        }
    }
    glm_gated_mlp(
        plan.shared,
        bindings.shared_readers,
        input,
        &mut scratch.mlp,
        expert_output,
    )?;
    for (routed, shared) in accumulator.iter_mut().zip(expert_output.iter()) {
        *routed += shared;
        if !routed.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "sparse MoE output is non-finite",
            ));
        }
    }
    output.copy_from_slice(accumulator);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{IdentityDType, IdentityLinearPlan, LinearScratch};

    struct CountingReader {
        bytes: Vec<u8>,
        reads: Cell<usize>,
    }

    impl CountingReader {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                reads: Cell::new(0),
            }
        }

        fn reads(&self) -> usize {
            self.reads.get()
        }
    }

    impl RangeReader for CountingReader {
        fn len(&self) -> u64 {
            u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
            let start = usize::try_from(offset).map_err(|_| {
                AmsError::new(ErrorCode::IoFailure, "test reader offset exceeds usize")
            })?;
            let end = start
                .checked_add(destination.len())
                .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "test reader range overflow"))?;
            let source = self.bytes.get(start..end).ok_or_else(|| {
                AmsError::new(ErrorCode::IoFailure, "test reader range exceeds object")
            })?;
            destination.copy_from_slice(source);
            self.reads.set(self.reads.get().saturating_add(1));
            Ok(())
        }
    }

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn mlp_payload(scale: f32) -> Vec<u8> {
        encode_f32(&[1.0, 0.0, 1.0, 0.0, scale, -scale])
    }

    fn tiny_mlp_plan() -> Result<GatedMlpPlan, AmsError> {
        let gate = IdentityLinearPlan::from_arena(1, 2, 0, 16, IdentityDType::Float32)?;
        let up = IdentityLinearPlan::from_arena(1, 2, 8, 16, IdentityDType::Float32)?;
        let down = IdentityLinearPlan::from_arena(2, 1, 16, 12, IdentityDType::Float32)?;
        GatedMlpPlan::new(gate.into(), up.into(), down.into())
    }

    #[test]
    #[allow(clippy::suboptimal_flops, clippy::too_many_lines)]
    // One fixture proves preflight rejection, selective reads, and source-order parity.
    fn sparse_moe_reads_only_selected_expert_and_adds_shared_output() -> Result<(), AmsError> {
        let router_reader = CountingReader::new(encode_f32(&[0.0; 8]));
        let expert_zero = CountingReader::new(mlp_payload(1.0));
        let expert_one = CountingReader::new(mlp_payload(2.0));
        let expert_two = CountingReader::new(mlp_payload(3.0));
        let expert_three = CountingReader::new(mlp_payload(4.0));
        let shared = CountingReader::new(mlp_payload(10.0));
        let router_linear = IdentityLinearPlan::from_arena(4, 2, 0, 16, IdentityDType::Float32)?;
        let router = GlmRouterPlan::new(4, 1, 2, 1, 1.0)?;
        let expert_plan = tiny_mlp_plan()?;
        let expert_plans = [expert_plan; 4];
        let plan = SparseMoePlan::new(router_linear.into(), router, &expert_plans, expert_plan)?;
        let requirement = plan.scratch();
        assert_eq!(requirement.mlp.total_bytes, 32);
        assert_eq!(requirement.total_bytes, 200);

        let expert_readers = [
            GatedMlpReaders::new(&expert_zero, &expert_zero, &expert_zero),
            GatedMlpReaders::new(&expert_one, &expert_one, &expert_one),
            GatedMlpReaders::new(&expert_two, &expert_two, &expert_two),
            GatedMlpReaders::new(&expert_three, &expert_three, &expert_three),
        ];
        let shared_readers = GatedMlpReaders::new(&shared, &shared, &shared);
        let bindings = SparseMoeBindings::new(
            &router_reader,
            &expert_plans,
            &expert_readers,
            &shared_readers,
        );
        {
            let mut encoded = [0u8; 8];
            let mut decoded = [0.0f32; 0];
            let mut linear_accumulators = [0.0f64; 0];
            let linear = LinearScratch::new(&mut encoded, &mut decoded, &mut linear_accumulators);
            let mut gate = [0.0f64; 1];
            let mut up = [0.0f64; 1];
            let mlp = GatedMlpScratch::new(linear, &mut gate, &mut up);
            let mut short_router_logits = [0.0f64; 3];
            let mut probabilities = [0.0f64; 4];
            let mut corrected = [0.0f64; 4];
            let mut group_scores = [0.0f64; 2];
            let mut selected_groups = [usize::MAX; 1];
            let routing = GlmRouterScratch::new(
                &mut probabilities,
                &mut corrected,
                &mut group_scores,
                &mut selected_groups,
            );
            let mut expert_indices = [usize::MAX; 1];
            let mut expert_weights = [0.0f64; 1];
            let mut expert_output = [0.0f64; 2];
            let mut accumulator = [0.0f64; 2];
            let mut short_scratch = SparseMoeScratch::new(
                mlp,
                &mut short_router_logits,
                routing,
                &mut expert_indices,
                &mut expert_weights,
                &mut expert_output,
                &mut accumulator,
            );
            let mut rejected_output = [0.0f64; 2];
            let error = glm_sparse_moe(
                &plan,
                &bindings,
                &[1.0, 2.0],
                &[0.0, 0.0, 10.0, 9.0],
                &mut short_scratch,
                &mut rejected_output,
            )
            .err();
            assert_eq!(
                error.map(AmsError::code),
                Some(ErrorCode::PreflightNoWorkingSet)
            );
        }
        assert_eq!(router_reader.reads(), 0);
        assert_eq!(expert_zero.reads(), 0);
        assert_eq!(expert_one.reads(), 0);
        assert_eq!(expert_two.reads(), 0);
        assert_eq!(expert_three.reads(), 0);
        assert_eq!(shared.reads(), 0);
        let mut encoded = [0u8; 8];
        let mut decoded = [0.0f32; 0];
        let mut linear_accumulators = [0.0f64; 0];
        let linear = LinearScratch::new(&mut encoded, &mut decoded, &mut linear_accumulators);
        let mut gate = [0.0f64; 1];
        let mut up = [0.0f64; 1];
        let mlp = GatedMlpScratch::new(linear, &mut gate, &mut up);
        let mut router_logits = [0.0f64; 4];
        let mut probabilities = [0.0f64; 4];
        let mut corrected = [0.0f64; 4];
        let mut group_scores = [0.0f64; 2];
        let mut selected_groups = [usize::MAX; 1];
        let routing = GlmRouterScratch::new(
            &mut probabilities,
            &mut corrected,
            &mut group_scores,
            &mut selected_groups,
        );
        let mut expert_indices = [usize::MAX; 1];
        let mut expert_weights = [0.0f64; 1];
        let mut expert_output = [0.0f64; 2];
        let mut accumulator = [0.0f64; 2];
        let mut scratch = SparseMoeScratch::new(
            mlp,
            &mut router_logits,
            routing,
            &mut expert_indices,
            &mut expert_weights,
            &mut expert_output,
            &mut accumulator,
        );
        let mut output = [0.0f64; 2];
        glm_sparse_moe(
            &plan,
            &bindings,
            &[1.0, 2.0],
            &[0.0, 0.0, 10.0, 9.0],
            &mut scratch,
            &mut output,
        )?;
        let activation = crate::glm_silu(1.0)?;
        assert!((output[0] - 13.0 * activation).abs() <= 1e-14);
        assert!((output[1] + 13.0 * activation).abs() <= 1e-14);
        assert!(router_reader.reads() > 0);
        assert_eq!(expert_zero.reads(), 0);
        assert_eq!(expert_one.reads(), 0);
        assert!(expert_two.reads() > 0);
        assert_eq!(expert_three.reads(), 0);
        assert!(shared.reads() > 0);

        let bad_shared = CountingReader::new(mlp_payload(f32::NAN));
        let bad_shared_readers = GatedMlpReaders::new(&bad_shared, &bad_shared, &bad_shared);
        let bad_bindings = SparseMoeBindings::new(
            &router_reader,
            &expert_plans,
            &expert_readers,
            &bad_shared_readers,
        );
        let mut rejected_output = [7.0f64, 7.0];
        let error = glm_sparse_moe(
            &plan,
            &bad_bindings,
            &[1.0, 2.0],
            &[0.0, 0.0, 10.0, 9.0],
            &mut scratch,
            &mut rejected_output,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert!(
            rejected_output
                .iter()
                .all(|value| value.to_bits() == 7.0f64.to_bits())
        );
        Ok(())
    }

    #[test]
    fn sparse_moe_rejects_an_incomplete_expert_inventory() -> Result<(), AmsError> {
        let router_linear = IdentityLinearPlan::from_arena(4, 2, 0, 16, IdentityDType::Float32)?;
        let router = GlmRouterPlan::new(4, 1, 2, 1, 1.0)?;
        let expert = tiny_mlp_plan()?;
        let error = SparseMoePlan::new(router_linear.into(), router, &[expert; 3], expert).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        Ok(())
    }
}
