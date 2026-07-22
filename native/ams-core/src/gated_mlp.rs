use crate::checked::{add, mul};
use crate::{
    AmsError, ErrorCode, LinearPlan, LinearScratch, LinearScratchRequirements, RangeReader,
    glm_silu, stream_linear,
};

/// Immutable native plan for one GLM gate/up/activation/down MLP.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedMlpPlan {
    gate: LinearPlan,
    up: LinearPlan,
    down: LinearPlan,
    input_elements: usize,
    intermediate_elements: usize,
    output_elements: usize,
    scratch: GatedMlpScratchRequirements,
}

impl GatedMlpPlan {
    /// Validate the three matrix shapes and derive their reusable scratch high-water.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for inconsistent shapes or checked-size overflow.
    pub fn new(gate: LinearPlan, up: LinearPlan, down: LinearPlan) -> Result<Self, AmsError> {
        if gate.rows() != up.rows() || gate.columns() != up.columns() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "gated MLP gate and up matrix dimensions differ",
            ));
        }
        if down.columns() != gate.rows() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "gated MLP down reduction differs from the intermediate width",
            ));
        }
        let linear = gate.scratch().union(up.scratch())?.union(down.scratch())?;
        let intermediate_buffer_elements = mul(
            gate.rows(),
            2,
            "gated MLP intermediate element count overflow",
        )?;
        let intermediate_bytes = mul(
            intermediate_buffer_elements,
            size_of::<f64>(),
            "gated MLP intermediate bytes overflow",
        )?;
        let total_bytes = add(
            linear.total_bytes,
            intermediate_bytes,
            "gated MLP total scratch bytes overflow",
        )?;
        Ok(Self {
            gate,
            up,
            down,
            input_elements: gate.columns(),
            intermediate_elements: gate.rows(),
            output_elements: down.rows(),
            scratch: GatedMlpScratchRequirements {
                linear,
                intermediate_elements: intermediate_buffer_elements,
                total_bytes,
            },
        })
    }

    /// Exact logical caller-owned scratch required by this plan.
    #[must_use]
    pub const fn scratch(self) -> GatedMlpScratchRequirements {
        self.scratch
    }

    /// Logical input width.
    #[must_use]
    pub const fn input_elements(self) -> usize {
        self.input_elements
    }

    /// Logical output width.
    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }
}

/// Reusable scratch and logical high-water for one gated MLP.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedMlpScratchRequirements {
    /// Shared storage-linear scratch regions.
    pub linear: LinearScratchRequirements,
    /// FP64 gate and up elements resident together.
    pub intermediate_elements: usize,
    /// Sum of linear scratch, local bytes, and both intermediate vectors.
    pub total_bytes: usize,
}

impl GatedMlpScratchRequirements {
    /// Combine sequential MLP requirements for one reusable scratch allocation.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` if the combined byte count overflows.
    pub fn union(self, other: Self) -> Result<Self, AmsError> {
        let linear = self.linear.union(other.linear)?;
        let intermediate_elements = self.intermediate_elements.max(other.intermediate_elements);
        let intermediate_bytes = mul(
            intermediate_elements,
            size_of::<f64>(),
            "gated MLP union intermediate bytes overflow",
        )?;
        let total_bytes = add(
            linear.total_bytes,
            intermediate_bytes,
            "gated MLP union total bytes overflow",
        )?;
        Ok(Self {
            linear,
            intermediate_elements,
            total_bytes,
        })
    }

    /// Whether this allocation admits another sequential MLP requirement.
    #[must_use]
    pub const fn admits(self, other: Self) -> bool {
        self.linear.encoded_bytes >= other.linear.encoded_bytes
            && self.linear.decoded_elements >= other.linear.decoded_elements
            && self.linear.accumulator_elements >= other.linear.accumulator_elements
            && self.linear.local_bytes >= other.linear.local_bytes
            && self.intermediate_elements >= other.intermediate_elements
    }
}

/// Three immutable storage objects used by a gated MLP plan.
pub struct GatedMlpReaders<'a> {
    gate: &'a dyn RangeReader,
    up: &'a dyn RangeReader,
    down: &'a dyn RangeReader,
}

impl<'a> GatedMlpReaders<'a> {
    /// Bind gate, up, and down storage readers in plan order.
    #[must_use]
    pub const fn new(
        gate: &'a dyn RangeReader,
        up: &'a dyn RangeReader,
        down: &'a dyn RangeReader,
    ) -> Self {
        Self { gate, up, down }
    }
}

/// Caller-owned scratch for mixed-storage gated MLP execution.
pub struct GatedMlpScratch<'a> {
    linear: LinearScratch<'a>,
    gate: &'a mut [f64],
    up: &'a mut [f64],
}

impl<'a> GatedMlpScratch<'a> {
    /// Group reusable linear scratch and the two live intermediate vectors.
    #[must_use]
    pub const fn new(linear: LinearScratch<'a>, gate: &'a mut [f64], up: &'a mut [f64]) -> Self {
        Self { linear, gate, up }
    }

    /// Borrow the reusable linear region for a composed crate-internal operator.
    pub(crate) const fn linear_mut(&mut self) -> &mut LinearScratch<'a> {
        &mut self.linear
    }

    /// Whether these caller-owned regions satisfy one admitted MLP requirement.
    pub(crate) const fn admits(&self, requirement: GatedMlpScratchRequirements) -> bool {
        let per_intermediate = requirement.intermediate_elements / 2;
        self.linear.admits(requirement.linear)
            && self.gate.len() >= per_intermediate
            && self.up.len() >= per_intermediate
    }
}

/// Execute one GLM gated MLP directly from mixed identity or ternary storage.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error. Callers must
/// commit the output only after this operation succeeds.
pub fn glm_gated_mlp(
    plan: GatedMlpPlan,
    readers: &GatedMlpReaders<'_>,
    input: &[f64],
    scratch: &mut GatedMlpScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    if input.len() != plan.input_elements || output.len() != plan.output_elements {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "gated MLP input or output dimensions differ from the plan",
        ));
    }
    if scratch.gate.len() < plan.intermediate_elements
        || scratch.up.len() < plan.intermediate_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "gated MLP intermediate scratch is smaller than the admitted plan",
        ));
    }
    let gate = &mut scratch.gate[..plan.intermediate_elements];
    let up = &mut scratch.up[..plan.intermediate_elements];
    stream_linear(
        readers.gate,
        plan.gate,
        input,
        None,
        &mut scratch.linear,
        gate,
    )?;
    stream_linear(readers.up, plan.up, input, None, &mut scratch.linear, up)?;
    for (gate_value, up_value) in gate.iter_mut().zip(up) {
        *gate_value = glm_silu(*gate_value)? * *up_value;
        if !gate_value.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "gated MLP activation is non-finite",
            ));
        }
    }
    stream_linear(
        readers.down,
        plan.down,
        gate,
        None,
        &mut scratch.linear,
        output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IdentityDType, IdentityLinearPlan, SliceReader, TernaryConfig, TernaryLinearPlan};

    fn ternary_record(scale: f32) -> [u8; 5] {
        let mut record = [0u8; 5];
        record[..4].copy_from_slice(&scale.to_le_bytes());
        record[4] = 225;
        record
    }

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn encode_bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| {
                let word = u16::try_from(value.to_bits() >> 16).unwrap_or_default();
                word.to_le_bytes()
            })
            .collect()
    }

    #[test]
    #[allow(clippy::suboptimal_flops)] // Match source-order multiply/add in the semantic oracle.
    fn mixed_storage_gated_mlp_matches_materialized_reference() -> Result<(), AmsError> {
        let mut gate_payload = Vec::new();
        for _ in 0..3 {
            gate_payload.extend_from_slice(&ternary_record(1.0));
        }
        let up_weights = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let down_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.5, -0.5, 2.0];
        let up_payload = encode_f32(&up_weights);
        let down_payload = encode_bf16(&down_weights);
        let gate_reader = SliceReader::new(&gate_payload);
        let up_reader = SliceReader::new(&up_payload);
        let down_reader = SliceReader::new(&down_payload);
        let gate_plan = TernaryLinearPlan::from_arena(5, 3, 0, 41, TernaryConfig::new(5)?)?;
        let up_plan = IdentityLinearPlan::from_arena(5, 3, 0, 20, IdentityDType::Float32)?;
        let down_plan = IdentityLinearPlan::from_arena(2, 5, 0, 18, IdentityDType::BFloat16)?;
        let plan = GatedMlpPlan::new(gate_plan.into(), up_plan.into(), down_plan.into())?;
        let requirement = plan.scratch();
        assert_eq!(requirement.linear.encoded_bytes, 12);
        assert_eq!(requirement.linear.decoded_elements, 5);
        assert_eq!(requirement.linear.accumulator_elements, 2);
        assert_eq!(requirement.linear.total_bytes, 56);
        assert_eq!(requirement.intermediate_elements, 10);
        assert_eq!(requirement.total_bytes, 136);

        let mut encoded = [0u8; 12];
        let mut decoded = [0.0f32; 5];
        let mut accumulators = [0.0f64; 2];
        let linear = LinearScratch::new(&mut encoded, &mut decoded, &mut accumulators);
        let mut gate = [0.0f64; 5];
        let mut up = [0.0f64; 5];
        let mut scratch = GatedMlpScratch::new(linear, &mut gate, &mut up);
        let readers = GatedMlpReaders::new(&gate_reader, &up_reader, &down_reader);
        let input = [1.0, 2.0, -1.0];
        let mut output = [0.0f64; 2];
        glm_gated_mlp(plan, &readers, &input, &mut scratch, &mut output)?;

        let gate_weights = [
            -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0,
        ];
        let mut activated = [0.0f64; 5];
        for row in 0..5 {
            let mut gate_sum = 0.0;
            let mut up_sum = 0.0;
            for column in 0..3 {
                gate_sum += gate_weights[row * 3 + column] * input[column];
                up_sum += f64::from(up_weights[row * 3 + column]) * input[column];
            }
            activated[row] = glm_silu(gate_sum)? * up_sum;
        }
        let mut expected = [0.0f64; 2];
        for row in 0..2 {
            for column in 0..5 {
                expected[row] += f64::from(down_weights[row * 5 + column]) * activated[column];
            }
        }
        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1e-14)
        );
        Ok(())
    }

    #[test]
    fn gated_mlp_rejects_inconsistent_matrix_shapes() -> Result<(), AmsError> {
        let gate = IdentityLinearPlan::from_arena(5, 3, 0, 20, IdentityDType::Float32)?;
        let up = IdentityLinearPlan::from_arena(4, 3, 0, 20, IdentityDType::Float32)?;
        let down = IdentityLinearPlan::from_arena(2, 5, 0, 20, IdentityDType::Float32)?;
        let error = GatedMlpPlan::new(gate.into(), up.into(), down.into()).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        Ok(())
    }
}
