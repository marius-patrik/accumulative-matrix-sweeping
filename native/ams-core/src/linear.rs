use crate::checked::{add, mul};
use crate::{
    AmsError, ErrorCode, IdentityLinearPlan, IdentityScratch, RangeReader, TernaryLinearPlan,
    TernaryScratch, stream_linear_identity, stream_linear_ternary,
};

/// One reviewed storage-specific linear plan behind the native execution boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearPlan {
    /// Direct FP16, BF16, or FP32 storage.
    Identity(IdentityLinearPlan),
    /// Grouped trit5 ternary storage with FP32 scales.
    Ternary(TernaryLinearPlan),
}

impl LinearPlan {
    /// Logical output rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        match self {
            Self::Identity(plan) => plan.rows(),
            Self::Ternary(plan) => plan.rows(),
        }
    }

    /// Logical reduction columns.
    #[must_use]
    pub const fn columns(self) -> usize {
        match self {
            Self::Identity(plan) => plan.columns(),
            Self::Ternary(plan) => plan.columns(),
        }
    }

    /// Exact scratch requirement for this storage-specific plan.
    #[must_use]
    pub const fn scratch(self) -> LinearScratchRequirements {
        match self {
            Self::Identity(plan) => {
                let requirement = plan.scratch();
                LinearScratchRequirements {
                    encoded_bytes: requirement.encoded_bytes,
                    decoded_elements: 0,
                    accumulator_elements: 0,
                    local_bytes: size_of::<f64>(),
                    total_bytes: requirement.total_bytes,
                }
            }
            Self::Ternary(plan) => {
                let requirement = plan.scratch();
                LinearScratchRequirements {
                    encoded_bytes: requirement.encoded_bytes,
                    decoded_elements: requirement.decoded_elements,
                    accumulator_elements: requirement.accumulator_elements,
                    local_bytes: 0,
                    total_bytes: requirement.total_bytes,
                }
            }
        }
    }
}

impl From<IdentityLinearPlan> for LinearPlan {
    fn from(plan: IdentityLinearPlan) -> Self {
        Self::Identity(plan)
    }
}

impl From<TernaryLinearPlan> for LinearPlan {
    fn from(plan: TernaryLinearPlan) -> Self {
        Self::Ternary(plan)
    }
}

/// Caller-owned scratch regions and logical high-water for native linear execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearScratchRequirements {
    /// Maximum encoded tile bytes.
    pub encoded_bytes: usize,
    /// Maximum decoded FP32 elements.
    pub decoded_elements: usize,
    /// Maximum FP64 accumulator elements.
    pub accumulator_elements: usize,
    /// Maximum stack-local scalar bytes included in admission.
    pub local_bytes: usize,
    /// Sum of all simultaneously resident scratch and local bytes.
    pub total_bytes: usize,
}

impl LinearScratchRequirements {
    /// Combine sequential plan requirements for one reusable scratch allocation.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` if the combined byte count overflows.
    pub fn union(self, other: Self) -> Result<Self, AmsError> {
        let encoded_bytes = self.encoded_bytes.max(other.encoded_bytes);
        let decoded_elements = self.decoded_elements.max(other.decoded_elements);
        let accumulator_elements = self.accumulator_elements.max(other.accumulator_elements);
        let local_bytes = self.local_bytes.max(other.local_bytes);
        let decoded_bytes = mul(
            decoded_elements,
            size_of::<f32>(),
            "linear union decoded bytes overflow",
        )?;
        let accumulator_bytes = mul(
            accumulator_elements,
            size_of::<f64>(),
            "linear union accumulator bytes overflow",
        )?;
        let total_bytes = add(
            add(
                add(
                    encoded_bytes,
                    decoded_bytes,
                    "linear union encoded and decoded bytes overflow",
                )?,
                accumulator_bytes,
                "linear union accumulator bytes overflow",
            )?,
            local_bytes,
            "linear union local bytes overflow",
        )?;
        Ok(Self {
            encoded_bytes,
            decoded_elements,
            accumulator_elements,
            local_bytes,
            total_bytes,
        })
    }
}

/// Reusable caller-owned regions for identity or ternary linear execution.
pub struct LinearScratch<'a> {
    encoded: &'a mut [u8],
    decoded: &'a mut [f32],
    accumulators: &'a mut [f64],
}

impl<'a> LinearScratch<'a> {
    /// Group caller-owned encoded, decoded, and accumulator regions.
    #[must_use]
    pub const fn new(
        encoded: &'a mut [u8],
        decoded: &'a mut [f32],
        accumulators: &'a mut [f64],
    ) -> Self {
        Self {
            encoded,
            decoded,
            accumulators,
        }
    }

    /// Whether these caller-owned regions satisfy one admitted requirement.
    pub(crate) const fn admits(&self, requirement: LinearScratchRequirements) -> bool {
        self.encoded.len() >= requirement.encoded_bytes
            && self.decoded.len() >= requirement.decoded_elements
            && self.accumulators.len() >= requirement.accumulator_elements
    }
}

/// Execute one identity or ternary matrix-vector multiplication into caller output.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error. Output is not
/// committed by a caller until this operation succeeds.
pub fn stream_linear(
    reader: &dyn RangeReader,
    plan: LinearPlan,
    input: &[f64],
    bias: Option<&[f64]>,
    scratch: &mut LinearScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    if output.len() != plan.rows() {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "linear output dimensions differ from the plan",
        ));
    }
    let requirement = plan.scratch();
    if scratch.encoded.len() < requirement.encoded_bytes
        || scratch.decoded.len() < requirement.decoded_elements
        || scratch.accumulators.len() < requirement.accumulator_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "linear scratch is smaller than the admitted plan",
        ));
    }
    match plan {
        LinearPlan::Identity(identity_plan) => {
            let mut identity_scratch = IdentityScratch::new(&mut *scratch.encoded);
            stream_linear_identity(
                reader,
                identity_plan,
                input,
                bias,
                &mut identity_scratch,
                |row, value| {
                    output[row] = value;
                    Ok(())
                },
            )
        }
        LinearPlan::Ternary(ternary_plan) => {
            let mut ternary_scratch = TernaryScratch::new(
                &mut *scratch.encoded,
                &mut *scratch.decoded,
                &mut *scratch.accumulators,
            );
            stream_linear_ternary(
                reader,
                ternary_plan,
                input,
                bias,
                &mut ternary_scratch,
                |row, value| {
                    output[row] = value;
                    Ok(())
                },
            )
        }
    }
}
