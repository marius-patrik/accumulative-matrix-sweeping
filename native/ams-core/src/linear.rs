use crate::checked::{add, mul};
use crate::{
    AmsError, ErrorCode, IdentityLinearPlan, IdentityScratch, Int4LinearPlan, Int4Scratch,
    RangeReader, TernaryLinearPlan, TernaryScratch, stream_linear_identity, stream_linear_int4,
    stream_linear_ternary,
};

/// One reviewed storage-specific linear plan behind the native execution boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearPlan {
    /// Direct FP16, BF16, or FP32 storage.
    Identity(IdentityLinearPlan),
    /// Grouped trit5 ternary storage with FP32 scales.
    Ternary(TernaryLinearPlan),
    /// Grouped symmetric signed INT4 storage with FP32 scales.
    Int4(Int4LinearPlan),
}

impl LinearPlan {
    /// Logical output rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        match self {
            Self::Identity(plan) => plan.rows(),
            Self::Ternary(plan) => plan.rows(),
            Self::Int4(plan) => plan.rows(),
        }
    }

    /// Logical reduction columns.
    #[must_use]
    pub const fn columns(self) -> usize {
        match self {
            Self::Identity(plan) => plan.columns(),
            Self::Ternary(plan) => plan.columns(),
            Self::Int4(plan) => plan.columns(),
        }
    }

    /// Minimum reader length required by this storage-specific matrix range.
    #[must_use]
    pub const fn reader_end(self) -> u64 {
        match self {
            Self::Identity(plan) => plan.reader_end(),
            Self::Ternary(plan) => plan.reader_end(),
            Self::Int4(plan) => plan.reader_end(),
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
            Self::Int4(plan) => {
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

impl From<Int4LinearPlan> for LinearPlan {
    fn from(plan: Int4LinearPlan) -> Self {
        Self::Int4(plan)
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

/// Reusable caller-owned regions for identity, ternary, or INT4 linear execution.
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

/// Execute one reviewed storage-specific matrix-vector multiplication into caller output.
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
        LinearPlan::Int4(int4_plan) => {
            let mut int4_scratch =
                Int4Scratch::new(&mut *scratch.encoded, &mut *scratch.accumulators);
            stream_linear_int4(
                reader,
                int4_plan,
                input,
                bias,
                &mut int4_scratch,
                |row, value| {
                    output[row] = value;
                    Ok(())
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IdentityDType, Int4Config, SliceReader};

    fn known_int4_record() -> [u8; 8] {
        let mut record = [0u8; 8];
        record[..4].copy_from_slice(&(8.0f32 / 7.0).to_le_bytes());
        record[4..].copy_from_slice(&[0xd9, 0x0f, 0x31, 0x07]);
        record
    }

    #[test]
    fn int4_dispatch_and_union_use_the_shared_scratch_contract() -> Result<(), AmsError> {
        let int4: LinearPlan = Int4LinearPlan::from_arena(1, 7, 0, 16, Int4Config::new(7)?)?.into();
        let identity: LinearPlan =
            IdentityLinearPlan::from_arena(1, 1, 0, 12, IdentityDType::Float32)?.into();
        let combined = int4.scratch().union(identity.scratch())?;
        assert_eq!(combined.encoded_bytes, 8);
        assert_eq!(combined.decoded_elements, 0);
        assert_eq!(combined.accumulator_elements, 1);
        assert_eq!(combined.local_bytes, 8);
        assert_eq!(combined.total_bytes, 24);

        let payload = known_int4_record();
        let reader = SliceReader::new(&payload);
        let mut encoded = [0u8; 8];
        let mut decoded = [0.0f32; 0];
        let mut accumulators = [0.0f64; 1];
        let mut scratch = LinearScratch::new(&mut encoded, &mut decoded, &mut accumulators);
        let mut output = [0.0f64; 1];
        stream_linear(
            &reader,
            int4,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            None,
            &mut scratch,
            &mut output,
        )?;
        let expected = f64::from(8.0f32 / 7.0) * 56.0;
        assert!((output[0] - expected).abs() <= f64::EPSILON);
        Ok(())
    }
}
