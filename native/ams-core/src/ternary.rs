use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, RangeReader};

const SCALE_BYTES: usize = 4;
const TRITS_PER_BYTE: usize = 5;
const TRIT_POWERS: [u8; TRITS_PER_BYTE] = [1, 3, 9, 27, 81];

/// Version-one grouped ternary codec parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TernaryConfig {
    group_size: usize,
}

impl TernaryConfig {
    /// Create a supported trit5/FP32-scale configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::PlanInvalid`] when the group is zero or greater than 65,536.
    pub const fn new(group_size: usize) -> Result<Self, AmsError> {
        if group_size == 0 || group_size > 65_536 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "ternary group size is outside 1..=65536",
            ));
        }
        Ok(Self { group_size })
    }

    /// Number of logical weights in a complete group.
    #[must_use]
    pub const fn group_size(self) -> usize {
        self.group_size
    }

    /// Encoded bytes for a group containing `element_count` weights.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::PlanInvalid`] for an invalid group count or checked overflow.
    pub fn group_record_size(self, element_count: usize) -> Result<usize, AmsError> {
        if element_count == 0 || element_count > self.group_size {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "ternary group element count is invalid",
            ));
        }
        add(
            SCALE_BYTES,
            element_count.div_ceil(TRITS_PER_BYTE),
            "ternary group record size overflow",
        )
    }

    /// Encoded bytes for a complete flattened tensor.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::PlanInvalid`] for an empty tensor or checked overflow.
    pub fn encoded_size(self, element_count: usize) -> Result<usize, AmsError> {
        if element_count == 0 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "ternary tensor must contain an element",
            ));
        }
        let groups = element_count.div_ceil(self.group_size);
        let tail = element_count - (groups - 1) * self.group_size;
        add(
            mul(
                groups - 1,
                self.group_record_size(self.group_size)?,
                "ternary full record bytes overflow",
            )?,
            self.group_record_size(tail)?,
            "ternary encoded bytes overflow",
        )
    }
}

/// Caller-owned scratch requirements for a ternary linear plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TernaryScratchRequirements {
    /// Encoded group bytes.
    pub encoded_bytes: usize,
    /// Decoded FP32 group elements.
    pub decoded_elements: usize,
    /// FP64 output accumulators.
    pub accumulator_elements: usize,
    /// Logical byte total used for admission.
    pub total_bytes: usize,
}

/// Borrowed, caller-owned scratch passed to one native ternary execution.
pub struct TernaryScratch<'a> {
    encoded: &'a mut [u8],
    decoded: &'a mut [f32],
    accumulators: &'a mut [f64],
}

impl<'a> TernaryScratch<'a> {
    /// Group encoded, decoded, and accumulator arenas under one ownership token.
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
}

/// Immutable plan for row-major matrix-vector multiplication from ternary storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TernaryLinearPlan {
    rows: usize,
    columns: usize,
    weight_offset: u64,
    output_row_tile: usize,
    encoded_bytes: usize,
    config: TernaryConfig,
    scratch: TernaryScratchRequirements,
}

impl TernaryLinearPlan {
    /// Select the largest legal output-row tile for a logical arena.
    ///
    /// # Errors
    ///
    /// Returns a typed planning or capacity error for invalid dimensions, checked
    /// overflow, or an arena below the minimum primitive working set.
    pub fn from_arena(
        rows: usize,
        columns: usize,
        weight_offset: u64,
        arena_bytes: usize,
        config: TernaryConfig,
    ) -> Result<Self, AmsError> {
        if rows == 0 || columns == 0 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "ternary linear dimensions must be positive",
            ));
        }
        let element_count = mul(rows, columns, "ternary linear element count overflow")?;
        let encoded_bytes = config.encoded_size(element_count)?;
        let record_bytes = config.group_record_size(config.group_size())?;
        let decoded_bytes = mul(
            config.group_size(),
            size_of::<f32>(),
            "ternary decoded scratch overflow",
        )?;
        let fixed_bytes = add(
            record_bytes,
            decoded_bytes,
            "ternary fixed scratch overflow",
        )?;
        let minimum = add(
            fixed_bytes,
            size_of::<f64>(),
            "ternary minimum scratch overflow",
        )?;
        if arena_bytes < minimum {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "arena cannot hold one ternary group and accumulator",
            ));
        }
        let output_row_tile = rows.min((arena_bytes - fixed_bytes) / size_of::<f64>());
        let accumulator_bytes = mul(
            output_row_tile,
            size_of::<f64>(),
            "ternary accumulator scratch overflow",
        )?;
        let total_bytes = add(
            fixed_bytes,
            accumulator_bytes,
            "ternary total scratch overflow",
        )?;
        let encoded_u64 = usize_to_u64(encoded_bytes, "ternary encoded size exceeds u64")?;
        add_u64(
            weight_offset,
            encoded_u64,
            "ternary weight range overflows u64",
        )?;
        Ok(Self {
            rows,
            columns,
            weight_offset,
            output_row_tile,
            encoded_bytes,
            config,
            scratch: TernaryScratchRequirements {
                encoded_bytes: record_bytes,
                decoded_elements: config.group_size(),
                accumulator_elements: output_row_tile,
                total_bytes,
            },
        })
    }

    /// Caller-owned scratch required by this plan.
    #[must_use]
    pub const fn scratch(self) -> TernaryScratchRequirements {
        self.scratch
    }

    /// Selected output rows per pass.
    #[must_use]
    pub const fn output_row_tile(self) -> usize {
        self.output_row_tile
    }

    /// Complete encoded tensor byte length.
    #[must_use]
    pub const fn encoded_bytes(self) -> usize {
        self.encoded_bytes
    }
}

/// Decode one complete group into caller-owned FP32 scratch.
///
/// # Errors
///
/// Returns a typed plan, package, numeric, or invariant error when the output is too
/// small or the encoded group is malformed.
pub fn decode_ternary_group(
    record: &[u8],
    element_count: usize,
    output: &mut [f32],
) -> Result<(), AmsError> {
    if element_count == 0 || output.len() < element_count {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "ternary decode output is too small",
        ));
    }
    let expected = add(
        SCALE_BYTES,
        element_count.div_ceil(TRITS_PER_BYTE),
        "ternary decode record size overflow",
    )?;
    if record.len() != expected {
        return Err(AmsError::new(
            ErrorCode::InvalidPackage,
            "ternary group record length is invalid",
        ));
    }
    let scale_bytes: [u8; SCALE_BYTES] = record[..SCALE_BYTES]
        .try_into()
        .map_err(|_| AmsError::new(ErrorCode::InternalInvariant, "scale slice length changed"))?;
    let scale = f32::from_le_bytes(scale_bytes);
    if !scale.is_finite() || scale.is_sign_negative() {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "ternary scale is invalid",
        ));
    }
    let mut produced = 0usize;
    for &packed in &record[SCALE_BYTES..] {
        if packed > 242 {
            return Err(AmsError::new(
                ErrorCode::InvalidPackage,
                "ternary packed byte exceeds 242",
            ));
        }
        for power in TRIT_POWERS {
            let digit = (packed / power) % 3;
            if produced >= element_count {
                if digit != 1 {
                    return Err(AmsError::new(
                        ErrorCode::InvalidPackage,
                        "ternary tail padding is not canonical zero",
                    ));
                }
                continue;
            }
            output[produced] = match digit {
                0 => -scale,
                1 => 0.0,
                2 => scale,
                _ => {
                    return Err(AmsError::new(
                        ErrorCode::InternalInvariant,
                        "base-3 digit exceeded two",
                    ));
                }
            };
            produced += 1;
        }
    }
    if produced != element_count {
        return Err(AmsError::new(
            ErrorCode::InternalInvariant,
            "ternary decoder produced the wrong element count",
        ));
    }
    Ok(())
}

/// Execute matrix-vector multiplication without allocating or reconstructing the matrix.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, callback, or numeric error. The
/// function emits no later row after an error and never retains caller-owned scratch.
#[allow(clippy::suboptimal_flops)] // Preserve explicit multiply-then-add reference order.
pub fn stream_linear_ternary<R, F>(
    reader: &R,
    plan: TernaryLinearPlan,
    input: &[f64],
    bias: Option<&[f64]>,
    scratch: &mut TernaryScratch<'_>,
    mut emit: F,
) -> Result<(), AmsError>
where
    R: RangeReader,
    F: FnMut(usize, f64) -> Result<(), AmsError>,
{
    if input.len() != plan.columns || bias.is_some_and(|values| values.len() != plan.rows) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "ternary linear input or bias shape is invalid",
        ));
    }
    let required = plan.scratch;
    if scratch.encoded.len() < required.encoded_bytes
        || scratch.decoded.len() < required.decoded_elements
        || scratch.accumulators.len() < required.accumulator_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "caller scratch is smaller than the admitted plan",
        ));
    }
    let encoded_u64 = usize_to_u64(plan.encoded_bytes, "encoded size exceeds u64")?;
    if add_u64(
        plan.weight_offset,
        encoded_u64,
        "ternary weight end overflows u64",
    )? > reader.len()
    {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "ternary weights exceed the storage object",
        ));
    }
    let element_count = mul(plan.rows, plan.columns, "ternary element count overflow")?;
    let full_record_bytes = plan.config.group_record_size(plan.config.group_size())?;
    let mut row_start = 0usize;
    while row_start < plan.rows {
        let row_count = plan.output_row_tile.min(plan.rows - row_start);
        for (index, accumulator) in scratch.accumulators[..row_count].iter_mut().enumerate() {
            *accumulator = bias.map_or(0.0, |values| values[row_start + index]);
        }
        let flat_start = mul(row_start, plan.columns, "flat row start overflow")?;
        let flat_end = mul(row_start + row_count, plan.columns, "flat row end overflow")?;
        let first_group = flat_start / plan.config.group_size();
        let final_group = (flat_end - 1) / plan.config.group_size();
        for group_index in first_group..=final_group {
            let group_flat_start = mul(
                group_index,
                plan.config.group_size(),
                "group flat start overflow",
            )?;
            let count = plan
                .config
                .group_size()
                .min(element_count - group_flat_start);
            let record_size = plan.config.group_record_size(count)?;
            let relative = mul(
                group_index,
                full_record_bytes,
                "group record offset overflow",
            )?;
            let record_offset = add_u64(
                plan.weight_offset,
                usize_to_u64(relative, "group offset exceeds u64")?,
                "group absolute offset overflow",
            )?;
            reader.read_exact_at(record_offset, &mut scratch.encoded[..record_size])?;
            decode_ternary_group(&scratch.encoded[..record_size], count, scratch.decoded)?;
            let local_start = flat_start.max(group_flat_start) - group_flat_start;
            let local_end = flat_end.min(group_flat_start + count) - group_flat_start;
            for (local_index, &decoded) in scratch
                .decoded
                .iter()
                .enumerate()
                .take(local_end)
                .skip(local_start)
            {
                let flat_index = group_flat_start + local_index;
                let output_index = flat_index / plan.columns - row_start;
                let input_index = flat_index % plan.columns;
                scratch.accumulators[output_index] += f64::from(decoded) * input[input_index];
            }
        }
        for (index, &value) in scratch.accumulators[..row_count].iter().enumerate() {
            emit(row_start + index, value)?;
        }
        row_start += row_count;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SliceReader;

    fn known_record() -> [u8; 5] {
        let mut record = [0u8; 5];
        record[..4].copy_from_slice(&1.5f32.to_le_bytes());
        record[4] = 225;
        record
    }

    #[test]
    fn known_group_decodes_with_canonical_values() -> Result<(), AmsError> {
        let mut output = [0.0f32; 5];
        decode_ternary_group(&known_record(), 5, &mut output)?;
        let expected = [-1.5, -1.5, 0.0, 1.5, 1.5];
        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(left, right)| (*left - right).abs() <= f32::EPSILON)
        );
        Ok(())
    }

    #[test]
    fn tail_padding_is_validated() {
        let mut record = known_record();
        record[4] = 2 + 2 * 3 + 9 + 27 + 81;
        let mut output = [0.0f32; 1];
        let error = decode_ternary_group(&record, 1, &mut output).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::InvalidPackage));
    }

    #[test]
    fn direct_linear_uses_caller_owned_scratch() -> Result<(), AmsError> {
        let config = TernaryConfig::new(5)?;
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&known_record());
        encoded.extend_from_slice(&known_record());
        let reader = SliceReader::new(&encoded);
        let plan = TernaryLinearPlan::from_arena(2, 5, 0, 33, config)?;
        let scratch = plan.scratch();
        let mut encoded_scratch = [0u8; 5];
        let mut decoded_scratch = [0.0f32; 5];
        let mut accumulators = [0.0f64; 1];
        assert_eq!(scratch.encoded_bytes, encoded_scratch.len());
        assert_eq!(scratch.decoded_elements, decoded_scratch.len());
        assert_eq!(scratch.accumulator_elements, accumulators.len());
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f64; 2];
        let mut scratch_buffers = TernaryScratch::new(
            &mut encoded_scratch,
            &mut decoded_scratch,
            &mut accumulators,
        );
        stream_linear_ternary(
            &reader,
            plan,
            &input,
            None,
            &mut scratch_buffers,
            |row, value| {
                output[row] = value;
                Ok(())
            },
        )?;
        assert!(
            output
                .iter()
                .all(|value| (*value - 9.0).abs() <= f64::EPSILON)
        );
        Ok(())
    }

    #[test]
    fn arena_below_minimum_is_rejected() -> Result<(), AmsError> {
        let config = TernaryConfig::new(5)?;
        let error = TernaryLinearPlan::from_arena(1, 1, 0, 32, config).err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        Ok(())
    }
}
