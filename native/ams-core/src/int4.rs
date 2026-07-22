use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, RangeReader};

const SCALE_BYTES: usize = 4;
const VALUES_PER_BYTE: usize = 2;

/// Version-one grouped symmetric INT4 codec parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4Config {
    group_size: usize,
}

impl Int4Config {
    /// Create a supported signed-nibble/FP32-scale configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::PlanInvalid`] when the group is zero or greater than 65,536.
    pub const fn new(group_size: usize) -> Result<Self, AmsError> {
        if group_size == 0 || group_size > 65_536 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "INT4 group size is outside 1..=65536",
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
                "INT4 group element count is invalid",
            ));
        }
        add(
            SCALE_BYTES,
            element_count.div_ceil(VALUES_PER_BYTE),
            "INT4 group record size overflow",
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
                "INT4 tensor must contain an element",
            ));
        }
        let groups = element_count.div_ceil(self.group_size);
        let tail = element_count - (groups - 1) * self.group_size;
        add(
            mul(
                groups - 1,
                self.group_record_size(self.group_size)?,
                "INT4 full record bytes overflow",
            )?,
            self.group_record_size(tail)?,
            "INT4 encoded bytes overflow",
        )
    }
}

/// Caller-owned scratch requirements for an INT4 linear plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ScratchRequirements {
    /// Encoded group bytes.
    pub encoded_bytes: usize,
    /// Decoded FP32 group elements (zero because nibbles are consumed directly).
    pub decoded_elements: usize,
    /// FP64 output accumulators.
    pub accumulator_elements: usize,
    /// Logical byte total used for admission.
    pub total_bytes: usize,
}

/// Borrowed, caller-owned scratch passed to one native INT4 execution.
pub struct Int4Scratch<'a> {
    encoded: &'a mut [u8],
    accumulators: &'a mut [f64],
}

impl<'a> Int4Scratch<'a> {
    /// Group encoded and accumulator arenas under one ownership token.
    #[must_use]
    pub const fn new(encoded: &'a mut [u8], accumulators: &'a mut [f64]) -> Self {
        Self {
            encoded,
            accumulators,
        }
    }
}

/// Immutable plan for row-major matrix-vector multiplication from symmetric INT4 storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4LinearPlan {
    rows: usize,
    columns: usize,
    weight_offset: u64,
    output_row_tile: usize,
    encoded_bytes: usize,
    weight_end: u64,
    config: Int4Config,
    scratch: Int4ScratchRequirements,
}

impl Int4LinearPlan {
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
        config: Int4Config,
    ) -> Result<Self, AmsError> {
        if rows == 0 || columns == 0 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "INT4 linear dimensions must be positive",
            ));
        }
        let element_count = mul(rows, columns, "INT4 linear element count overflow")?;
        let encoded_bytes = config.encoded_size(element_count)?;
        let record_bytes = config.group_record_size(config.group_size())?;
        let minimum = add(
            record_bytes,
            size_of::<f64>(),
            "INT4 minimum scratch overflow",
        )?;
        if arena_bytes < minimum {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "arena cannot hold one INT4 group and accumulator",
            ));
        }
        let output_row_tile = rows.min((arena_bytes - record_bytes) / size_of::<f64>());
        let accumulator_bytes = mul(
            output_row_tile,
            size_of::<f64>(),
            "INT4 accumulator scratch overflow",
        )?;
        let total_bytes = add(
            record_bytes,
            accumulator_bytes,
            "INT4 total scratch overflow",
        )?;
        let encoded_u64 = usize_to_u64(encoded_bytes, "INT4 encoded size exceeds u64")?;
        let weight_end = add_u64(
            weight_offset,
            encoded_u64,
            "INT4 weight range overflows u64",
        )?;
        Ok(Self {
            rows,
            columns,
            weight_offset,
            output_row_tile,
            encoded_bytes,
            weight_end,
            config,
            scratch: Int4ScratchRequirements {
                encoded_bytes: record_bytes,
                decoded_elements: 0,
                accumulator_elements: output_row_tile,
                total_bytes,
            },
        })
    }

    /// Caller-owned scratch required by this plan.
    #[must_use]
    pub const fn scratch(self) -> Int4ScratchRequirements {
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

    /// Minimum reader length required by this encoded matrix range.
    #[must_use]
    pub const fn reader_end(self) -> u64 {
        self.weight_end
    }

    /// Logical output rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Logical reduction columns.
    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }
}

fn visit_int4_group<F>(record: &[u8], element_count: usize, mut visit: F) -> Result<(), AmsError>
where
    F: FnMut(usize, f64),
{
    let expected = add(
        SCALE_BYTES,
        element_count.div_ceil(VALUES_PER_BYTE),
        "INT4 decode record size overflow",
    )?;
    if element_count == 0 || record.len() != expected {
        return Err(AmsError::new(
            ErrorCode::InvalidPackage,
            "INT4 group record length is invalid",
        ));
    }
    let scale_bytes: [u8; SCALE_BYTES] = record[..SCALE_BYTES]
        .try_into()
        .map_err(|_| AmsError::new(ErrorCode::InternalInvariant, "scale slice length changed"))?;
    let scale = f32::from_le_bytes(scale_bytes);
    if !scale.is_finite() || scale < 0.0 {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "INT4 scale is invalid",
        ));
    }
    let mut produced = 0usize;
    for &packed in &record[SCALE_BYTES..] {
        for nibble in [packed & 0x0f, packed >> 4] {
            if produced >= element_count {
                if nibble != 0 {
                    return Err(AmsError::new(
                        ErrorCode::InvalidPackage,
                        "INT4 tail padding is not canonical zero",
                    ));
                }
                continue;
            }
            let signed = if nibble >= 8 {
                i16::from(nibble) - 16
            } else {
                i16::from(nibble)
            };
            if signed == -8 {
                return Err(AmsError::new(
                    ErrorCode::InvalidPackage,
                    "INT4 reserved value -8 is invalid",
                ));
            }
            visit(produced, f64::from(scale) * f64::from(signed));
            produced += 1;
        }
    }
    if produced != element_count {
        return Err(AmsError::new(
            ErrorCode::InternalInvariant,
            "INT4 decoder produced the wrong element count",
        ));
    }
    Ok(())
}

/// Decode one complete symmetric INT4 group into caller-owned FP64 output.
///
/// Values use two's-complement nibbles in low-nibble-first order. The `-8` code is
/// reserved and an unused high tail nibble must be canonical zero.
///
/// # Errors
///
/// Returns a typed plan, package, numeric, or invariant error when the output is too
/// small or the encoded group is malformed.
pub fn decode_int4_group(
    record: &[u8],
    element_count: usize,
    output: &mut [f64],
) -> Result<(), AmsError> {
    if element_count == 0 || output.len() < element_count {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "INT4 decode output is too small",
        ));
    }
    visit_int4_group(record, element_count, |index, value| output[index] = value)
}

/// Execute matrix-vector multiplication without allocating or reconstructing the matrix.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, callback, or numeric error. The
/// function emits no later row after an error and never retains caller-owned scratch.
#[allow(clippy::suboptimal_flops)] // Preserve explicit multiply-then-add reference order.
pub fn stream_linear_int4<R, F>(
    reader: &R,
    plan: Int4LinearPlan,
    input: &[f64],
    bias: Option<&[f64]>,
    scratch: &mut Int4Scratch<'_>,
    mut emit: F,
) -> Result<(), AmsError>
where
    R: RangeReader + ?Sized,
    F: FnMut(usize, f64) -> Result<(), AmsError>,
{
    if input.len() != plan.columns || bias.is_some_and(|values| values.len() != plan.rows) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "INT4 linear input or bias shape is invalid",
        ));
    }
    if input.iter().any(|value| !value.is_finite())
        || bias.is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "INT4 linear input or bias is non-finite",
        ));
    }
    let required = plan.scratch;
    if scratch.encoded.len() < required.encoded_bytes
        || scratch.accumulators.len() < required.accumulator_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "caller scratch is smaller than the admitted INT4 plan",
        ));
    }
    if plan.reader_end() > reader.len() {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "INT4 weights exceed the storage object",
        ));
    }
    let element_count = mul(plan.rows, plan.columns, "INT4 element count overflow")?;
    let full_record_bytes = plan.config.group_record_size(plan.config.group_size())?;
    let mut row_start = 0usize;
    while row_start < plan.rows {
        let row_count = plan.output_row_tile.min(plan.rows - row_start);
        for (index, accumulator) in scratch.accumulators[..row_count].iter_mut().enumerate() {
            *accumulator = bias.map_or(0.0, |values| values[row_start + index]);
        }
        let flat_start = mul(row_start, plan.columns, "INT4 flat row start overflow")?;
        let flat_end = mul(
            row_start + row_count,
            plan.columns,
            "INT4 flat row end overflow",
        )?;
        let first_group = flat_start / plan.config.group_size();
        let final_group = (flat_end - 1) / plan.config.group_size();
        for group_index in first_group..=final_group {
            let group_flat_start = mul(
                group_index,
                plan.config.group_size(),
                "INT4 group flat start overflow",
            )?;
            let count = plan
                .config
                .group_size()
                .min(element_count - group_flat_start);
            let record_size = plan.config.group_record_size(count)?;
            let relative = mul(
                group_index,
                full_record_bytes,
                "INT4 group record offset overflow",
            )?;
            let record_offset = add_u64(
                plan.weight_offset,
                usize_to_u64(relative, "INT4 group offset exceeds u64")?,
                "INT4 group absolute offset overflow",
            )?;
            reader.read_exact_at(record_offset, &mut scratch.encoded[..record_size])?;
            let local_start = flat_start.max(group_flat_start) - group_flat_start;
            let local_end = flat_end.min(group_flat_start + count) - group_flat_start;
            let encoded = &scratch.encoded[..record_size];
            let accumulators = &mut scratch.accumulators;
            visit_int4_group(encoded, count, |local_index, weight| {
                if local_index >= local_start && local_index < local_end {
                    let flat_index = group_flat_start + local_index;
                    let output_index = flat_index / plan.columns - row_start;
                    let input_index = flat_index % plan.columns;
                    accumulators[output_index] += weight * input[input_index];
                }
            })?;
        }
        for (index, &value) in scratch.accumulators[..row_count].iter().enumerate() {
            if !value.is_finite() {
                return Err(AmsError::new(
                    ErrorCode::NumericFailure,
                    "INT4 linear output is non-finite",
                ));
            }
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

    fn known_record() -> [u8; 8] {
        let mut record = [0u8; 8];
        record[..4].copy_from_slice(&(8.0f32 / 7.0).to_le_bytes());
        record[4..].copy_from_slice(&[0xd9, 0x0f, 0x31, 0x07]);
        record
    }

    #[test]
    fn known_group_matches_the_python_codec_fixture() -> Result<(), AmsError> {
        let mut output = [0.0f64; 7];
        decode_int4_group(&known_record(), 7, &mut output)?;
        let scale = f64::from(8.0f32 / 7.0);
        let expected = [
            -7.0 * scale,
            -3.0 * scale,
            -scale,
            0.0,
            scale,
            3.0 * scale,
            7.0 * scale,
        ];
        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(left, right)| (*left - right).abs() <= f64::EPSILON)
        );
        Ok(())
    }

    #[test]
    fn reserved_code_and_tail_padding_are_rejected() {
        let mut output = [0.0f64; 1];
        let mut reserved = [0u8; 5];
        reserved[..4].copy_from_slice(&1.0f32.to_le_bytes());
        reserved[4] = 0x08;
        let reserved_error = decode_int4_group(&reserved, 1, &mut output).err();
        assert_eq!(
            reserved_error.map(AmsError::code),
            Some(ErrorCode::InvalidPackage)
        );
        let mut padded = reserved;
        padded[4] = 0x10;
        let padding_error = decode_int4_group(&padded, 1, &mut output).err();
        assert_eq!(
            padding_error.map(AmsError::code),
            Some(ErrorCode::InvalidPackage)
        );
    }

    #[test]
    fn negative_zero_scale_matches_the_python_reference() -> Result<(), AmsError> {
        let mut record = [0u8; 5];
        record[..4].copy_from_slice(&(-0.0f32).to_le_bytes());
        record[4] = 0x07;
        let mut output = [1.0f64; 1];
        decode_int4_group(&record, 1, &mut output)?;
        assert!(output[0].abs() <= f64::EPSILON);
        assert!(output[0].is_sign_negative());
        Ok(())
    }

    #[test]
    fn direct_linear_uses_caller_owned_scratch() -> Result<(), AmsError> {
        let config = Int4Config::new(7)?;
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&known_record());
        encoded.extend_from_slice(&known_record());
        let reader = SliceReader::new(&encoded);
        let plan = Int4LinearPlan::from_arena(2, 7, 0, 16, config)?;
        let scratch = plan.scratch();
        let mut encoded_scratch = [0u8; 8];
        let mut accumulators = [0.0f64; 1];
        assert_eq!(scratch.encoded_bytes, encoded_scratch.len());
        assert_eq!(scratch.decoded_elements, 0);
        assert_eq!(scratch.accumulator_elements, accumulators.len());
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = [0.0f64; 2];
        let mut scratch_buffers = Int4Scratch::new(&mut encoded_scratch, &mut accumulators);
        stream_linear_int4(
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
        let expected = f64::from(8.0f32 / 7.0) * 56.0;
        assert!(
            output
                .iter()
                .all(|value| (*value - expected).abs() <= f64::EPSILON)
        );
        Ok(())
    }

    #[test]
    fn python_multigroup_fixture_matches_direct_linear() -> Result<(), AmsError> {
        // Produced by the reviewed Python v1 encoder for two mirrored five-value rows.
        let payload = [
            0x25, 0x49, 0x92, 0x3e, 0xd9, 0x30, 0x07, 0x25, 0x49, 0x92, 0x3e, 0x37, 0xd0, 0x09,
        ];
        let reader = SliceReader::new(&payload);
        let plan = Int4LinearPlan::from_arena(2, 5, 0, 15, Int4Config::new(5)?)?;
        let mut encoded = [0u8; 7];
        let mut accumulators = [0.0f64; 1];
        let mut scratch = Int4Scratch::new(&mut encoded, &mut accumulators);
        let mut output = [0.0f64; 2];
        stream_linear_int4(
            &reader,
            plan,
            &[0.5, -1.0, 2.0, 1.5, -0.25],
            None,
            &mut scratch,
            |row, value| {
                output[row] = value;
                Ok(())
            },
        )?;
        let expected = [0.642_857_171_595_096_6, -0.642_857_171_595_096_6];
        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() <= f64::EPSILON)
        );
        Ok(())
    }

    #[test]
    fn arena_below_minimum_is_rejected() -> Result<(), AmsError> {
        let error = Int4LinearPlan::from_arena(1, 1, 0, 15, Int4Config::new(7)?).err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        Ok(())
    }

    #[test]
    fn int4_linear_rejects_nonfinite_input() -> Result<(), AmsError> {
        let payload = known_record();
        let reader = SliceReader::new(&payload);
        let plan = Int4LinearPlan::from_arena(1, 7, 0, 16, Int4Config::new(7)?)?;
        let mut encoded_scratch = [0u8; 8];
        let mut accumulators = [0.0f64; 1];
        let mut scratch = Int4Scratch::new(&mut encoded_scratch, &mut accumulators);
        let error = stream_linear_int4(
            &reader,
            plan,
            &[1.0, 2.0, f64::INFINITY, 4.0, 5.0, 6.0, 7.0],
            None,
            &mut scratch,
            |_, _| Ok(()),
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        Ok(())
    }
}
