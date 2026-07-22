use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, RangeReader};

const ACCUMULATOR_BYTES: usize = size_of::<f64>();

/// Supported uncompressed floating-point weight representations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityDType {
    /// IEEE 754 binary16.
    Float16,
    /// Brain floating point with eight exponent bits and seven fraction bits.
    BFloat16,
    /// IEEE 754 binary32.
    Float32,
}

impl IdentityDType {
    const fn item_bytes(self) -> usize {
        match self {
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Caller-owned encoded tile requirements for identity linear execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdentityScratchRequirements {
    /// Encoded bytes required for one reduction tile.
    pub encoded_bytes: usize,
    /// Logical total including the one scalar accumulator.
    pub total_bytes: usize,
}

/// Borrowed caller-owned tile passed to identity linear execution.
pub struct IdentityScratch<'a> {
    encoded: &'a mut [u8],
}

impl<'a> IdentityScratch<'a> {
    /// Wrap one caller-owned encoded byte arena.
    #[must_use]
    pub const fn new(encoded: &'a mut [u8]) -> Self {
        Self { encoded }
    }
}

/// Immutable row-major identity matrix-vector plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IdentityLinearPlan {
    rows: usize,
    columns: usize,
    weight_offset: u64,
    dtype: IdentityDType,
    reduction_tile: usize,
    encoded_bytes: usize,
    scratch: IdentityScratchRequirements,
}

impl IdentityLinearPlan {
    /// Select the largest legal reduction tile for a logical arena.
    ///
    /// # Errors
    ///
    /// Returns a typed planning or capacity error for invalid dimensions, checked
    /// overflow, or an arena below one encoded value plus one accumulator.
    pub fn from_arena(
        rows: usize,
        columns: usize,
        weight_offset: u64,
        arena_bytes: usize,
        dtype: IdentityDType,
    ) -> Result<Self, AmsError> {
        if rows == 0 || columns == 0 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "identity linear dimensions must be positive",
            ));
        }
        let item_bytes = dtype.item_bytes();
        let minimum = add(
            item_bytes,
            ACCUMULATOR_BYTES,
            "identity minimum scratch overflow",
        )?;
        if arena_bytes < minimum {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "arena cannot hold one identity weight and accumulator",
            ));
        }
        let reduction_tile = columns.min((arena_bytes - ACCUMULATOR_BYTES) / item_bytes);
        let encoded_bytes = mul(
            mul(rows, columns, "identity element count overflow")?,
            item_bytes,
            "identity encoded bytes overflow",
        )?;
        let scratch_encoded = mul(
            reduction_tile,
            item_bytes,
            "identity encoded scratch overflow",
        )?;
        let total_bytes = add(
            scratch_encoded,
            ACCUMULATOR_BYTES,
            "identity total scratch overflow",
        )?;
        add_u64(
            weight_offset,
            usize_to_u64(encoded_bytes, "identity encoded size exceeds u64")?,
            "identity weight range overflows u64",
        )?;
        Ok(Self {
            rows,
            columns,
            weight_offset,
            dtype,
            reduction_tile,
            encoded_bytes,
            scratch: IdentityScratchRequirements {
                encoded_bytes: scratch_encoded,
                total_bytes,
            },
        })
    }

    /// Caller-owned scratch required by this plan.
    #[must_use]
    pub const fn scratch(self) -> IdentityScratchRequirements {
        self.scratch
    }

    /// Selected reduction elements per range read.
    #[must_use]
    pub const fn reduction_tile(self) -> usize {
        self.reduction_tile
    }
}

fn f16_to_f32(word: u16) -> f32 {
    let sign = (u32::from(word & 0x8000)) << 16;
    let exponent = (word >> 10) & 0x1f;
    let fraction = word & 0x03ff;
    let bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut normalized = fraction;
            let mut unbiased_exponent = -14i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased_exponent -= 1;
            }
            normalized &= 0x03ff;
            let exponent_bits = u32::try_from(unbiased_exponent + 127).unwrap_or(0);
            sign | (exponent_bits << 23) | (u32::from(normalized) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (u32::from(fraction) << 13)
    } else {
        let exponent_bits = u32::from(exponent) + 112;
        sign | (exponent_bits << 23) | (u32::from(fraction) << 13)
    };
    f32::from_bits(bits)
}

fn decode_identity(encoded: &[u8], index: usize, dtype: IdentityDType) -> Result<f32, AmsError> {
    let offset = mul(index, dtype.item_bytes(), "identity decode offset overflow")?;
    let value = match dtype {
        IdentityDType::Float16 => {
            let bytes: [u8; 2] = encoded
                .get(offset..offset + 2)
                .ok_or_else(|| {
                    AmsError::new(
                        ErrorCode::InternalInvariant,
                        "identity F16 tile is too small",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    AmsError::new(ErrorCode::InternalInvariant, "identity F16 slice changed")
                })?;
            f16_to_f32(u16::from_le_bytes(bytes))
        }
        IdentityDType::BFloat16 => {
            let bytes: [u8; 2] = encoded
                .get(offset..offset + 2)
                .ok_or_else(|| {
                    AmsError::new(
                        ErrorCode::InternalInvariant,
                        "identity BF16 tile is too small",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    AmsError::new(ErrorCode::InternalInvariant, "identity BF16 slice changed")
                })?;
            f32::from_bits(u32::from(u16::from_le_bytes(bytes)) << 16)
        }
        IdentityDType::Float32 => {
            let bytes: [u8; 4] = encoded
                .get(offset..offset + 4)
                .ok_or_else(|| {
                    AmsError::new(
                        ErrorCode::InternalInvariant,
                        "identity F32 tile is too small",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    AmsError::new(ErrorCode::InternalInvariant, "identity F32 slice changed")
                })?;
            f32::from_le_bytes(bytes)
        }
    };
    if !value.is_finite() {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "identity weight is non-finite",
        ));
    }
    Ok(value)
}

/// Execute matrix-vector multiplication from FP16, BF16, or FP32 storage without allocation.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, numeric, callback, or overflow error.
#[allow(clippy::suboptimal_flops)] // Preserve explicit multiply-then-add reference order.
pub fn stream_linear_identity<R, F>(
    reader: &R,
    plan: IdentityLinearPlan,
    input: &[f64],
    bias: Option<&[f64]>,
    scratch: &mut IdentityScratch<'_>,
    mut emit: F,
) -> Result<(), AmsError>
where
    R: RangeReader,
    F: FnMut(usize, f64) -> Result<(), AmsError>,
{
    if input.len() != plan.columns || bias.is_some_and(|values| values.len() != plan.rows) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "identity linear input or bias shape is invalid",
        ));
    }
    if scratch.encoded.len() < plan.scratch.encoded_bytes {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "caller identity scratch is smaller than the admitted plan",
        ));
    }
    if add_u64(
        plan.weight_offset,
        usize_to_u64(plan.encoded_bytes, "identity encoded size exceeds u64")?,
        "identity weight end overflows u64",
    )? > reader.len()
    {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "identity weights exceed the storage object",
        ));
    }
    let item_bytes = plan.dtype.item_bytes();
    for row in 0..plan.rows {
        let mut accumulator = bias.map_or(0.0, |values| values[row]);
        for start in (0..plan.columns).step_by(plan.reduction_tile) {
            let count = plan.reduction_tile.min(plan.columns - start);
            let flat_index = add(
                mul(row, plan.columns, "identity row offset overflow")?,
                start,
                "identity flat offset overflow",
            )?;
            let relative = mul(flat_index, item_bytes, "identity byte offset overflow")?;
            let byte_count = mul(count, item_bytes, "identity tile bytes overflow")?;
            let absolute = add_u64(
                plan.weight_offset,
                usize_to_u64(relative, "identity tile offset exceeds u64")?,
                "identity absolute offset overflow",
            )?;
            reader.read_exact_at(absolute, &mut scratch.encoded[..byte_count])?;
            for index in 0..count {
                let weight = decode_identity(&scratch.encoded[..byte_count], index, plan.dtype)?;
                accumulator += f64::from(weight) * input[start + index];
            }
        }
        emit(row, accumulator)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SliceReader;

    fn encoded(dtype: IdentityDType) -> Vec<u8> {
        match dtype {
            IdentityDType::Float16 => [0x3c00u16, 0xc000, 0x3800]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            IdentityDType::BFloat16 => [0x3f80u16, 0xc000, 0x3f00]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            IdentityDType::Float32 => [1.0f32, -2.0, 0.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        }
    }

    #[test]
    fn all_identity_dtypes_execute_with_one_caller_owned_tile() -> Result<(), AmsError> {
        for dtype in [
            IdentityDType::Float16,
            IdentityDType::BFloat16,
            IdentityDType::Float32,
        ] {
            let payload = encoded(dtype);
            let reader = SliceReader::new(&payload);
            let plan = IdentityLinearPlan::from_arena(1, 3, 0, 16, dtype)?;
            let requirements = plan.scratch();
            let mut encoded_scratch = [0u8; 8];
            assert!(requirements.encoded_bytes <= encoded_scratch.len());
            let mut scratch = IdentityScratch::new(&mut encoded_scratch);
            let mut output = 0.0;
            stream_linear_identity(
                &reader,
                plan,
                &[2.0, 3.0, 4.0],
                None,
                &mut scratch,
                |_, value| {
                    output = value;
                    Ok(())
                },
            )?;
            assert!((output + 2.0).abs() <= f64::EPSILON);
        }
        Ok(())
    }

    #[test]
    fn identity_linear_rejects_nonfinite_weights() -> Result<(), AmsError> {
        let payload = 0x7e00u16.to_le_bytes();
        let reader = SliceReader::new(&payload);
        let plan = IdentityLinearPlan::from_arena(1, 1, 0, 10, IdentityDType::Float16)?;
        let mut encoded_scratch = [0u8; 2];
        let mut scratch = IdentityScratch::new(&mut encoded_scratch);
        let error =
            stream_linear_identity(&reader, plan, &[1.0], None, &mut scratch, |_, _| Ok(())).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        Ok(())
    }
}
