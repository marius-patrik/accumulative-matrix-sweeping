use crate::checked::{add, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, IdentityDType, RangeReader};

/// Exact caller-owned storage and staging requirements for one K/V cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheRequirements {
    /// Complete fixed-capacity key arena bytes.
    pub key_storage_bytes: usize,
    /// Complete fixed-capacity value arena bytes.
    pub value_storage_bytes: usize,
    /// One encoded key row plus one encoded value row.
    pub staging_bytes: usize,
    /// Total durable/in-memory cache capacity bytes, excluding staging.
    pub storage_bytes: usize,
}

/// Immutable plan for a fixed-capacity, append-only K/V cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCachePlan {
    head_count: usize,
    key_head_dimension: usize,
    value_head_dimension: usize,
    capacity_tokens: usize,
    key_dtype: IdentityDType,
    value_dtype: IdentityDType,
    key_row_elements: usize,
    value_row_elements: usize,
    key_row_bytes: usize,
    value_row_bytes: usize,
    requirements: KvCacheRequirements,
}

impl KvCachePlan {
    /// Validate cache dimensions and derive exact fixed-capacity byte requirements.
    ///
    /// The first bring-up boundary admits BF16 and FP32 cache storage. INT8/INT4 require
    /// separate versioned codecs; FP16 is rejected until its encoder is independently qualified.
    ///
    /// # Errors
    ///
    /// Returns a typed capability or plan error for an unsupported dtype, zero dimension, or overflow.
    pub fn new(
        head_count: usize,
        key_head_dimension: usize,
        value_head_dimension: usize,
        capacity_tokens: usize,
        key_dtype: IdentityDType,
        value_dtype: IdentityDType,
    ) -> Result<Self, AmsError> {
        if head_count == 0
            || key_head_dimension == 0
            || value_head_dimension == 0
            || capacity_tokens == 0
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "K/V cache dimensions must be positive",
            ));
        }
        if key_dtype == IdentityDType::Float16 || value_dtype == IdentityDType::Float16 {
            return Err(AmsError::new(
                ErrorCode::CapabilityMismatch,
                "K/V cache FP16 encoding is not qualified",
            ));
        }
        let key_row_elements = mul(
            head_count,
            key_head_dimension,
            "K/V cache key row elements overflow",
        )?;
        let value_row_elements = mul(
            head_count,
            value_head_dimension,
            "K/V cache value row elements overflow",
        )?;
        let key_row_bytes = mul(
            key_row_elements,
            key_dtype.item_bytes(),
            "K/V cache key row bytes overflow",
        )?;
        let value_row_bytes = mul(
            value_row_elements,
            value_dtype.item_bytes(),
            "K/V cache value row bytes overflow",
        )?;
        let key_storage_bytes = mul(
            capacity_tokens,
            key_row_bytes,
            "K/V cache key storage bytes overflow",
        )?;
        let value_storage_bytes = mul(
            capacity_tokens,
            value_row_bytes,
            "K/V cache value storage bytes overflow",
        )?;
        let staging_bytes = add(
            key_row_bytes,
            value_row_bytes,
            "K/V cache staging bytes overflow",
        )?;
        let storage_bytes = add(
            key_storage_bytes,
            value_storage_bytes,
            "K/V cache total storage bytes overflow",
        )?;
        usize_to_u64(key_storage_bytes, "K/V cache key storage exceeds u64")?;
        usize_to_u64(value_storage_bytes, "K/V cache value storage exceeds u64")?;
        Ok(Self {
            head_count,
            key_head_dimension,
            value_head_dimension,
            capacity_tokens,
            key_dtype,
            value_dtype,
            key_row_elements,
            value_row_elements,
            key_row_bytes,
            value_row_bytes,
            requirements: KvCacheRequirements {
                key_storage_bytes,
                value_storage_bytes,
                staging_bytes,
                storage_bytes,
            },
        })
    }

    /// Exact fixed-capacity storage and per-append staging requirements.
    #[must_use]
    pub const fn requirements(self) -> KvCacheRequirements {
        self.requirements
    }

    /// Number of attention heads in every cache row.
    #[must_use]
    pub const fn head_count(self) -> usize {
        self.head_count
    }

    /// Per-head key dimension.
    #[must_use]
    pub const fn key_head_dimension(self) -> usize {
        self.key_head_dimension
    }

    /// Per-head value dimension.
    #[must_use]
    pub const fn value_head_dimension(self) -> usize {
        self.value_head_dimension
    }

    /// Maximum committed token rows.
    #[must_use]
    pub const fn capacity_tokens(self) -> usize {
        self.capacity_tokens
    }

    /// Key storage dtype.
    #[must_use]
    pub const fn key_dtype(self) -> IdentityDType {
        self.key_dtype
    }

    /// Value storage dtype.
    #[must_use]
    pub const fn value_dtype(self) -> IdentityDType {
        self.value_dtype
    }
}

/// Caller-owned fixed-capacity cache with prefix-only visibility.
pub struct KvCache<'a> {
    plan: KvCachePlan,
    keys: &'a mut [u8],
    values: &'a mut [u8],
    committed_tokens: usize,
}

impl<'a> KvCache<'a> {
    /// Bind exact-sized key and value arenas without initializing or exposing any row.
    ///
    /// # Errors
    ///
    /// Returns `PREFLIGHT_NO_WORKING_SET` unless both arenas exactly match the admitted plan.
    pub const fn new(
        plan: KvCachePlan,
        keys: &'a mut [u8],
        values: &'a mut [u8],
    ) -> Result<Self, AmsError> {
        let requirement = plan.requirements;
        if keys.len() != requirement.key_storage_bytes
            || values.len() != requirement.value_storage_bytes
        {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "K/V cache arenas differ from the admitted fixed capacity",
            ));
        }
        Ok(Self {
            plan,
            keys,
            values,
            committed_tokens: 0,
        })
    }

    /// Number of sequential token rows currently visible to readers.
    #[must_use]
    pub const fn committed_tokens(&self) -> usize {
        self.committed_tokens
    }

    /// Encode and append exactly the next token row, publishing the prefix length last.
    ///
    /// Numeric and capacity failures leave both storage arenas and the committed prefix unchanged.
    /// A caller may safely retry the same position after any returned error.
    ///
    /// # Errors
    ///
    /// Returns a typed plan, capacity, or numeric error.
    pub fn append(
        &mut self,
        position: usize,
        key: &[f64],
        value: &[f64],
        staging: &mut [u8],
    ) -> Result<(), AmsError> {
        if position != self.committed_tokens {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "K/V cache append position is not the next uncommitted row",
            ));
        }
        if position >= self.plan.capacity_tokens {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "K/V cache fixed token capacity is exhausted",
            ));
        }
        if key.len() != self.plan.key_row_elements || value.len() != self.plan.value_row_elements {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "K/V cache append vector dimensions differ from the plan",
            ));
        }
        if staging.len() < self.plan.requirements.staging_bytes {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "K/V cache staging row is smaller than the admitted plan",
            ));
        }
        if key.iter().chain(value).any(|scalar| !scalar.is_finite()) {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "K/V cache append contains a non-finite value",
            ));
        }

        let (staged_key, staged_value) =
            staging[..self.plan.requirements.staging_bytes].split_at_mut(self.plan.key_row_bytes);
        encode_row(key, self.plan.key_dtype, staged_key)?;
        encode_row(value, self.plan.value_dtype, staged_value)?;

        let key_start = mul(
            position,
            self.plan.key_row_bytes,
            "K/V cache key append offset overflow",
        )?;
        let value_start = mul(
            position,
            self.plan.value_row_bytes,
            "K/V cache value append offset overflow",
        )?;
        self.keys[key_start..key_start + self.plan.key_row_bytes].copy_from_slice(staged_key);
        self.values[value_start..value_start + self.plan.value_row_bytes]
            .copy_from_slice(staged_value);
        self.committed_tokens = add(
            self.committed_tokens,
            1,
            "K/V cache committed prefix overflow",
        )?;
        Ok(())
    }

    /// Borrow a read-only view limited to the completely committed token prefix.
    #[must_use]
    pub fn view(&self) -> KvCacheView<'_> {
        let key_bytes = self.committed_tokens * self.plan.key_row_bytes;
        let value_bytes = self.committed_tokens * self.plan.value_row_bytes;
        KvCacheView {
            keys: KvCacheReader {
                bytes: &self.keys[..key_bytes],
            },
            values: KvCacheReader {
                bytes: &self.values[..value_bytes],
            },
            committed_tokens: self.committed_tokens,
        }
    }
}

#[allow(clippy::cast_possible_truncation)] // Conversion is range-checked before publication.
fn encode_row(values: &[f64], dtype: IdentityDType, output: &mut [u8]) -> Result<(), AmsError> {
    let item_bytes = dtype.item_bytes();
    if output.len() != values.len() * item_bytes {
        return Err(AmsError::new(
            ErrorCode::InternalInvariant,
            "K/V cache encoded row dimensions differ",
        ));
    }
    for (index, scalar) in values.iter().copied().enumerate() {
        let value = scalar as f32;
        if !value.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "K/V cache value exceeds the admitted storage dtype",
            ));
        }
        let offset = index * item_bytes;
        match dtype {
            IdentityDType::BFloat16 => {
                let bits = value.to_bits();
                let rounding_bias = 0x7fff + ((bits >> 16) & 1);
                let encoded =
                    u16::try_from(bits.wrapping_add(rounding_bias) >> 16).map_err(|_| {
                        AmsError::new(ErrorCode::InternalInvariant, "BF16 encoding exceeds u16")
                    })?;
                if encoded & 0x7f80 == 0x7f80 {
                    return Err(AmsError::new(
                        ErrorCode::NumericFailure,
                        "K/V cache BF16 rounding produced a non-finite value",
                    ));
                }
                output[offset..offset + 2].copy_from_slice(&encoded.to_le_bytes());
            }
            IdentityDType::Float32 => {
                output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            IdentityDType::Float16 => {
                return Err(AmsError::new(
                    ErrorCode::InternalInvariant,
                    "unqualified K/V cache FP16 plan reached execution",
                ));
            }
        }
    }
    Ok(())
}

/// Read-only prefix view returned by [`KvCache::view`].
pub struct KvCacheView<'a> {
    keys: KvCacheReader<'a>,
    values: KvCacheReader<'a>,
    committed_tokens: usize,
}

impl KvCacheView<'_> {
    /// Completely committed token rows visible through both readers.
    #[must_use]
    pub const fn committed_tokens(&self) -> usize {
        self.committed_tokens
    }

    /// Prefix-limited key reader.
    #[must_use]
    pub const fn key_reader(&self) -> &dyn RangeReader {
        &self.keys
    }

    /// Prefix-limited value reader.
    #[must_use]
    pub const fn value_reader(&self) -> &dyn RangeReader {
        &self.values
    }
}

struct KvCacheReader<'a> {
    bytes: &'a [u8],
}

impl RangeReader for KvCacheReader<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
        let start = usize::try_from(offset).map_err(|_| {
            AmsError::new(ErrorCode::IoFailure, "K/V cache read offset exceeds usize")
        })?;
        let end = start
            .checked_add(destination.len())
            .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "K/V cache read range overflow"))?;
        let source = self.bytes.get(start..end).ok_or_else(|| {
            AmsError::new(
                ErrorCode::IoFailure,
                "K/V cache read exceeds the committed prefix",
            )
        })?;
        destination.copy_from_slice(source);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FullAttentionPlan, FullAttentionReaders, FullAttentionScratch, FullAttentionShape,
        FullKvLayout, glm_full_attention,
    };

    #[test]
    #[allow(clippy::suboptimal_flops)] // Keep the independent two-term reference explicit.
    fn cache_publishes_exact_prefixes_consumable_by_full_attention() -> Result<(), AmsError> {
        let plan = KvCachePlan::new(1, 2, 2, 3, IdentityDType::BFloat16, IdentityDType::Float32)?;
        let requirement = plan.requirements();
        assert_eq!(requirement.key_storage_bytes, 12);
        assert_eq!(requirement.value_storage_bytes, 24);
        assert_eq!(requirement.staging_bytes, 12);
        let mut key_storage = [0xa5u8; 12];
        let mut value_storage = [0xa5u8; 24];
        let mut cache = KvCache::new(plan, &mut key_storage, &mut value_storage)?;
        let mut staging = [0u8; 12];
        cache.append(0, &[1.0, 0.0], &[1.0, 10.0], &mut staging)?;
        assert_eq!(cache.committed_tokens(), 1);
        assert_eq!(cache.view().key_reader().len(), 4);
        cache.append(1, &[0.0, 1.0], &[3.0, 30.0], &mut staging)?;
        let view = cache.view();
        assert_eq!(view.committed_tokens(), 2);
        assert_eq!(view.key_reader().len(), 8);
        assert_eq!(view.value_reader().len(), 16);

        let attention_shape = FullAttentionShape::new(1, 2, 2, 2, 1)?;
        let attention_layout =
            FullKvLayout::new(0, 0, IdentityDType::BFloat16, IdentityDType::Float32);
        let attention_plan = FullAttentionPlan::from_arena(attention_shape, attention_layout, 56)?;
        let readers = FullAttentionReaders::new(view.key_reader(), view.value_reader());
        let mut encoded = [0u8; 8];
        let mut key = [0.0f64; 2];
        let mut value = [0.0f64; 2];
        let mut transactional = [0.0f64; 2];
        let mut scratch =
            FullAttentionScratch::new(&mut encoded, &mut key, &mut value, &mut transactional);
        let mut output = [0.0f64; 2];
        glm_full_attention(
            attention_plan,
            &readers,
            &[1.0, 0.0],
            &mut scratch,
            &mut output,
        )?;
        let first_weight = (1.0 / 2.0f64.sqrt()).exp();
        let second_weight = 1.0;
        let denominator = first_weight + second_weight;
        let expected = [
            (first_weight + 3.0 * second_weight) / denominator,
            (10.0 * first_weight + 30.0 * second_weight) / denominator,
        ];
        for (actual, reference) in output.iter().zip(expected) {
            assert!((actual - reference).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn failed_append_preserves_storage_and_committed_prefix() -> Result<(), AmsError> {
        let plan = KvCachePlan::new(1, 2, 1, 2, IdentityDType::BFloat16, IdentityDType::BFloat16)?;
        let mut key_storage = [0x5au8; 8];
        let mut value_storage = [0x5au8; 4];
        let original_keys = key_storage;
        let original_values = value_storage;
        let mut cache = KvCache::new(plan, &mut key_storage, &mut value_storage)?;
        let mut staging = [0u8; 6];
        let error = cache
            .append(0, &[1.0, f64::INFINITY], &[2.0], &mut staging)
            .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert_eq!(cache.committed_tokens(), 0);
        assert_eq!(cache.keys, original_keys);
        assert_eq!(cache.values, original_values);

        let error = cache
            .append(0, &[f64::MAX, 1.0], &[2.0], &mut staging)
            .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert_eq!(cache.committed_tokens(), 0);
        assert_eq!(cache.keys, original_keys);
        assert_eq!(cache.values, original_values);
        Ok(())
    }

    #[test]
    fn cache_rejects_out_of_order_short_and_over_capacity_appends() -> Result<(), AmsError> {
        let plan = KvCachePlan::new(1, 1, 1, 1, IdentityDType::Float32, IdentityDType::Float32)?;
        let mut key_storage = [0u8; 4];
        let mut value_storage = [0u8; 4];
        let mut cache = KvCache::new(plan, &mut key_storage, &mut value_storage)?;
        let mut short_staging = [0u8; 7];
        let error = cache.append(0, &[1.0], &[2.0], &mut short_staging).err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        let mut staging = [0u8; 8];
        let error = cache.append(1, &[1.0], &[2.0], &mut staging).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        cache.append(0, &[1.0], &[2.0], &mut staging)?;
        let error = cache.append(1, &[1.0], &[2.0], &mut staging).err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        assert_eq!(cache.committed_tokens(), 1);
        assert_eq!(
            KvCachePlan::new(1, 1, 1, 1, IdentityDType::Float16, IdentityDType::BFloat16,)
                .err()
                .map(AmsError::code),
            Some(ErrorCode::CapabilityMismatch)
        );
        Ok(())
    }
}
