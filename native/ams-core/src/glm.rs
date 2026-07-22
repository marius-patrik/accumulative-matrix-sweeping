use crate::checked::{add, mul};
use crate::{AmsError, ErrorCode};

fn require_finite(values: &[f64], context: &'static str) -> Result<(), AmsError> {
    if values.is_empty() {
        return Err(AmsError::new(ErrorCode::PlanInvalid, context));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(ErrorCode::NumericFailure, context));
    }
    Ok(())
}

fn finite_positive(value: f64, context: &'static str) -> Result<(), AmsError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(AmsError::new(ErrorCode::PlanInvalid, context));
    }
    Ok(())
}

fn dimension_as_f64(value: usize, context: &'static str) -> Result<f64, AmsError> {
    let bounded =
        u32::try_from(value).map_err(|_| AmsError::new(ErrorCode::PlanInvalid, context))?;
    Ok(f64::from(bounded))
}

/// Apply GLM `RMSNorm` with a fixed source-order square reduction.
///
/// # Errors
///
/// Returns a typed plan or numeric error for invalid dimensions or non-finite data.
#[allow(clippy::suboptimal_flops)] // Preserve the Python semantic oracle's operation order.
pub fn glm_rms_norm(
    values: &[f64],
    weight: &[f64],
    epsilon: f64,
    output: &mut [f64],
) -> Result<(), AmsError> {
    require_finite(values, "RMSNorm values are empty or non-finite")?;
    require_finite(weight, "RMSNorm weight is empty or non-finite")?;
    finite_positive(epsilon, "RMSNorm epsilon must be finite and positive")?;
    if values.len() != weight.len() || values.len() != output.len() {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "RMSNorm vector dimensions differ",
        ));
    }
    let mut square_sum = 0.0;
    for value in values {
        square_sum += value * value;
    }
    let width = dimension_as_f64(values.len(), "RMSNorm width exceeds supported range")?;
    let inverse_root_mean_square = 1.0 / (square_sum / width + epsilon).sqrt();
    for ((destination, value), scale) in output.iter_mut().zip(values).zip(weight) {
        *destination = value * inverse_root_mean_square * scale;
        if !destination.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "RMSNorm output is non-finite",
            ));
        }
    }
    Ok(())
}

/// Apply the GLM DSA indexer `LayerNorm` with fixed two-pass reductions.
///
/// # Errors
///
/// Returns a typed plan or numeric error for invalid dimensions or non-finite data.
#[allow(clippy::suboptimal_flops)] // Preserve the Python semantic oracle's operation order.
pub fn glm_layer_norm(
    values: &[f64],
    weight: &[f64],
    bias: &[f64],
    epsilon: f64,
    output: &mut [f64],
) -> Result<(), AmsError> {
    require_finite(values, "LayerNorm values are empty or non-finite")?;
    require_finite(weight, "LayerNorm weight is empty or non-finite")?;
    require_finite(bias, "LayerNorm bias is empty or non-finite")?;
    finite_positive(epsilon, "LayerNorm epsilon must be finite and positive")?;
    if values.len() != weight.len() || values.len() != bias.len() || values.len() != output.len() {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "LayerNorm vector dimensions differ",
        ));
    }
    let width = dimension_as_f64(values.len(), "LayerNorm width exceeds supported range")?;
    let mut value_sum = 0.0;
    for value in values {
        value_sum += value;
    }
    let mean = value_sum / width;
    let mut square_deviation_sum = 0.0;
    for value in values {
        let deviation = value - mean;
        square_deviation_sum += deviation * deviation;
    }
    let inverse_standard_deviation = 1.0 / (square_deviation_sum / width + epsilon).sqrt();
    for (((destination, value), scale), offset) in
        output.iter_mut().zip(values).zip(weight).zip(bias)
    {
        *destination = (value - mean) * inverse_standard_deviation * scale + offset;
        if !destination.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "LayerNorm output is non-finite",
            ));
        }
    }
    Ok(())
}

/// Evaluate the stable scalar `SiLU` used by GLM gated MLPs.
///
/// # Errors
///
/// Returns `NUMERIC_FAILURE` for a non-finite input or result.
pub fn glm_silu(value: f64) -> Result<f64, AmsError> {
    if !value.is_finite() {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "SiLU input is non-finite",
        ));
    }
    let result = if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        value * exponential / (1.0 + exponential)
    };
    if !result.is_finite() {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "SiLU output is non-finite",
        ));
    }
    Ok(result)
}

/// Apply max-shifted GLM softmax using the output buffer as exponential scratch.
///
/// # Errors
///
/// Returns a typed plan or numeric error for invalid dimensions or non-finite data.
pub fn glm_softmax(values: &[f64], output: &mut [f64]) -> Result<(), AmsError> {
    require_finite(values, "softmax values are empty or non-finite")?;
    if values.len() != output.len() {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "softmax vector dimensions differ",
        ));
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut denominator = 0.0;
    for (destination, value) in output.iter_mut().zip(values) {
        *destination = (value - maximum).exp();
        denominator += *destination;
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "softmax denominator is invalid",
        ));
    }
    for value in output {
        *value /= denominator;
    }
    Ok(())
}

fn validate_rope(values: &[f64], output: &[f64], theta: f64) -> Result<usize, AmsError> {
    require_finite(values, "RoPE values are empty or non-finite")?;
    finite_positive(theta, "RoPE theta must be finite and positive")?;
    if values.len() != output.len() || values.len() % 2 != 0 {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "RoPE dimensions must match and be even",
        ));
    }
    Ok(values.len() / 2)
}

fn rope_angle(
    position: usize,
    pair_index: usize,
    dimension: usize,
    theta: f64,
) -> Result<f64, AmsError> {
    let position = dimension_as_f64(position, "RoPE position exceeds supported range")?;
    let pair_index = dimension_as_f64(pair_index, "RoPE pair index exceeds supported range")?;
    let dimension = dimension_as_f64(dimension, "RoPE dimension exceeds supported range")?;
    Ok(position / theta.powf(2.0 * pair_index / dimension))
}

/// Apply the main MLA interleaved-pair rotary layout.
///
/// # Errors
///
/// Returns a typed plan or numeric error for invalid dimensions or non-finite data.
#[allow(clippy::suboptimal_flops)] // Preserve the Python semantic oracle's operation order.
pub fn glm_rope_interleaved(
    values: &[f64],
    position: usize,
    theta: f64,
    output: &mut [f64],
) -> Result<(), AmsError> {
    let pair_count = validate_rope(values, output, theta)?;
    for pair_index in 0..pair_count {
        let angle = rope_angle(position, pair_index, values.len(), theta)?;
        let cosine = angle.cos();
        let sine = angle.sin();
        let source_offset = pair_index * 2;
        let left = values[source_offset];
        let right = values[source_offset + 1];
        output[source_offset] = left * cosine - right * sine;
        output[source_offset + 1] = right * cosine + left * sine;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "interleaved RoPE output is non-finite",
        ));
    }
    Ok(())
}

/// Apply the DSA indexer half-split rotary layout.
///
/// # Errors
///
/// Returns a typed plan or numeric error for invalid dimensions or non-finite data.
#[allow(clippy::suboptimal_flops)] // Preserve the Python semantic oracle's operation order.
pub fn glm_rope_half_split(
    values: &[f64],
    position: usize,
    theta: f64,
    output: &mut [f64],
) -> Result<(), AmsError> {
    let half = validate_rope(values, output, theta)?;
    for index in 0..half {
        let angle = rope_angle(position, index, values.len(), theta)?;
        let cosine = angle.cos();
        let sine = angle.sin();
        let left = values[index];
        let right = values[half + index];
        output[index] = left * cosine - right * sine;
        output[half + index] = right * cosine + left * sine;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "half-split RoPE output is non-finite",
        ));
    }
    Ok(())
}

/// Immutable dimensions and selection bounds for one causal DSA top-k operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DsaTopKPlan {
    head_count: usize,
    head_dimension: usize,
    key_count: usize,
    query_position: usize,
    selected_count: usize,
    score_scratch_bytes: usize,
}

impl DsaTopKPlan {
    /// Validate DSA dimensions and derive the exact causal score scratch requirement.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for zero, inconsistent, or overflowing dimensions.
    pub fn new(
        head_count: usize,
        head_dimension: usize,
        key_count: usize,
        query_position: usize,
        top_k: usize,
    ) -> Result<Self, AmsError> {
        if head_count == 0
            || head_dimension == 0
            || key_count == 0
            || top_k == 0
            || query_position >= key_count
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "DSA dimensions or selection bounds are invalid",
            ));
        }
        mul(
            head_count,
            head_dimension,
            "DSA query element count overflow",
        )?;
        mul(key_count, head_dimension, "DSA key element count overflow")?;
        let causal_count = add(query_position, 1, "DSA causal count overflow")?;
        let score_scratch_bytes = mul(
            causal_count,
            size_of::<f64>(),
            "DSA score scratch bytes overflow",
        )?;
        Ok(Self {
            head_count,
            head_dimension,
            key_count,
            query_position,
            selected_count: top_k.min(causal_count),
            score_scratch_bytes,
        })
    }

    /// Exact number of causal score elements required from the caller.
    #[must_use]
    pub const fn score_scratch_len(self) -> usize {
        self.query_position + 1
    }

    /// Exact score scratch bytes required from the caller.
    #[must_use]
    pub const fn score_scratch_bytes(self) -> usize {
        self.score_scratch_bytes
    }

    /// Number of selected indices emitted by this plan.
    #[must_use]
    pub const fn selected_count(self) -> usize {
        self.selected_count
    }
}

/// Rank causal DSA keys with deterministic score and lowest-index tie breaking.
///
/// Query heads and key vectors are flattened in head-major and key-major order.
///
/// # Errors
///
/// Returns a typed plan, capacity, or numeric error for invalid buffers or data.
#[allow(clippy::suboptimal_flops)] // Preserve the Python semantic oracle's operation order.
pub fn glm_dsa_topk(
    plan: DsaTopKPlan,
    query_heads: &[f64],
    key_vectors: &[f64],
    head_weights: &[f64],
    score_scratch: &mut [f64],
    selected: &mut [usize],
) -> Result<(), AmsError> {
    let query_elements = mul(
        plan.head_count,
        plan.head_dimension,
        "DSA query element count overflow",
    )?;
    let key_elements = mul(
        plan.key_count,
        plan.head_dimension,
        "DSA key element count overflow",
    )?;
    if query_heads.len() != query_elements
        || key_vectors.len() != key_elements
        || head_weights.len() != plan.head_count
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "DSA input dimensions are invalid",
        ));
    }
    if score_scratch.len() < plan.score_scratch_len() || selected.len() < plan.selected_count {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "DSA output or scratch is smaller than the admitted plan",
        ));
    }
    require_finite(query_heads, "DSA query heads are empty or non-finite")?;
    require_finite(key_vectors, "DSA key vectors are empty or non-finite")?;
    require_finite(head_weights, "DSA head weights are empty or non-finite")?;
    let scale = 1.0
        / dimension_as_f64(
            plan.head_dimension,
            "DSA head dimension exceeds supported range",
        )?
        .sqrt();
    for (key_index, score_slot) in score_scratch
        .iter_mut()
        .take(plan.score_scratch_len())
        .enumerate()
    {
        let key_start = key_index * plan.head_dimension;
        let key = &key_vectors[key_start..key_start + plan.head_dimension];
        let mut score = 0.0;
        for (head_index, query) in query_heads.chunks_exact(plan.head_dimension).enumerate() {
            let mut similarity = 0.0;
            for (query_value, key_value) in query.iter().zip(key) {
                similarity += query_value * key_value;
            }
            score += head_weights[head_index] * (similarity * scale).max(0.0);
        }
        if !score.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "DSA score is non-finite",
            ));
        }
        *score_slot = score;
    }
    for rank in 0..plan.selected_count {
        let mut best: Option<(usize, f64)> = None;
        for (key_index, score) in score_scratch
            .iter()
            .take(plan.score_scratch_len())
            .copied()
            .enumerate()
        {
            if selected[..rank].contains(&key_index) {
                continue;
            }
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((key_index, score));
            }
        }
        selected[rank] = best
            .ok_or_else(|| AmsError::new(ErrorCode::InternalInvariant, "DSA selection exhausted"))?
            .0;
    }
    Ok(())
}

/// Immutable dimensions and policy for GLM `sigmoid/noaux_tc` expert routing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlmRouterPlan {
    expert_count: usize,
    experts_per_token: usize,
    group_count: usize,
    top_groups: usize,
    experts_per_group: usize,
    routed_scaling_factor: f64,
    scratch_bytes: usize,
}

impl GlmRouterPlan {
    /// Validate routing dimensions and derive the caller-owned scratch requirement.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for inconsistent policy or checked-size overflow.
    pub fn new(
        expert_count: usize,
        experts_per_token: usize,
        group_count: usize,
        top_groups: usize,
        routed_scaling_factor: f64,
    ) -> Result<Self, AmsError> {
        if expert_count == 0
            || experts_per_token == 0
            || group_count == 0
            || top_groups == 0
            || expert_count % group_count != 0
            || experts_per_token > expert_count
            || top_groups > group_count
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "router dimensions or selection bounds are invalid",
            ));
        }
        let experts_per_group = expert_count / group_count;
        let selected_capacity = mul(
            top_groups,
            experts_per_group,
            "router selected-group capacity overflow",
        )?;
        if experts_per_token > selected_capacity {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "router selected groups cannot contain the requested experts",
            ));
        }
        finite_positive(
            routed_scaling_factor,
            "router scaling factor must be finite and positive",
        )?;
        let float_elements = add(
            mul(expert_count, 2, "router expert scratch count overflow")?,
            group_count,
            "router float scratch count overflow",
        )?;
        let scratch_bytes = add(
            mul(
                float_elements,
                size_of::<f64>(),
                "router float scratch bytes overflow",
            )?,
            mul(
                top_groups,
                size_of::<usize>(),
                "router group scratch bytes overflow",
            )?,
            "router total scratch bytes overflow",
        )?;
        Ok(Self {
            expert_count,
            experts_per_token,
            group_count,
            top_groups,
            experts_per_group,
            routed_scaling_factor,
            scratch_bytes,
        })
    }

    /// Exact logical caller-owned scratch bytes required by the route operation.
    #[must_use]
    pub const fn scratch_bytes(self) -> usize {
        self.scratch_bytes
    }

    /// Number of expert indices and weights emitted by this plan.
    #[must_use]
    pub const fn selected_count(self) -> usize {
        self.experts_per_token
    }
}

/// Borrowed scratch for allocation-free GLM expert routing.
pub struct GlmRouterScratch<'a> {
    probabilities: &'a mut [f64],
    corrected: &'a mut [f64],
    group_scores: &'a mut [f64],
    selected_groups: &'a mut [usize],
}

impl<'a> GlmRouterScratch<'a> {
    /// Group the four caller-owned routing scratch regions.
    #[must_use]
    pub const fn new(
        probabilities: &'a mut [f64],
        corrected: &'a mut [f64],
        group_scores: &'a mut [f64],
        selected_groups: &'a mut [usize],
    ) -> Self {
        Self {
            probabilities,
            corrected,
            group_scores,
            selected_groups,
        }
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// Execute GLM `sigmoid/noaux_tc` routing with deterministic lowest-index ties.
///
/// Correction bias affects selection only; emitted weights use unbiased sigmoid probabilities.
///
/// # Errors
///
/// Returns a typed plan, capacity, or numeric error for invalid buffers or data.
#[allow(clippy::too_many_lines)] // The phases share one explicit caller-owned scratch contract.
pub fn glm_route_experts(
    plan: GlmRouterPlan,
    router_logits: &[f64],
    correction_bias: &[f64],
    scratch: &mut GlmRouterScratch<'_>,
    expert_indices: &mut [usize],
    expert_weights: &mut [f64],
) -> Result<(), AmsError> {
    if router_logits.len() != plan.expert_count || correction_bias.len() != plan.expert_count {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "router input dimensions are invalid",
        ));
    }
    if scratch.probabilities.len() < plan.expert_count
        || scratch.corrected.len() < plan.expert_count
        || scratch.group_scores.len() < plan.group_count
        || scratch.selected_groups.len() < plan.top_groups
        || expert_indices.len() < plan.experts_per_token
        || expert_weights.len() < plan.experts_per_token
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "router output or scratch is smaller than the admitted plan",
        ));
    }
    require_finite(router_logits, "router logits are empty or non-finite")?;
    require_finite(
        correction_bias,
        "router correction bias is empty or non-finite",
    )?;
    for (((probability, corrected), logit), bias) in scratch
        .probabilities
        .iter_mut()
        .zip(scratch.corrected.iter_mut())
        .zip(router_logits)
        .zip(correction_bias)
        .take(plan.expert_count)
    {
        *probability = sigmoid(*logit);
        *corrected = *probability + bias;
        if !corrected.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "router corrected probability is non-finite",
            ));
        }
    }
    for (group_score, group_values) in scratch.group_scores.iter_mut().take(plan.group_count).zip(
        scratch
            .corrected
            .chunks_exact(plan.experts_per_group)
            .take(plan.group_count),
    ) {
        let mut first = f64::NEG_INFINITY;
        let mut second = f64::NEG_INFINITY;
        for value in group_values {
            if *value > first {
                second = first;
                first = *value;
            } else if *value > second {
                second = *value;
            }
        }
        *group_score = if plan.experts_per_group == 1 {
            first
        } else {
            first + second
        };
        if !group_score.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "router group score is non-finite",
            ));
        }
    }
    for rank in 0..plan.top_groups {
        let mut best: Option<(usize, f64)> = None;
        for (group_index, score) in scratch
            .group_scores
            .iter()
            .take(plan.group_count)
            .copied()
            .enumerate()
        {
            if scratch.selected_groups[..rank].contains(&group_index) {
                continue;
            }
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((group_index, score));
            }
        }
        scratch.selected_groups[rank] = best
            .ok_or_else(|| {
                AmsError::new(
                    ErrorCode::InternalInvariant,
                    "router group selection exhausted",
                )
            })?
            .0;
    }
    for rank in 0..plan.experts_per_token {
        let mut best: Option<(usize, f64)> = None;
        for (expert_index, corrected) in scratch
            .corrected
            .iter()
            .take(plan.expert_count)
            .copied()
            .enumerate()
        {
            let group_index = expert_index / plan.experts_per_group;
            if !scratch.selected_groups[..plan.top_groups].contains(&group_index)
                || expert_indices[..rank].contains(&expert_index)
            {
                continue;
            }
            if best.is_none_or(|(_, best_score)| corrected > best_score) {
                best = Some((expert_index, corrected));
            }
        }
        expert_indices[rank] = best
            .ok_or_else(|| {
                AmsError::new(
                    ErrorCode::InternalInvariant,
                    "router expert selection exhausted",
                )
            })?
            .0;
    }
    let mut denominator = 0.0;
    for expert_index in expert_indices.iter().take(plan.experts_per_token) {
        denominator += scratch.probabilities[*expert_index];
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "router weight denominator is invalid",
        ));
    }
    for (destination, expert_index) in expert_weights
        .iter_mut()
        .zip(expert_indices.iter())
        .take(plan.experts_per_token)
    {
        *destination =
            scratch.probabilities[*expert_index] / denominator * plan.routed_scaling_factor;
        if !destination.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "router expert weight is non-finite",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(left, right)| (left - right).abs() <= tolerance)
        );
    }

    #[test]
    fn native_norms_follow_fixed_reference_reductions() -> Result<(), AmsError> {
        let values = [1.0, -2.0, 3.0, -4.0];
        let mut rms = [0.0; 4];
        glm_rms_norm(&values, &[1.0, 0.5, 2.0, -1.0], 1e-5, &mut rms)?;
        let inverse_rms = 1.0 / (30.0f64 / 4.0 + 1e-5).sqrt();
        let expected = [
            inverse_rms,
            -inverse_rms,
            6.0 * inverse_rms,
            4.0 * inverse_rms,
        ];
        assert_close(&rms, &expected, 1e-15);
        assert_close(
            &rms,
            &[
                0.365_148_128_238_106_4,
                -0.365_148_128_238_106_4,
                2.190_888_769_428_638_3,
                1.460_592_512_952_425_5,
            ],
            1e-15,
        );

        let mut normalized = [0.0; 4];
        glm_layer_norm(&values, &[1.0; 4], &[0.0; 4], 1e-6, &mut normalized)?;
        let sum: f64 = normalized.iter().sum();
        let square_mean: f64 = normalized.iter().map(|value| value * value).sum::<f64>() / 4.0;
        assert!(sum.abs() <= 1e-15);
        assert!((square_mean - 1.0).abs() <= 1e-6);
        assert_close(
            &normalized,
            &[
                0.557_085_976_111_434_4,
                -0.557_085_976_111_434_4,
                1.299_867_277_593_347,
                -1.299_867_277_593_347,
            ],
            1e-15,
        );
        Ok(())
    }

    #[test]
    fn native_rope_layouts_are_distinct_and_norm_preserving() -> Result<(), AmsError> {
        let values = [1.0, 2.0, 3.0, 4.0];
        let mut interleaved = [0.0; 4];
        let mut half_split = [0.0; 4];
        glm_rope_interleaved(&values, 7, 10_000.0, &mut interleaved)?;
        glm_rope_half_split(&values, 7, 10_000.0, &mut half_split)?;
        assert!(
            interleaved
                .iter()
                .zip(half_split)
                .any(|(left, right)| (left - right).abs() > f64::EPSILON)
        );
        let expected_norm: f64 = values.iter().map(|value| value * value).sum();
        let interleaved_norm: f64 = interleaved.iter().map(|value| value * value).sum();
        let half_split_norm: f64 = half_split.iter().map(|value| value * value).sum();
        assert!((interleaved_norm - expected_norm).abs() <= 1e-12);
        assert!((half_split_norm - expected_norm).abs() <= 1e-12);
        assert_close(
            &interleaved,
            &[
                -0.560_070_943_094_273_5,
                2.164_791_107_405_398_5,
                2.712_881_611_409_707_6,
                4.200_032_543_025_717,
            ],
            2e-15,
        );
        assert_close(
            &half_split,
            &[
                -1.217_057_541_813_062_5,
                1.715_330_611_156_428_2,
                2.918_693_361_748_703,
                4.130_089_695_688_184,
            ],
            2e-15,
        );
        Ok(())
    }

    #[test]
    fn native_activation_and_softmax_are_stable() -> Result<(), AmsError> {
        assert!(glm_silu(-1000.0)?.abs() <= 1e-300);
        let mut probabilities = [0.0; 3];
        glm_softmax(&[1000.0, 999.0, -1000.0], &mut probabilities)?;
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() <= f64::EPSILON);
        assert!(probabilities[0] > probabilities[1]);
        assert!(probabilities[1] > probabilities[2]);
        Ok(())
    }

    #[test]
    fn native_dsa_is_causal_and_breaks_ties_by_key_index() -> Result<(), AmsError> {
        let plan = DsaTopKPlan::new(2, 2, 4, 2, 3)?;
        assert_eq!(plan.score_scratch_len(), 3);
        assert_eq!(plan.score_scratch_bytes(), 3 * size_of::<f64>());
        let mut scores = [0.0; 3];
        let mut selected = [usize::MAX; 3];
        glm_dsa_topk(
            plan,
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0, 3.0, 0.0, 100.0, 100.0],
            &[0.5, 0.5],
            &mut scores,
            &mut selected,
        )?;
        assert_eq!(selected, [2, 0, 1]);
        Ok(())
    }

    #[test]
    fn native_noaux_router_separates_bias_from_output_weights() -> Result<(), AmsError> {
        let plan = GlmRouterPlan::new(8, 2, 2, 1, 2.5)?;
        assert_eq!(plan.selected_count(), 2);
        assert_eq!(
            plan.scratch_bytes(),
            18 * size_of::<f64>() + size_of::<usize>()
        );
        let mut probabilities = [0.0; 8];
        let mut corrected = [0.0; 8];
        let mut group_scores = [0.0; 2];
        let mut selected_groups = [usize::MAX; 1];
        let mut scratch = GlmRouterScratch::new(
            &mut probabilities,
            &mut corrected,
            &mut group_scores,
            &mut selected_groups,
        );
        let mut indices = [usize::MAX; 2];
        let mut weights = [0.0; 2];
        glm_route_experts(
            plan,
            &[4.0, 3.0, 2.0, 1.0, 5.0, 4.0, -1.0, -2.0],
            &[0.0, 0.0, 0.0, 0.0, -10.0, -10.0, 10.0, 10.0],
            &mut scratch,
            &mut indices,
            &mut weights,
        )?;
        assert_eq!(indices, [6, 7]);
        let raw = [1.0 / (1.0 + 1.0f64.exp()), 1.0 / (1.0 + 2.0f64.exp())];
        let denominator = raw.iter().sum::<f64>();
        let first_expected = raw[0] / denominator * 2.5;
        let second_expected = raw[1] / denominator * 2.5;
        assert!((weights[0] - first_expected).abs() <= 1e-15);
        assert!((weights[1] - second_expected).abs() <= 1e-15);
        Ok(())
    }

    #[test]
    fn native_router_accepts_exact_selected_group_capacity() -> Result<(), AmsError> {
        let plan = GlmRouterPlan::new(4, 2, 2, 1, 1.0)?;
        let mut probabilities = [0.0; 4];
        let mut corrected = [0.0; 4];
        let mut group_scores = [0.0; 2];
        let mut selected_groups = [usize::MAX; 1];
        let mut scratch = GlmRouterScratch::new(
            &mut probabilities,
            &mut corrected,
            &mut group_scores,
            &mut selected_groups,
        );
        let mut indices = [usize::MAX; 2];
        let mut weights = [0.0; 2];
        glm_route_experts(
            plan,
            &[4.0, 3.0, 2.0, 1.0],
            &[0.0; 4],
            &mut scratch,
            &mut indices,
            &mut weights,
        )?;
        assert_eq!(indices, [0, 1]);
        Ok(())
    }

    #[test]
    fn native_router_rejects_selection_larger_than_retained_groups() {
        let error = GlmRouterPlan::new(4, 3, 2, 1, 1.0).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
    }

    #[test]
    fn native_control_ops_reject_short_caller_scratch() -> Result<(), AmsError> {
        let plan = DsaTopKPlan::new(1, 1, 1, 0, 1)?;
        let mut selected = [usize::MAX; 1];
        let error = glm_dsa_topk(plan, &[1.0], &[1.0], &[1.0], &mut [], &mut selected).err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        let mut score = [0.0; 1];
        let error = glm_dsa_topk(plan, &[], &[1.0], &[1.0], &mut score, &mut selected).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        Ok(())
    }
}
