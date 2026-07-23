use crate::{
    AmsError, ErrorCode, Glm4ModelPlan, Glm4ModelReaders, Glm4ModelScratch, KvCache,
    glm4_model_cache_token, glm4_model_next_token,
};

/// Terminal reason for a deterministic GLM-4 generation session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm4FinishReason {
    /// The selected token matched one of the admitted stop-token IDs.
    EndOfSequence,
    /// The request emitted its admitted maximum number of new tokens.
    Length,
}

/// One observable, ordered transition from a GLM-4 generation session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm4GenerationStep {
    /// One prompt token was cached; no discarded prompt logit was evaluated.
    Prefill {
        /// Prompt tokens now committed to every layer cache.
        consumed_tokens: usize,
        /// Immutable prompt length owned by the session.
        total_tokens: usize,
    },
    /// One non-terminal token was selected and is ready to stream.
    Token {
        /// Tokenizer-mapped selected token ID.
        token_id: usize,
    },
    /// Generation reached a terminal condition, optionally while emitting its final token.
    Finished {
        /// Final emitted token for a length stop, or `None` for an un-emitted EOS token.
        token_id: Option<usize>,
        /// Authoritative terminal reason.
        reason: Glm4FinishReason,
    },
}

/// Allocation-free, retryable state for one bounded greedy generation.
///
/// The prompt and EOS slices are borrowed for the complete session, so callers cannot resupply a
/// different prompt or stop policy while caches are live.
#[derive(Debug)]
pub struct Glm4GreedySession<'a> {
    prompt: &'a [usize],
    eos_token_ids: &'a [usize],
    tokenizer_vocabulary_size: usize,
    context_capacity_tokens: usize,
    max_new_tokens: usize,
    prompt_cursor: usize,
    position: usize,
    generated_tokens: usize,
    pending_input: Option<usize>,
    finished: Option<Glm4FinishReason>,
}

impl<'a> Glm4GreedySession<'a> {
    /// Admit an immutable prompt, stop set, and output bound against one model plan.
    ///
    /// The cache requirement includes every prompt token and all generated tokens except the final
    /// sampled token, which is emitted but never consumed after a length stop.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for malformed token policy and `PREFLIGHT_NO_WORKING_SET` when the
    /// complete worst-case session cannot fit the fixed cache capacity.
    pub fn new(
        plan: &Glm4ModelPlan,
        prompt: &'a [usize],
        eos_token_ids: &'a [usize],
        max_new_tokens: usize,
    ) -> Result<Self, AmsError> {
        Self::new_with_limits(
            plan.tokenizer_vocabulary_size(),
            plan.context_capacity_tokens(),
            prompt,
            eos_token_ids,
            max_new_tokens,
        )
    }

    fn new_with_limits(
        tokenizer_vocabulary_size: usize,
        context_capacity_tokens: usize,
        prompt: &'a [usize],
        eos_token_ids: &'a [usize],
        max_new_tokens: usize,
    ) -> Result<Self, AmsError> {
        if tokenizer_vocabulary_size == 0
            || context_capacity_tokens == 0
            || prompt.is_empty()
            || eos_token_ids.is_empty()
            || max_new_tokens == 0
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 generation dimensions and token policy must be nonzero",
            ));
        }
        if prompt
            .iter()
            .chain(eos_token_ids)
            .any(|token| *token >= tokenizer_vocabulary_size)
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 generation token is outside the tokenizer vocabulary",
            ));
        }
        if eos_token_ids
            .iter()
            .enumerate()
            .any(|(index, token)| eos_token_ids[..index].contains(token))
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 generation stop-token IDs must be unique",
            ));
        }
        let generated_inputs = max_new_tokens.checked_sub(1).ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "GLM-4 generation output bound underflowed",
            )
        })?;
        let required_cache_tokens =
            prompt.len().checked_add(generated_inputs).ok_or_else(|| {
                AmsError::new(
                    ErrorCode::PlanInvalid,
                    "GLM-4 generation cache requirement overflowed",
                )
            })?;
        if required_cache_tokens > context_capacity_tokens {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "GLM-4 generation exceeds the fixed cache capacity",
            ));
        }
        Ok(Self {
            prompt,
            eos_token_ids,
            tokenizer_vocabulary_size,
            context_capacity_tokens,
            max_new_tokens,
            prompt_cursor: 0,
            position: 0,
            generated_tokens: 0,
            pending_input: None,
            finished: None,
        })
    }

    /// Number of input tokens committed to every layer cache.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Number of immutable prompt tokens already committed.
    #[must_use]
    pub const fn prompt_consumed(&self) -> usize {
        self.prompt_cursor
    }

    /// Number of non-EOS output tokens already exposed to the caller.
    #[must_use]
    pub const fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    /// Most recently exposed token waiting to become the next model input.
    #[must_use]
    pub const fn pending_input(&self) -> Option<usize> {
        self.pending_input
    }

    /// Terminal state, if generation has ended.
    #[must_use]
    pub const fn finish_reason(&self) -> Option<Glm4FinishReason> {
        self.finished
    }

    const fn agrees_with(&self, plan: &Glm4ModelPlan) -> bool {
        self.tokenizer_vocabulary_size == plan.tokenizer_vocabulary_size()
            && self.context_capacity_tokens == plan.context_capacity_tokens()
    }
}

fn advance_with(
    session: &mut Glm4GreedySession<'_>,
    cancelled: bool,
    mut execute: impl FnMut(usize, usize, bool) -> Result<Option<usize>, AmsError>,
) -> Result<Glm4GenerationStep, AmsError> {
    if session.finished.is_some() {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 generation cannot advance after completion",
        ));
    }
    if cancelled {
        return Err(AmsError::new(
            ErrorCode::Cancelled,
            "GLM-4 generation observed cancellation at a token boundary",
        ));
    }

    let in_prompt = session.prompt_cursor < session.prompt.len();
    let selects_next = !in_prompt || session.prompt_cursor + 1 == session.prompt.len();
    let input_token = if in_prompt {
        session.prompt[session.prompt_cursor]
    } else {
        session.pending_input.ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "GLM-4 generation has no pending decode input",
            )
        })?
    };
    let next_position = session.position.checked_add(1).ok_or_else(|| {
        AmsError::new(
            ErrorCode::InternalInvariant,
            "GLM-4 generation position overflowed",
        )
    })?;
    let next_prompt_cursor = if in_prompt {
        Some(session.prompt_cursor.checked_add(1).ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "GLM-4 generation prompt cursor overflowed",
            )
        })?)
    } else {
        None
    };
    let next_generated_tokens = if selects_next {
        Some(session.generated_tokens.checked_add(1).ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "GLM-4 generated token count overflowed",
            )
        })?)
    } else {
        None
    };

    let selected = execute(session.position, input_token, selects_next)?;
    if selected.is_some() != selects_next {
        return Err(AmsError::new(
            ErrorCode::InternalInvariant,
            "GLM-4 model selection disagrees with the session phase",
        ));
    }

    session.position = next_position;
    if let Some(cursor) = next_prompt_cursor {
        session.prompt_cursor = cursor;
    }
    let Some(token_id) = selected else {
        return Ok(Glm4GenerationStep::Prefill {
            consumed_tokens: session.prompt_cursor,
            total_tokens: session.prompt.len(),
        });
    };
    if token_id >= session.tokenizer_vocabulary_size {
        return Err(AmsError::new(
            ErrorCode::InternalInvariant,
            "GLM-4 model selected an unmapped token",
        ));
    }
    if session.eos_token_ids.contains(&token_id) {
        session.pending_input = None;
        session.finished = Some(Glm4FinishReason::EndOfSequence);
        return Ok(Glm4GenerationStep::Finished {
            token_id: None,
            reason: Glm4FinishReason::EndOfSequence,
        });
    }

    let generated_tokens = next_generated_tokens.ok_or_else(|| {
        AmsError::new(
            ErrorCode::InternalInvariant,
            "GLM-4 selected a token outside a selection phase",
        )
    })?;
    session.generated_tokens = generated_tokens;
    session.pending_input = Some(token_id);
    if generated_tokens == session.max_new_tokens {
        session.finished = Some(Glm4FinishReason::Length);
        return Ok(Glm4GenerationStep::Finished {
            token_id: Some(token_id),
            reason: Glm4FinishReason::Length,
        });
    }
    Ok(Glm4GenerationStep::Token { token_id })
}

/// Advance one prompt or decode token while keeping session and cache prefixes in agreement.
///
/// Every cache prefix is checked against the session before cancellation or model execution. A model
/// failure leaves the session unchanged; the nested model/layer transactions leave cache prefixes
/// unchanged, so the same call is retryable. Cancellation is observed at token boundaries. Native
/// operator loops must add cooperative polling before sub-token cancellation latency can be claimed.
///
/// # Errors
///
/// Returns a typed cancellation, plan, capacity, storage, codec, or numeric error.
pub fn glm4_greedy_advance(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    session: &mut Glm4GreedySession<'_>,
    scratch: &mut Glm4ModelScratch<'_>,
    cancelled: bool,
) -> Result<Glm4GenerationStep, AmsError> {
    if !session.agrees_with(plan)
        || caches.len() != plan.decoder().layer_count()
        || caches
            .iter()
            .any(|cache| cache.committed_tokens() != session.position)
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 generation session, plan, and cache prefixes disagree",
        ));
    }
    advance_with(session, cancelled, |position, input_token, selects_next| {
        if selects_next {
            glm4_model_next_token(plan, readers, caches, position, input_token, scratch).map(Some)
        } else {
            glm4_model_cache_token(plan, readers, caches, position, input_token, scratch)?;
            Ok(None)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session<'a>(
        prompt: &'a [usize],
        eos: &'a [usize],
        max_new_tokens: usize,
    ) -> Result<Glm4GreedySession<'a>, AmsError> {
        Glm4GreedySession::new_with_limits(8, 6, prompt, eos, max_new_tokens)
    }

    #[test]
    fn session_preflights_complete_worst_case_capacity_and_token_policy() {
        assert_eq!(
            session(&[1, 2, 3], &[7], 5).err().map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        assert_eq!(
            session(&[1, 8], &[7], 1).err().map(AmsError::code),
            Some(ErrorCode::PlanInvalid)
        );
        assert_eq!(
            session(&[1], &[7, 7], 1).err().map(AmsError::code),
            Some(ErrorCode::PlanInvalid)
        );
        assert_eq!(
            session(&[], &[7], 1).err().map(AmsError::code),
            Some(ErrorCode::PlanInvalid)
        );
    }

    #[test]
    fn session_prefills_without_selection_then_streams_to_length() -> Result<(), AmsError> {
        let mut session = session(&[1, 2], &[7], 2)?;
        let mut actions = Vec::new();
        let first = advance_with(&mut session, false, |position, token, select| {
            actions.push((position, token, select));
            Ok(None)
        })?;
        assert_eq!(
            first,
            Glm4GenerationStep::Prefill {
                consumed_tokens: 1,
                total_tokens: 2
            }
        );
        let second = advance_with(&mut session, false, |position, token, select| {
            actions.push((position, token, select));
            Ok(Some(3))
        })?;
        assert_eq!(second, Glm4GenerationStep::Token { token_id: 3 });
        let third = advance_with(&mut session, false, |position, token, select| {
            actions.push((position, token, select));
            Ok(Some(4))
        })?;
        assert_eq!(
            third,
            Glm4GenerationStep::Finished {
                token_id: Some(4),
                reason: Glm4FinishReason::Length
            }
        );
        assert_eq!(actions, [(0, 1, false), (1, 2, true), (2, 3, true)]);
        assert_eq!(session.position(), 3);
        assert_eq!(session.prompt_consumed(), 2);
        assert_eq!(session.generated_tokens(), 2);
        assert_eq!(session.pending_input(), Some(4));
        assert_eq!(session.finish_reason(), Some(Glm4FinishReason::Length));
        assert_eq!(
            advance_with(&mut session, false, |_, _, _| Ok(Some(0)))
                .err()
                .map(AmsError::code),
            Some(ErrorCode::PlanInvalid)
        );
        Ok(())
    }

    #[test]
    fn session_does_not_emit_or_cache_a_selected_eos() -> Result<(), AmsError> {
        let mut session = session(&[1], &[7], 4)?;
        let step = advance_with(&mut session, false, |position, token, select| {
            assert_eq!((position, token, select), (0, 1, true));
            Ok(Some(7))
        })?;
        assert_eq!(
            step,
            Glm4GenerationStep::Finished {
                token_id: None,
                reason: Glm4FinishReason::EndOfSequence
            }
        );
        assert_eq!(session.position(), 1);
        assert_eq!(session.generated_tokens(), 0);
        assert_eq!(session.pending_input(), None);
        Ok(())
    }

    #[test]
    fn cancellation_and_failure_leave_the_authoritative_state_retryable() -> Result<(), AmsError> {
        let mut session = session(&[1, 2], &[7], 1)?;
        let mut calls = 0;
        let cancelled = advance_with(&mut session, true, |_, _, _| {
            calls += 1;
            Ok(None)
        })
        .err();
        assert_eq!(cancelled.map(AmsError::code), Some(ErrorCode::Cancelled));
        assert_eq!(calls, 0);
        assert_eq!(session.position(), 0);

        let failed = advance_with(&mut session, false, |_, _, _| {
            calls += 1;
            Err(AmsError::new(
                ErrorCode::BackendFailure,
                "injected model failure",
            ))
        })
        .err();
        assert_eq!(failed.map(AmsError::code), Some(ErrorCode::BackendFailure));
        assert_eq!(session.position(), 0);
        assert_eq!(session.prompt_consumed(), 0);

        let retried = advance_with(&mut session, false, |_, _, _| {
            calls += 1;
            Ok(None)
        })?;
        assert!(matches!(retried, Glm4GenerationStep::Prefill { .. }));
        assert_eq!(session.position(), 1);
        assert_eq!(session.prompt_consumed(), 1);
        assert_eq!(calls, 2);
        Ok(())
    }
}
