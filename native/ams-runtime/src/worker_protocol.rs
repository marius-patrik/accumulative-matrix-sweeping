//! Persistent, bounded, cancellable process protocol for one admitted GLM-4 binding.

use std::io::{self, BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use ams_core::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::{AdmittedGlm4Binding, Glm4AdmissionEvidence, Glm4GreedyOutput, RuntimeError};

const MAX_WORKER_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
enum WorkerInput {
    Generate {
        request_id: u64,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
    },
    Cancel {
        request_id: u64,
    },
    Shutdown,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerateInput {
    schema_id: String,
    request_id: u64,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CancelInput {
    schema_id: String,
    request_id: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShutdownInput {
    schema_id: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StrictWorkerInput {
    Generate(GenerateInput),
    Cancel(CancelInput),
    Shutdown(ShutdownInput),
}

enum WorkerCommand {
    Generate {
        request_id: u64,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
        cancellation: Arc<AtomicBool>,
    },
    Shutdown,
}

struct ActiveRequest {
    request_id: u64,
    cancellation: Arc<AtomicBool>,
}

#[derive(Default)]
struct RequestState {
    active: Option<ActiveRequest>,
    last_terminal_request_id: Option<u64>,
}

#[derive(Serialize)]
struct ReadyFrame<'a> {
    schema_id: &'static str,
    evidence: &'a Glm4AdmissionEvidence,
    context_capacity_tokens: usize,
    tokenizer_vocabulary_size: usize,
}

#[derive(Serialize)]
struct TokenFrame {
    schema_id: &'static str,
    request_id: u64,
    index: usize,
    token_id: usize,
}

#[derive(Serialize)]
struct CompletedFrame<'a> {
    schema_id: &'static str,
    request_id: u64,
    output: &'a Glm4GreedyOutput,
}

#[derive(Serialize)]
struct ErrorFrame<'a> {
    schema_id: &'static str,
    request_id: Option<u64>,
    code: &'static str,
    message: &'a str,
}

fn runtime_error(code: ErrorCode, message: &'static str) -> RuntimeError {
    RuntimeError::new(code, message)
}

fn lock<'a, T>(
    value: &'a Mutex<T>,
    message: &'static str,
) -> Result<MutexGuard<'a, T>, RuntimeError> {
    value
        .lock()
        .map_err(|_| runtime_error(ErrorCode::InternalInvariant, message))
}

fn write_frame(output: &Mutex<io::Stdout>, value: &impl Serialize) -> Result<(), RuntimeError> {
    let mut output = lock(output, "native worker output lock was poisoned")?;
    serde_json::to_writer(&mut *output, value).map_err(|_| {
        runtime_error(
            ErrorCode::IoFailure,
            "native worker could not serialize an output frame",
        )
    })?;
    output.write_all(b"\n").map_err(|_| {
        runtime_error(
            ErrorCode::IoFailure,
            "native worker could not terminate an output frame",
        )
    })?;
    output.flush().map_err(|_| {
        runtime_error(
            ErrorCode::IoFailure,
            "native worker could not flush an output frame",
        )
    })
}

fn write_error(
    output: &Mutex<io::Stdout>,
    request_id: Option<u64>,
    error: &RuntimeError,
) -> Result<(), RuntimeError> {
    write_frame(
        output,
        &ErrorFrame {
            schema_id: "ams.native.worker.error.v1",
            request_id,
            code: error.code().as_str(),
            message: error.message(),
        },
    )
}

fn drain_through_newline(reader: &mut impl BufRead) -> Result<(), RuntimeError> {
    loop {
        let available = reader.fill_buf().map_err(|_| {
            runtime_error(
                ErrorCode::IoFailure,
                "native worker could not read an input frame",
            )
        })?;
        if available.is_empty() {
            return Ok(());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |index| index + 1);
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(());
        }
    }
}

fn read_bounded_frame(
    reader: &mut impl BufRead,
    frame: &mut Vec<u8>,
) -> Result<bool, RuntimeError> {
    frame.clear();
    loop {
        let available = reader.fill_buf().map_err(|_| {
            runtime_error(
                ErrorCode::IoFailure,
                "native worker could not read an input frame",
            )
        })?;
        if available.is_empty() {
            if frame.is_empty() {
                return Ok(false);
            }
            break;
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let copied = newline.unwrap_or(available.len());
        let total = frame.len().checked_add(copied).ok_or_else(|| {
            runtime_error(
                ErrorCode::PlanInvalid,
                "native worker input frame length overflowed",
            )
        })?;
        if total > MAX_WORKER_FRAME_BYTES {
            let consumed = newline.map_or(available.len(), |index| index + 1);
            reader.consume(consumed);
            if newline.is_none() {
                drain_through_newline(reader)?;
            }
            return Err(runtime_error(
                ErrorCode::PlanInvalid,
                "native worker input frame exceeds the admitted byte bound",
            ));
        }
        frame.extend_from_slice(&available[..copied]);
        let consumed = newline.map_or(available.len(), |index| index + 1);
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }
    if frame.last() == Some(&b'\r') {
        frame.pop();
    }
    if frame.is_empty() {
        return Err(runtime_error(
            ErrorCode::InvalidPackage,
            "native worker input frame is empty",
        ));
    }
    Ok(true)
}

fn parse_input(frame: &[u8]) -> Result<WorkerInput, RuntimeError> {
    let input: StrictWorkerInput = serde_json::from_slice(frame).map_err(|_| {
        runtime_error(
            ErrorCode::InvalidPackage,
            "native worker input is malformed or contains unreviewed fields",
        )
    })?;
    match input {
        StrictWorkerInput::Generate(value)
            if value.schema_id == "ams.native.worker.generate.v1" =>
        {
            Ok(WorkerInput::Generate {
                request_id: value.request_id,
                prompt_token_ids: value.prompt_token_ids,
                max_new_tokens: value.max_new_tokens,
            })
        }
        StrictWorkerInput::Cancel(value) if value.schema_id == "ams.native.worker.cancel.v1" => {
            Ok(WorkerInput::Cancel {
                request_id: value.request_id,
            })
        }
        StrictWorkerInput::Shutdown(value)
            if value.schema_id == "ams.native.worker.shutdown.v1" =>
        {
            Ok(WorkerInput::Shutdown)
        }
        _ => Err(runtime_error(
            ErrorCode::CapabilityMismatch,
            "native worker input schema is unsupported",
        )),
    }
}

fn clear_active(
    active: &Mutex<RequestState>,
    request_id: u64,
) -> Result<MutexGuard<'_, RequestState>, RuntimeError> {
    let state = lock(active, "native worker request-state lock was poisoned")?;
    if state
        .active
        .as_ref()
        .is_none_or(|current| current.request_id != request_id)
    {
        return Err(runtime_error(
            ErrorCode::InternalInvariant,
            "native worker lost authoritative request ownership",
        ));
    }
    Ok(state)
}

fn request_shutdown(
    sender: &SyncSender<WorkerCommand>,
    active: &Mutex<RequestState>,
    shutting_down: &AtomicBool,
) -> Result<(), RuntimeError> {
    shutting_down.store(true, Ordering::Release);
    if let Some(current) = lock(active, "native worker request-state lock was poisoned")?
        .active
        .as_ref()
    {
        current.cancellation.store(true, Ordering::Release);
    }
    sender.send(WorkerCommand::Shutdown).map_err(|_| {
        runtime_error(
            ErrorCode::InternalInvariant,
            "native worker command loop ended before shutdown",
        )
    })
}

#[allow(clippy::too_many_lines)] // Keep the bounded input grammar and state transitions together.
fn read_commands(
    sender: &SyncSender<WorkerCommand>,
    output: &Arc<Mutex<io::Stdout>>,
    active: &Arc<Mutex<RequestState>>,
    shutting_down: &Arc<AtomicBool>,
) -> Result<(), RuntimeError> {
    let mut reader = BufReader::new(io::stdin());
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(MAX_WORKER_FRAME_BYTES)
        .map_err(|_| {
            runtime_error(
                ErrorCode::PreflightNoWorkingSet,
                "native worker input-frame allocation failed",
            )
        })?;
    loop {
        let has_frame = match read_bounded_frame(&mut reader, &mut frame) {
            Ok(value) => value,
            Err(error) => {
                write_error(output, None, &error)?;
                continue;
            }
        };
        if !has_frame {
            return request_shutdown(sender, active, shutting_down);
        }
        let input = match parse_input(&frame) {
            Ok(value) => value,
            Err(error) => {
                write_error(output, None, &error)?;
                continue;
            }
        };
        match input {
            WorkerInput::Generate {
                request_id,
                prompt_token_ids,
                max_new_tokens,
            } => {
                if shutting_down.load(Ordering::Acquire) {
                    write_error(
                        output,
                        Some(request_id),
                        &runtime_error(
                            ErrorCode::CapabilityMismatch,
                            "native worker is shutting down",
                        ),
                    )?;
                    continue;
                }
                let cancellation = Arc::new(AtomicBool::new(false));
                {
                    let mut state = lock(active, "native worker request-state lock was poisoned")?;
                    if state.active.is_some() {
                        write_error(
                            output,
                            Some(request_id),
                            &runtime_error(
                                ErrorCode::PreflightNoWorkingSet,
                                "native worker already has an active request",
                            ),
                        )?;
                        continue;
                    }
                    state.active = Some(ActiveRequest {
                        request_id,
                        cancellation: Arc::clone(&cancellation),
                    });
                }
                let command = WorkerCommand::Generate {
                    request_id,
                    prompt_token_ids,
                    max_new_tokens,
                    cancellation,
                };
                if let Err(send_error) = sender.try_send(command) {
                    let mut state = lock(active, "native worker request-state lock was poisoned")?;
                    state.active = None;
                    drop(state);
                    match send_error {
                        TrySendError::Full(_) => write_error(
                            output,
                            Some(request_id),
                            &runtime_error(
                                ErrorCode::InternalInvariant,
                                "native worker command queue violated single-request ordering",
                            ),
                        )?,
                        TrySendError::Disconnected(_) => {
                            return Err(runtime_error(
                                ErrorCode::InternalInvariant,
                                "native worker command loop disconnected",
                            ));
                        }
                    }
                }
            }
            WorkerInput::Cancel { request_id } => {
                let state = lock(active, "native worker request-state lock was poisoned")?;
                match state.active.as_ref() {
                    Some(current) if current.request_id == request_id => {
                        current.cancellation.store(true, Ordering::Release);
                    }
                    None if state.last_terminal_request_id == Some(request_id) => {}
                    _ => write_error(
                        output,
                        Some(request_id),
                        &runtime_error(
                            ErrorCode::PlanInvalid,
                            "native worker cancellation does not match the active request",
                        ),
                    )?,
                }
            }
            WorkerInput::Shutdown => {
                request_shutdown(sender, active, shutting_down)?;
                return Ok(());
            }
        }
    }
}

/// Keep one admitted binding alive while serving bounded JSON-line generation commands over stdio.
///
/// # Errors
///
/// Returns a typed admission-independent worker I/O, resource, protocol, cancellation, or execution
/// failure. Per-request failures are emitted as terminal frames and do not stop the worker.
pub fn run_glm4_worker_stdio(binding: &AdmittedGlm4Binding) -> Result<(), RuntimeError> {
    let output = Arc::new(Mutex::new(io::stdout()));
    write_frame(
        &output,
        &ReadyFrame {
            schema_id: "ams.native.worker.ready.v1",
            evidence: binding.evidence(),
            context_capacity_tokens: binding.context_capacity_tokens(),
            tokenizer_vocabulary_size: binding.tokenizer_vocabulary_size(),
        },
    )?;

    let active = Arc::new(Mutex::new(RequestState::default()));
    let shutting_down = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::sync_channel(1);
    let reader_output = Arc::clone(&output);
    let reader_active = Arc::clone(&active);
    let reader_shutdown = Arc::clone(&shutting_down);
    let reader = thread::Builder::new()
        .name("ams-native-worker-input".to_owned())
        .spawn(move || read_commands(&sender, &reader_output, &reader_active, &reader_shutdown))
        .map_err(|_| {
            runtime_error(
                ErrorCode::PreflightNoWorkingSet,
                "native worker input thread could not start",
            )
        })?;

    let mut fatal = None;
    while let Ok(command) = receiver.recv() {
        match command {
            WorkerCommand::Generate {
                request_id,
                prompt_token_ids,
                max_new_tokens,
                cancellation,
            } => {
                let result = binding.generate_greedy_with_control(
                    &prompt_token_ids,
                    max_new_tokens,
                    || cancellation.load(Ordering::Acquire),
                    |index, token_id| {
                        write_frame(
                            &output,
                            &TokenFrame {
                                schema_id: "ams.native.worker.token.v1",
                                request_id,
                                index,
                                token_id,
                            },
                        )
                    },
                );
                let mut state = clear_active(&active, request_id)?;
                let terminal = match &result {
                    Ok(generation) => write_frame(
                        &output,
                        &CompletedFrame {
                            schema_id: "ams.native.worker.completed.v1",
                            request_id,
                            output: generation,
                        },
                    ),
                    Err(error) => write_error(&output, Some(request_id), error),
                };
                state.active = None;
                state.last_terminal_request_id = Some(request_id);
                drop(state);
                if let Err(error) = terminal {
                    cancellation.store(true, Ordering::Release);
                    fatal = Some(error);
                    break;
                }
            }
            WorkerCommand::Shutdown => break,
        }
    }
    shutting_down.store(true, Ordering::Release);
    if let Some(current) = lock(&active, "native worker request-state lock was poisoned")?
        .active
        .as_ref()
    {
        current.cancellation.store(true, Ordering::Release);
    }
    if let Some(error) = fatal {
        return Err(error);
    }
    reader.join().unwrap_or_else(|_| {
        Err(runtime_error(
            ErrorCode::InternalInvariant,
            "native worker input thread panicked",
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor};

    use super::{MAX_WORKER_FRAME_BYTES, WorkerInput, parse_input, read_bounded_frame};

    #[test]
    fn bounded_frame_accepts_crlf_and_preserves_the_following_frame() {
        let payload = b"{\"schema_id\":\"ams.native.worker.cancel.v1\",\"request_id\":7}\r\nnext\n";
        let mut reader = BufReader::new(Cursor::new(payload));
        let mut frame = Vec::with_capacity(MAX_WORKER_FRAME_BYTES);
        assert!(read_bounded_frame(&mut reader, &mut frame).unwrap_or(false));
        assert!(matches!(
            parse_input(&frame),
            Ok(WorkerInput::Cancel { request_id: 7 })
        ));
        assert!(read_bounded_frame(&mut reader, &mut frame).unwrap_or(false));
        assert_eq!(frame, b"next");
    }

    #[test]
    fn bounded_frame_drains_an_oversized_line_before_retry() {
        let mut payload = vec![b'x'; MAX_WORKER_FRAME_BYTES + 1];
        payload.extend_from_slice(b"\n{\"schema_id\":\"ams.native.worker.shutdown.v1\"}\n");
        let mut reader = BufReader::new(Cursor::new(payload));
        let mut frame = Vec::with_capacity(MAX_WORKER_FRAME_BYTES);
        assert!(read_bounded_frame(&mut reader, &mut frame).is_err());
        assert!(read_bounded_frame(&mut reader, &mut frame).unwrap_or(false));
        assert!(matches!(parse_input(&frame), Ok(WorkerInput::Shutdown)));
    }

    #[test]
    fn protocol_rejects_unknown_fields() {
        let payload = br#"{"schema_id":"ams.native.worker.shutdown.v1","unreviewed":true}"#;
        assert!(parse_input(payload).is_err());
    }
}
