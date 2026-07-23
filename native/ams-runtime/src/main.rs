//! Command-line boundary for native AMS package admission.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use ams_runtime::{RuntimeError, admit_glm4_binding_file, run_glm4_worker_stdio};
use serde::{Deserialize, Serialize};

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;

#[derive(Serialize)]
struct ErrorOutput<'a> {
    schema_id: &'static str,
    code: &'static str,
    message: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GreedyRequest {
    schema_id: String,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationRequest {
    schema_id: String,
    input_token_ids: Vec<usize>,
}

fn static_error(code: ams_core::ErrorCode, context: &'static str) -> RuntimeError {
    RuntimeError::from(ams_core::AmsError::new(code, context))
}

fn parse_buffer(value: Option<OsString>) -> Result<usize, RuntimeError> {
    value.map_or(Ok(1024 * 1024), |value| {
        value.to_string_lossy().parse::<usize>().map_err(|_| {
            static_error(
                ams_core::ErrorCode::PlanInvalid,
                "verification buffer is not an integer",
            )
        })
    })
}

fn serialize_output(value: &impl Serialize) -> Result<String, RuntimeError> {
    serde_json::to_string(value).map_err(|_| {
        static_error(
            ams_core::ErrorCode::InternalInvariant,
            "native output serialization failed",
        )
    })
}

fn read_greedy_request(path: impl AsRef<Path>) -> Result<GreedyRequest, RuntimeError> {
    let path = path.as_ref();
    let metadata = path.symlink_metadata().map_err(|_| {
        static_error(
            ams_core::ErrorCode::IoFailure,
            "greedy request metadata read failed",
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(static_error(
            ams_core::ErrorCode::InvalidPackage,
            "greedy request is not a nonsymlink regular file",
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_REQUEST_BYTES {
        return Err(static_error(
            ams_core::ErrorCode::PlanInvalid,
            "greedy request size is outside the admitted bound",
        ));
    }
    let payload = fs::read(path)
        .map_err(|_| static_error(ams_core::ErrorCode::IoFailure, "greedy request read failed"))?;
    let request: GreedyRequest = serde_json::from_slice(&payload).map_err(|_| {
        static_error(
            ams_core::ErrorCode::InvalidPackage,
            "greedy request JSON is malformed or contains unreviewed fields",
        )
    })?;
    if request.schema_id != "ams.native.glm4-greedy-request.v1" {
        return Err(static_error(
            ams_core::ErrorCode::CapabilityMismatch,
            "greedy request schema is unsupported",
        ));
    }
    Ok(request)
}

fn read_observation_request(path: impl AsRef<Path>) -> Result<ObservationRequest, RuntimeError> {
    let path = path.as_ref();
    let metadata = path.symlink_metadata().map_err(|_| {
        static_error(
            ams_core::ErrorCode::IoFailure,
            "observation request metadata read failed",
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(static_error(
            ams_core::ErrorCode::InvalidPackage,
            "observation request is not a nonsymlink regular file",
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_REQUEST_BYTES {
        return Err(static_error(
            ams_core::ErrorCode::PlanInvalid,
            "observation request size is outside the admitted bound",
        ));
    }
    let payload = fs::read(path).map_err(|_| {
        static_error(
            ams_core::ErrorCode::IoFailure,
            "observation request read failed",
        )
    })?;
    let request: ObservationRequest = serde_json::from_slice(&payload).map_err(|_| {
        static_error(
            ams_core::ErrorCode::InvalidPackage,
            "observation request JSON is malformed or contains unreviewed fields",
        )
    })?;
    if request.schema_id != "ams.native.glm4-observation-request.v1" {
        return Err(static_error(
            ams_core::ErrorCode::CapabilityMismatch,
            "observation request schema is unsupported",
        ));
    }
    Ok(request)
}

fn run() -> Result<(), RuntimeError> {
    let mut arguments = env::args_os().skip(1);
    let command = arguments.next().ok_or_else(|| {
        static_error(
            ams_core::ErrorCode::PlanInvalid,
            "missing native runtime command",
        )
    })?;
    let path = arguments.next().ok_or_else(|| {
        static_error(
            ams_core::ErrorCode::PlanInvalid,
            "missing binding envelope path",
        )
    })?;
    match command.to_string_lossy().as_ref() {
        "inspect" => {
            let buffer_bytes = parse_buffer(arguments.next())?;
            if arguments.next().is_some() {
                return Err(static_error(
                    ams_core::ErrorCode::PlanInvalid,
                    "native inspect received unexpected arguments",
                ));
            }
            let admitted = admit_glm4_binding_file(path, buffer_bytes)?;
            println!("{}", serialize_output(admitted.evidence())?);
        }
        "generate" => {
            let request_path = arguments.next().ok_or_else(|| {
                static_error(
                    ams_core::ErrorCode::PlanInvalid,
                    "missing greedy request path",
                )
            })?;
            let buffer_bytes = parse_buffer(arguments.next())?;
            if arguments.next().is_some() {
                return Err(static_error(
                    ams_core::ErrorCode::PlanInvalid,
                    "native generate received unexpected arguments",
                ));
            }
            let request = read_greedy_request(request_path)?;
            let admitted = admit_glm4_binding_file(path, buffer_bytes)?;
            let output =
                admitted.generate_greedy(&request.prompt_token_ids, request.max_new_tokens)?;
            println!("{}", serialize_output(&output)?);
        }
        "observe" => {
            let request_path = arguments.next().ok_or_else(|| {
                static_error(
                    ams_core::ErrorCode::PlanInvalid,
                    "missing observation request path",
                )
            })?;
            let buffer_bytes = parse_buffer(arguments.next())?;
            if arguments.next().is_some() {
                return Err(static_error(
                    ams_core::ErrorCode::PlanInvalid,
                    "native observe received unexpected arguments",
                ));
            }
            let request = read_observation_request(request_path)?;
            let admitted = admit_glm4_binding_file(path, buffer_bytes)?;
            let output = admitted.observe_tokens(&request.input_token_ids)?;
            println!("{}", serialize_output(&output)?);
        }
        "worker" => {
            let buffer_bytes = parse_buffer(arguments.next())?;
            if arguments.next().is_some() {
                return Err(static_error(
                    ams_core::ErrorCode::PlanInvalid,
                    "native worker received unexpected arguments",
                ));
            }
            let admitted = admit_glm4_binding_file(path, buffer_bytes)?;
            run_glm4_worker_stdio(&admitted)?;
        }
        _ => {
            return Err(static_error(
                ams_core::ErrorCode::CapabilityMismatch,
                "native runtime command is unsupported",
            ));
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let output = ErrorOutput {
                schema_id: "ams.native.error.v1",
                code: error.code().as_str(),
                message: error.message(),
            };
            let serialized = serde_json::to_string(&output).unwrap_or_else(|_| {
                "{\"schema_id\":\"ams.native.error.v1\",\"code\":\"INTERNAL_INVARIANT\",\
                 \"message\":\"error serialization failed\"}"
                    .to_owned()
            });
            eprintln!("{serialized}");
            ExitCode::FAILURE
        }
    }
}
