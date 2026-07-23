//! Command-line boundary for native AMS package admission.

use std::env;
use std::process::ExitCode;

use ams_runtime::{RuntimeError, admit_glm4_binding_file};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorOutput<'a> {
    schema_id: &'static str,
    code: &'static str,
    message: &'a str,
}

fn run() -> Result<(), RuntimeError> {
    let mut arguments = env::args_os().skip(1);
    let command = arguments.next().ok_or_else(|| {
        RuntimeError::from(ams_core::AmsError::new(
            ams_core::ErrorCode::PlanInvalid,
            "missing native runtime command",
        ))
    })?;
    if command != "inspect" {
        return Err(RuntimeError::from(ams_core::AmsError::new(
            ams_core::ErrorCode::CapabilityMismatch,
            "native runtime command is unsupported",
        )));
    }
    let path = arguments.next().ok_or_else(|| {
        RuntimeError::from(ams_core::AmsError::new(
            ams_core::ErrorCode::PlanInvalid,
            "missing binding envelope path",
        ))
    })?;
    let buffer_bytes = match arguments.next() {
        Some(value) => value.to_string_lossy().parse::<usize>().map_err(|_| {
            RuntimeError::from(ams_core::AmsError::new(
                ams_core::ErrorCode::PlanInvalid,
                "verification buffer is not an integer",
            ))
        })?,
        None => 1024 * 1024,
    };
    if arguments.next().is_some() {
        return Err(RuntimeError::from(ams_core::AmsError::new(
            ams_core::ErrorCode::PlanInvalid,
            "native runtime received unexpected arguments",
        )));
    }
    let admitted = admit_glm4_binding_file(path, buffer_bytes)?;
    println!(
        "{}",
        serde_json::to_string(admitted.evidence()).map_err(|_| {
            RuntimeError::from(ams_core::AmsError::new(
                ams_core::ErrorCode::InternalInvariant,
                "admission evidence serialization failed",
            ))
        })?
    );
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
