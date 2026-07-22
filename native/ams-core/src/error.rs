use core::fmt;

/// Stable native error codes aligned with the AMS control-plane taxonomy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorCode {
    /// No admissible durable backing route exists.
    PreflightNoBacking,
    /// The admitted arena cannot hold the minimum legal primitive working set.
    PreflightNoWorkingSet,
    /// The graph requires an operator that this runtime does not implement.
    UnsupportedOp,
    /// A serialized package or codec record is malformed.
    InvalidPackage,
    /// A content hash, size, or other integrity assertion failed.
    IntegrityFailure,
    /// A required package or plugin signature did not validate.
    SignatureFailure,
    /// Runtime capabilities do not satisfy a required feature.
    CapabilityMismatch,
    /// A plan or caller-provided descriptor is inconsistent.
    PlanInvalid,
    /// A previously granted resource reservation is no longer valid.
    ReservationLost,
    /// Runtime resource accounting or ownership was contradicted.
    BrokerViolation,
    /// A checked storage operation failed or exceeded the declared object.
    IoFailure,
    /// A compute backend failed outside a more specific category.
    BackendFailure,
    /// Encoded or computed numeric data is non-finite or otherwise invalid.
    NumericFailure,
    /// A journaled or atomic state transition failed.
    TransactionFailure,
    /// Work could not complete within its admitted deadline.
    DeadlineExceeded,
    /// Work stopped because its cancellation token was observed.
    Cancelled,
    /// An internal invariant was contradicted after validation.
    InternalInvariant,
}

impl ErrorCode {
    /// Return the cross-language stable code spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreflightNoBacking => "PREFLIGHT_NO_BACKING",
            Self::PreflightNoWorkingSet => "PREFLIGHT_NO_WORKING_SET",
            Self::UnsupportedOp => "UNSUPPORTED_OP",
            Self::InvalidPackage => "INVALID_PACKAGE",
            Self::IntegrityFailure => "INTEGRITY_FAILURE",
            Self::SignatureFailure => "SIGNATURE_FAILURE",
            Self::CapabilityMismatch => "CAPABILITY_MISMATCH",
            Self::PlanInvalid => "PLAN_INVALID",
            Self::ReservationLost => "RESERVATION_LOST",
            Self::BrokerViolation => "BROKER_VIOLATION",
            Self::IoFailure => "IO_FAILURE",
            Self::BackendFailure => "BACKEND_FAILURE",
            Self::NumericFailure => "NUMERIC_FAILURE",
            Self::TransactionFailure => "TRANSACTION_FAILURE",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::Cancelled => "CANCELLED",
            Self::InternalInvariant => "INTERNAL_INVARIANT",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Allocation-free native error payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmsError {
    code: ErrorCode,
    context: &'static str,
}

impl AmsError {
    /// Construct an error with a static, redaction-safe context string.
    #[must_use]
    pub const fn new(code: ErrorCode, context: &'static str) -> Self {
        Self { code, context }
    }

    /// Return the stable machine-readable code.
    #[must_use]
    pub const fn code(self) -> ErrorCode {
        self.code
    }

    /// Return the static redaction-safe context.
    #[must_use]
    pub const fn context(self) -> &'static str {
        self.context
    }
}

impl fmt::Display for AmsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.context)
    }
}

impl std::error::Error for AmsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_use_the_normative_cross_language_spelling() {
        let error = AmsError::new(ErrorCode::PreflightNoWorkingSet, "test context");
        assert_eq!(error.code().as_str(), "PREFLIGHT_NO_WORKING_SET");
        assert_eq!(error.to_string(), "PREFLIGHT_NO_WORKING_SET: test context");
    }
}
