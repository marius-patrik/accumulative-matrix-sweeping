use crate::{AmsError, ErrorCode};

pub fn add(left: usize, right: usize, field: &'static str) -> Result<usize, AmsError> {
    left.checked_add(right)
        .ok_or_else(|| AmsError::new(ErrorCode::PlanInvalid, field))
}

pub fn mul(left: usize, right: usize, field: &'static str) -> Result<usize, AmsError> {
    left.checked_mul(right)
        .ok_or_else(|| AmsError::new(ErrorCode::PlanInvalid, field))
}

pub fn add_u64(left: u64, right: u64, field: &'static str) -> Result<u64, AmsError> {
    left.checked_add(right)
        .ok_or_else(|| AmsError::new(ErrorCode::PlanInvalid, field))
}

pub fn usize_to_u64(value: usize, field: &'static str) -> Result<u64, AmsError> {
    u64::try_from(value).map_err(|_| AmsError::new(ErrorCode::PlanInvalid, field))
}
