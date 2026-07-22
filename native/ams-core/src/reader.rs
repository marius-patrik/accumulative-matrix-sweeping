use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::{AmsError, ErrorCode};

/// Checked random-access storage used by native streaming primitives.
pub trait RangeReader {
    /// Declared immutable object length.
    fn len(&self) -> u64;

    /// Whether the declared object is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill the complete destination or return a typed error.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::IoFailure`](crate::ErrorCode::IoFailure) when the checked
    /// range cannot be read in full.
    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError>;
}

/// In-memory reader used by differential tests and embedded callers.
#[derive(Clone, Copy, Debug)]
pub struct SliceReader<'a> {
    bytes: &'a [u8],
}

impl<'a> SliceReader<'a> {
    /// Wrap an immutable byte slice.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
}

impl RangeReader for SliceReader<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
        let start = usize::try_from(offset)
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "reader offset exceeds usize"))?;
        let end = start
            .checked_add(destination.len())
            .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "reader range overflow"))?;
        let source = self
            .bytes
            .get(start..end)
            .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "reader range exceeds object"))?;
        destination.copy_from_slice(source);
        Ok(())
    }
}

/// Immutable positional reader for a regular local file.
#[derive(Debug)]
pub struct FileRangeReader {
    file: File,
    len: u64,
}

impl FileRangeReader {
    /// Open a nonsymlink regular file for positional reads.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::IoFailure`] when metadata or open operations fail, and
    /// [`ErrorCode::InvalidPackage`] when the path is a symlink or not a regular file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AmsError> {
        let path = path.as_ref();
        let link_metadata = path
            .symlink_metadata()
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "file metadata read failed"))?;
        if link_metadata.file_type().is_symlink() || !link_metadata.is_file() {
            return Err(AmsError::new(
                ErrorCode::InvalidPackage,
                "range reader path is not a nonsymlink regular file",
            ));
        }
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "file open failed"))?;
        let len = file
            .metadata()
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "open file metadata failed"))?
            .len();
        Ok(Self { file, len })
    }
}

impl RangeReader for FileRangeReader {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
        let destination_len = u64::try_from(destination.len())
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "read length exceeds u64"))?;
        let end = offset
            .checked_add(destination_len)
            .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "file read range overflow"))?;
        if end > self.len {
            return Err(AmsError::new(
                ErrorCode::IoFailure,
                "file read exceeds declared object",
            ));
        }
        let mut completed = 0usize;
        while completed < destination.len() {
            let completed_u64 = u64::try_from(completed)
                .map_err(|_| AmsError::new(ErrorCode::IoFailure, "read progress exceeds u64"))?;
            let position = offset
                .checked_add(completed_u64)
                .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "file position overflow"))?;
            #[cfg(windows)]
            let count = {
                use std::os::windows::fs::FileExt;
                self.file.seek_read(&mut destination[completed..], position)
            };
            #[cfg(unix)]
            let count = {
                use std::os::unix::fs::FileExt;
                self.file.read_at(&mut destination[completed..], position)
            };
            #[cfg(not(any(windows, unix)))]
            compile_error!("FileRangeReader requires Windows or Unix positional file I/O");
            let count = count
                .map_err(|_| AmsError::new(ErrorCode::IoFailure, "positional file read failed"))?;
            if count == 0 {
                return Err(AmsError::new(
                    ErrorCode::IoFailure,
                    "short positional file read",
                ));
            }
            completed = completed.checked_add(count).ok_or_else(|| {
                AmsError::new(ErrorCode::IoFailure, "file read progress overflow")
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

    fn temporary_path() -> std::path::PathBuf {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ams-core-{}-{id}.bin", std::process::id()))
    }

    #[test]
    fn file_reader_uses_exact_checked_positional_ranges() -> Result<(), AmsError> {
        let path = temporary_path();
        fs::write(&path, b"0123456789")
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "test file write failed"))?;
        let reader = FileRangeReader::open(&path)?;
        let mut destination = [0u8; 4];
        reader.read_exact_at(3, &mut destination)?;
        assert_eq!(&destination, b"3456");
        let error = reader.read_exact_at(8, &mut destination).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        fs::remove_file(path)
            .map_err(|_| AmsError::new(ErrorCode::IoFailure, "test file cleanup failed"))?;
        Ok(())
    }
}
