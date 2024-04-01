use core::ffi::c_char;
use core::fmt;

use crate::xr;

pub unsafe trait TaggedStructure {
    const STRUCTURE_TYPE: xr::StructureType;
}

#[inline]
pub(crate) fn wrap_c_str_slice_until_nul(
    str: &[c_char],
) -> Result<&core::ffi::CStr, core::ffi::FromBytesUntilNulError> {
    // SAFETY: The cast from c_char to u8 is ok because a c_char is always one byte.
    let bytes = unsafe { core::slice::from_raw_parts(str.as_ptr().cast(), str.len()) };
    core::ffi::CStr::from_bytes_until_nul(bytes)
}

#[derive(Debug)]
pub struct CStrTooLargeForStaticArray {
    pub static_array_size: usize,
    pub c_str_size: usize,
}
#[cfg(feature = "std")]
impl std::error::Error for CStrTooLargeForStaticArray {}
impl fmt::Display for CStrTooLargeForStaticArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "static `c_char` target array of length `{}` is too small to write a `CStr` (with `NUL`-terminator) of length `{}`",
            self.static_array_size, self.c_str_size
        )
    }
}

#[inline]
pub(crate) fn write_c_str_slice_with_nul(
    target: &mut [c_char],
    str: &core::ffi::CStr,
) -> Result<(), CStrTooLargeForStaticArray> {
    let bytes = str.to_bytes_with_nul();
    // SAFETY: The cast from c_char to u8 is ok because a c_char is always one byte.
    let bytes = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) };
    let static_array_size = target.len();
    target
        .get_mut(..bytes.len())
        .ok_or(CStrTooLargeForStaticArray {
            static_array_size,
            c_str_size: bytes.len(),
        })?
        .copy_from_slice(bytes);
    Ok(())
}
