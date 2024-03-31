use std::{
    fmt,
    mem,
    num::NonZeroI32,
};
use crate::xr;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Error(pub NonZeroI32);

impl Error {
    #[inline(always)]
    pub const unsafe fn new_unchecked(result: xr::Result) -> Self {
        Error(NonZeroI32::new_unchecked(result as _))
    }
}

impl Into<xr::Result> for Error {
    #[inline(always)]
    fn into(self) -> xr::Result {
        unsafe { std::mem::transmute(self.0) }
    }
}

pub type XrResult<T> = Result<T, Error>;

impl xr::Result {
    #[inline]
    pub fn result(self) -> XrResult<()> {
        self.result_with_success(())
    }

    #[inline]
    pub fn result_with_success<T>(self, v: T) -> XrResult<T> {
        NonZeroI32::new(self).map_or(
            Ok(v),
            |e| Err(Error(e))
        )
    }

    #[inline]
    pub fn assume_init_on_success<T>(self, v: mem::MaybeUninit<T>) -> XrResult<T> {
        self.result().map(move |()| v.assume_init())
    }

    #[inline]
    pub unsafe fn set_vec_len_on_success<T>(self, mut v: Vec<T>, len: usize) -> XrResult<Vec<T>> {
        self.result().map(move |()| {
            v.set_len(len);
            v
        })
    }
}
