use crate::common::to_rbool;
use mlir_capi::Support::*;
use std::cmp::{Eq, PartialEq};
use std::convert::{From, Into};
use std::default::Default;
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LlvmThreadPool {
    pub handle: MlirLlvmThreadPool,
}

impl From<MlirLlvmThreadPool> for LlvmThreadPool {
    fn from(value: MlirLlvmThreadPool) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirLlvmThreadPool> for &LlvmThreadPool {
    fn into(self) -> MlirLlvmThreadPool {
        self.handle
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TypeID {
    pub handle: MlirTypeID,
}

impl From<MlirTypeID> for TypeID {
    fn from(value: MlirTypeID) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirTypeID> for &TypeID {
    fn into(self) -> MlirTypeID {
        self.handle
    }
}

impl TypeID {
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
}

impl PartialEq<TypeID> for TypeID {
    fn eq(&self, other: &TypeID) -> bool {
        to_rbool(unsafe { mlir_capi::Support::FFIVal_::<u8>::mlirTypeIDEqual(self, other) })
    }
}
impl Eq for TypeID {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TypeIDAllocator {
    pub handle: MlirTypeIDAllocator,
}

impl From<MlirTypeIDAllocator> for TypeIDAllocator {
    fn from(value: MlirTypeIDAllocator) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirTypeIDAllocator> for &TypeIDAllocator {
    fn into(self) -> MlirTypeIDAllocator {
        self.handle
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct StrRef<'a> {
    pub handle: MlirStringRef,
    phantom: PhantomData<&'a str>,
}

impl<'a> From<MlirStringRef> for StrRef<'a> {
    fn from(value: MlirStringRef) -> Self {
        Self {
            handle: value,
            phantom: Default::default(),
        }
    }
}

impl<'a> Into<MlirStringRef> for StrRef<'a> {
    fn into(self) -> MlirStringRef {
        self.handle
    }
}

impl<'a> From<&'a str> for StrRef<'a> {
    fn from(s: &'a str) -> Self {
        let ptr = s.as_ptr();
        let len = s.len();
        Self {
            handle: MlirStringRef {
                data: ptr as _,
                length: len as _,
            },
            phantom: Default::default(),
        }
    }
}

impl<'a> Into<&'a str> for StrRef<'a> {
    fn into(self) -> &'a str {
        if self.handle.length == 0 {
            ""
        } else {
            unsafe {
                let slice =
                    std::slice::from_raw_parts(self.handle.data as _, self.handle.length as _);
                std::str::from_utf8_unchecked(slice)
            }
        }
    }
}

impl<'a, T: Sized + Copy> From<&'a [T]> for StrRef<'a> {
    fn from(s: &'a [T]) -> Self {
        let ptr = s.as_ptr();
        let len = s.len() * std::mem::size_of::<T>();
        Self {
            handle: MlirStringRef {
                data: ptr as _,
                length: len as _,
            },
            phantom: Default::default(),
        }
    }
}

impl<'a> Into<&'a [u8]> for StrRef<'a> {
    fn into(self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.handle.data as _, self.handle.length as _) }
    }
}

impl<'a, 'b> PartialEq<StrRef<'b>> for StrRef<'a> {
    fn eq(&self, other: &StrRef<'b>) -> bool {
        (unsafe { mlir_capi::Support::FFIVal_::<u8>::mlirStringRefEqual(*self, *other) }) != 0
    }
}
impl<'a> Eq for StrRef<'a> {}

impl<'a> StrRef<'a> {
    pub fn to_ffi(self) -> MlirStringRef {
        self.handle
    }
    pub fn from_ffi(handle: MlirStringRef) -> Self {
        Self {
            handle: handle,
            phantom: PhantomData::default(),
        }
    }
    pub fn from_str(input: &'a str) -> Self {
        let handle = MlirStringRef {
            data: input.as_ptr() as _,
            length: input.len() as _,
        };
        StrRef {
            handle,
            phantom: PhantomData,
        }
    }
    pub fn to_str(self) -> &'a str {
        unsafe {
            let data = self.handle.data as *const u8;
            let length = self.handle.length as usize;
            let slice = std::slice::from_raw_parts(data, length);
            std::str::from_utf8_unchecked(slice)
        }
    }
}

pub trait PrintCallback {
    fn print(&mut self, s: &str);
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LogicalResult {
    pub handle: MlirLogicalResult,
}

impl LogicalResult {
    pub fn is_success(self) -> bool {
        self.handle.value != 0
    }
    pub fn is_failure(self) -> bool {
        !self.is_success()
    }
    pub fn success() -> Self {
        Self {
            handle: MlirLogicalResult { value: 1 },
        }
    }
    pub fn failure() -> Self {
        Self {
            handle: MlirLogicalResult { value: 0 },
        }
    }
}

impl From<MlirLogicalResult> for LogicalResult {
    fn from(value: MlirLogicalResult) -> Self {
        LogicalResult { handle: value }
    }
}

impl Into<MlirLogicalResult> for &LogicalResult {
    fn into(self) -> MlirLogicalResult {
        self.handle
    }
}

impl PartialEq<LogicalResult> for LogicalResult {
    fn eq(&self, other: &LogicalResult) -> bool {
        if self.is_success() {
            other.is_success()
        } else {
            other.is_failure()
        }
    }
}
impl Eq for LogicalResult {}
