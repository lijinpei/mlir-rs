use crate::context::*;
use crate::location::*;
use crate::region::*;
use mlir_capi;
use mlir_capi::IR::*;
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Operation<'ctx> {
    pub handle: MlirOperation,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Operation<'ctx> {
    pub fn get_location(self) -> Location<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirOperationGetLocation(&self) }
    }
    pub fn get_first_region(self) -> Region<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirOperationGetFirstRegion(&self) }
    }
}

impl<'ctx> From<MlirOperation> for Operation<'ctx> {
    fn from(value: MlirOperation) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirOperation> for &Operation<'ctx> {
    fn into(self) -> MlirOperation {
        self.handle
    }
}
