use crate::context::*;
use crate::location::*;
use crate::operation::*;
use crate::r#type::*;
use crate::region::*;
use crate::value::*;
use mlir_capi::IR::*;
use std::cmp::{Eq, PartialEq};
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Block<'ctx> {
    pub handle: MlirBlock,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> From<MlirBlock> for Block<'ctx> {
    fn from(value: MlirBlock) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirBlock> for &Block<'ctx> {
    fn into(self) -> MlirBlock {
        self.handle
    }
}

impl<'ctx> Block<'ctx> {
    // FIXME: what about block of no args?
    pub fn create(types: &[Type], locs: &[Location]) -> Self {
        unsafe {
            mlir_capi::IR::FFIVal_::mlirBlockCreate(
                types.len() as i64,
                types.as_ptr() as *const _,
                locs.as_ptr() as *const _,
            )
        }
    }
    pub fn detach(self) {}
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null_mut()
    }
    pub fn get_parent_operation(self) -> Operation<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockGetParentOperation(&self) }
    }
    pub fn get_parent_region(self) -> Region<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockGetParentRegion(&self) }
    }
    pub fn get_next_in_region(self) -> Block<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockGetNextInRegion(&self) }
    }
    pub fn get_first_operation(self) -> Operation<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockGetFirstOperation(&self) }
    }
    pub fn get_terminator(self) -> Operation<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockGetTerminator(&self) }
    }
    pub fn append_owned_operation(self, op: Operation<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirBlockAppendOwnedOperation(&self, &op);
        }
    }
    pub fn insert_owned_operation(self, pos: usize, op: Operation<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirBlockInsertOwnedOperation(&self, pos as i64, &op);
        }
    }
    pub fn insert_owned_operation_after(self, reference: Operation<'ctx>, op: Operation<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirBlockInsertOwnedOperationAfter(&self, &reference, &op);
        }
    }
    pub fn insert_owned_operation_before(self, reference: Operation<'ctx>, op: Operation<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirBlockInsertOwnedOperationBefore(&self, &reference, &op);
        }
    }
    pub fn get_num_arguments(self) -> usize {
        (unsafe { mlir_capi::IR::FFIVal_::<i64>::mlirBlockGetNumArguments(&self) }) as usize
    }
    pub fn add_argument(self, arg_type: Type<'ctx>, loc: Location<'ctx>) -> Value<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockAddArgument(&self, arg_type, &loc) }
    }
    pub fn erase_argument(self, pos: usize) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirBlockEraseArgument(&self, pos as u32);
        }
    }
    pub fn insert_argument(
        self,
        pos: usize,
        arg_type: Type<'ctx>,
        loc: Location<'ctx>,
    ) -> Value<'ctx> {
        unsafe {
            mlir_capi::IR::FFIVal_::mlirBlockInsertArgument(&self, pos as i64, arg_type, &loc)
        }
    }
    pub fn get_argument(self, pos: usize) -> Value<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirBlockGetArgument(&self, pos as i64) }
    }
}

impl<'ctx> PartialEq<Block<'ctx>> for Block<'ctx> {
    fn eq(&self, other: &Block<'ctx>) -> bool {
        (unsafe { mlir_capi::IR::FFIVal_::<u8>::mlirBlockEqual(self, other) }) != 0
    }
}
impl<'ctx> Eq for Block<'ctx> {}
