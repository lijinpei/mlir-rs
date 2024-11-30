use crate::block::*;
use crate::context::*;
use mlir_capi;
use mlir_capi::IR::*;
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Region<'ctx> {
    pub handle: MlirRegion,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> From<MlirRegion> for Region<'ctx> {
    fn from(value: MlirRegion) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirRegion> for &Region<'ctx> {
    fn into(self) -> MlirRegion {
        self.handle
    }
}

impl<'ctx> Region<'ctx> {
    pub fn create() -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirRegionCreate() }
    }
    // TODO: destroy
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null_mut()
    }
    pub fn get_first_block(self) -> Block<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirRegionGetFirstBlock(&self) }
    }
    pub fn append_owned_block(self, block: Block<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirRegionAppendOwnedBlock(&self, &block);
        }
    }
    pub fn insert_owned_block(self, pos: usize, block: Block<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirRegionInsertOwnedBlock(&self, pos as i64, &block);
        }
    }
    pub fn insert_owned_block_after(self, reference: Block<'ctx>, block: Block<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirRegionInsertOwnedBlockAfter(&self, &reference, &block);
        }
    }
    pub fn insert_owned_block_before(self, reference: Block<'ctx>, block: Block<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirRegionInsertOwnedBlockBefore(&self, &reference, &block);
        }
    }
    pub fn get_next_in_operation(self) -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirRegionGetNextInOperation(&self) }
    }
    pub fn take_body_of(self, other: Self) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirRegionTakeBody(&self, &other);
        }
    }
}

impl<'ctx> PartialEq<Region<'ctx>> for Region<'ctx> {
    fn eq(&self, other: &Region) -> bool {
        (unsafe { mlir_capi::IR::FFIVal_::<u8>::mlirRegionEqual(self, other) }) != 0
    }
}
