use crate::block::*;
use crate::common::*;
use crate::context::*;

use mlir_capi;
use mlir_capi::IR::*;
use std::convert::Into;
use std::marker::PhantomData;

#[repr(C)]
pub struct Region<'ctx> {
    pub handle: MlirRegion,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirRegion> for &Region<'ctx> {
    fn into(self) -> MlirRegion {
        self.handle
    }
}

impl<'ctx> Drop for Region<'ctx> {
    fn drop(&mut self) {
        unsafe {
            FFIVoid_::mlirRegionDestroy(&*self);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RegionRef<'ctx> {
    pub handle: MlirRegion,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirRegion> for &RegionRef<'ctx> {
    fn into(self) -> MlirRegion {
        self.handle
    }
}

impl<'ctx> RegionRef<'ctx> {
    pub unsafe fn wrap(handle: MlirRegion, phantom: PhantomData<&'ctx Context>) -> Self {
        Self {
            handle: handle,
            phantom: phantom,
        }
    }
}

impl<'ctx> Region<'ctx> {
    pub fn create() -> Self {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirRegionCreate() };
        Self {
            handle: handle,
            phantom: PhantomData::default(),
        }
    }
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null_mut()
    }
    pub fn get_first_block(self) -> BlockRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirRegionGetFirstBlock(&self) };
        unsafe { BlockRef::wrap(handle, self.phantom) }
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
    pub fn get_next_in_operation(self) -> RegionRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirRegionGetNextInOperation(&self) };
        unsafe { RegionRef::wrap(handle, self.phantom) }
    }
    pub fn take_body_of(self, other: Self) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirRegionTakeBody(&self, &other);
        }
    }
}

impl<'ctx> PartialEq<Region<'ctx>> for Region<'ctx> {
    fn eq(&self, other: &Region) -> bool {
        to_rbool(unsafe { mlir_capi::IR::FFIVal_::<u8>::mlirRegionEqual(self, other) })
    }
}
