use crate::context::*;
use crate::support::*;
use mlir_capi;
use mlir_capi::IR::*;
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Dialect<'ctx> {
    pub handle: MlirDialect,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> From<MlirDialect> for Dialect<'ctx> {
    fn from(value: MlirDialect) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirDialect> for &Dialect<'ctx> {
    fn into(self) -> MlirDialect {
        self.handle
    }
}

impl<'ctx> Dialect<'ctx> {
    pub fn get_namespace(self) -> StrRef<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::<_>::mlirDialectGetNamespace(&self) }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DialectHandle {
    pub handle: MlirDialectHandle,
}

impl From<MlirDialectHandle> for DialectHandle {
    fn from(value: MlirDialectHandle) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirDialectHandle> for &DialectHandle {
    fn into(self) -> MlirDialectHandle {
        self.handle
    }
}

impl DialectHandle {
    pub fn insert_dialect(self, reg: DialectRegistry) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirDialectHandleInsertDialect(&self, &reg);
        }
    }
    pub fn register_dialect(self, ctx: &Context) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirDialectHandleRegisterDialect(&self, ctx);
        }
    }
    pub fn load_dialect<'ctx>(self, ctx: &'ctx Context) -> Dialect<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirDialectHandleLoadDialect(&self, ctx) }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DialectRegistry {
    pub handle: MlirDialectRegistry,
}

impl From<MlirDialectRegistry> for DialectRegistry {
    fn from(value: MlirDialectRegistry) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirDialectRegistry> for &DialectRegistry {
    fn into(self) -> MlirDialectRegistry {
        self.handle
    }
}

impl DialectRegistry {
    pub fn create() -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirDialectRegistryCreate() }
    }
    pub fn is_null(self) -> bool {
        self.handle.ptr != std::ptr::null_mut()
    }
    // FIXME: distroy
}
