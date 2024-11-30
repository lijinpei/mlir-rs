use crate::block::*;
use crate::context::*;
use crate::location::*;
use crate::operation::*;
use crate::support::*;
use mlir_capi::IR::*;
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Module<'ctx> {
    pub handle: MlirModule,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> From<MlirModule> for Module<'ctx> {
    fn from(value: MlirModule) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> HandleWithContext<'ctx> for Module<'ctx> {
    type HandleTy = MlirModule;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { FFIVal_::mlirModuleGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> Into<MlirModule> for Module<'ctx> {
    fn into(self) -> MlirModule {
        self.handle
    }
}

impl<'ctx> Module<'ctx> {
    pub fn create_empty(loc: Location) -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirModuleCreateEmpty(loc) }
    }
    pub fn create_parse(ctx: &'ctx Context, s: &str) -> Self {
        let str_ref = StrRef::from_str(s);
        unsafe { mlir_capi::IR::FFIVal_::mlirModuleCreateParse(ctx, str_ref) }
    }
    pub fn get_body(self) -> BlockRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirModuleGetBody(self) };
        unsafe { BlockRef::wrap(handle, self.phantom) }
    }
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    // FIXME
    pub fn get_operation_ref(self) -> OperationRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirModuleGetOperation(self) };
        OperationRef::from_handle_same_context(handle, &self)
    }
    pub fn from_operation(op: Operation<'ctx>) -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirModuleFromOperation(&op) }
    }
}
