use crate::attribute::*;
use crate::context::*;
use crate::support::*;
use mlir_capi::IR;
use mlir_capi::IR::*;
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Location<'ctx> {
    pub handle: MlirLocation,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> HandleWithContext<'ctx> for Location<'ctx> {
    type HandleTy = MlirLocation;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { IR::FFIVal_::mlirLocationGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

// FIXME: remove this
impl<'ctx> From<MlirLocation> for Location<'ctx> {
    fn from(value: MlirLocation) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirLocation> for Location<'ctx> {
    fn into(self) -> MlirLocation {
        self.handle
    }
}

impl<'ctx> Location<'ctx> {
    pub fn get_attribute(self) -> Attr<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirLocationGetAttribute(self) };
        Attr::from_handle_same_context(handle, &self)
    }
    pub fn from_attribute(attr: Attr<'ctx>) -> Self {
        unsafe { IR::FFIVal_::mlirLocationFromAttribute(attr) }
    }
    pub fn file_line_col_get(ctx: &'ctx Context, file: &str, line: u32, col: u32) -> Self {
        unsafe {
            IR::FFIVal_::mlirLocationFileLineColGet(ctx, Into::<StrRef>::into(file), line, col)
        }
    }
    pub fn call_site_get(callee: Self, caller: Self) -> Self {
        unsafe { IR::FFIVal_::mlirLocationCallSiteGet(callee, caller) }
    }
    pub fn fused_get(ctx: &'ctx Context, locations: &[Self], metadata: Attr<'ctx>) -> Self {
        unsafe {
            IR::FFIVal_::mlirLocationFusedGet(
                ctx,
                locations.len() as i64,
                locations.as_ptr() as *const _,
                metadata,
            )
        }
    }
    pub fn name_get(ctx: &'ctx Context, name: &str, child_loc: Self) -> Self {
        let name_str_ref = StrRef::from_str(name);
        unsafe { IR::FFIVal_::mlirLocationNameGet(ctx, name_str_ref, child_loc) }
    }
    pub fn unknown_get(ctx: &'ctx Context) -> Self {
        unsafe { IR::FFIVal_::mlirLocationUnknownGet(ctx) }
    }
    // FIXME
    //pub fn get_context(self) -> Context {
    //    unsafe { IR::FFIVal_::mlirLocationGetContext(&self) }
    //}
}