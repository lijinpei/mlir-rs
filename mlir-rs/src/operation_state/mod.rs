use crate::attribute::*;
use crate::block::*;
use crate::context::*;
use crate::location::*;
use crate::r#type::*;
use crate::region::*;
use crate::support::*;
use crate::value::*;

use mlir_capi::IR;
use mlir_capi::IR::MlirOperationState;

use std::marker::PhantomData;

#[repr(C)]
pub struct OperationState<'ctx> {
    handle: MlirOperationState,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirOperationState> for &OperationState<'ctx> {
    fn into(self) -> MlirOperationState {
        self.handle
    }
}

impl<'ctx> OperationState<'ctx> {
    pub fn get(name: &'ctx str, loc: Location<'ctx>) -> Self {
        let name_ref: StrRef = name.into();
        let handle = unsafe { IR::FFIVal_::mlirOperationStateGet(name_ref, loc) };
        Self {
            handle: handle,
            phantom: PhantomData::default(),
        }
    }
    pub fn add_results(&mut self, results: &[Type<'ctx>]) -> &mut Self {
        unsafe {
            IR::FFIVoid_::mlirOperationStateAddResults(
                (&mut self.handle) as *mut _,
                results.len() as i64,
                results.as_ptr() as *const _,
            );
        };
        self
    }
    pub fn add_operands(&mut self, operands: &[Value<'ctx>]) -> &mut Self {
        unsafe {
            IR::FFIVoid_::mlirOperationStateAddOperands(
                (&mut self.handle) as *mut _,
                operands.len() as i64,
                operands.as_ptr() as *const _,
            );
        };
        self
    }
    pub fn add_owned_regions(&mut self, regions: &[Region<'ctx>]) -> &mut Self {
        unsafe {
            IR::FFIVoid_::mlirOperationStateAddOwnedRegions(
                (&mut self.handle) as *mut _,
                regions.len() as i64,
                regions.as_ptr() as *const _,
            );
        };
        self
    }
    pub fn add_successors(&mut self, successors: &[Block<'ctx>]) -> &mut Self {
        unsafe {
            IR::FFIVoid_::mlirOperationStateAddSuccessors(
                (&mut self.handle) as *mut _,
                successors.len() as i64,
                successors.as_ptr() as *const _,
            );
        };
        self
    }
    pub fn add_attributes(&mut self, attributes: &[NamedAttr<'ctx>]) -> &mut Self {
        unsafe {
            IR::FFIVoid_::mlirOperationStateAddAttributes(
                (&mut self.handle) as *mut _,
                attributes.len() as i64,
                attributes.as_ptr() as *const _,
            );
        };
        self
    }
    pub fn enable_type_inference(&mut self) -> &mut Self {
        unsafe {
            IR::FFIVoid_::mlirOperationStateEnableResultTypeInference((&mut self.handle) as *mut _);
        };
        self
    }
}
