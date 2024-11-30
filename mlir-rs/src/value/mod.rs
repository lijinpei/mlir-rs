use crate::block::*;
use crate::context::*;
use crate::operation::*;
use crate::r#type::*;
use mlir_capi::IR::*;
use std::cmp::{Eq, PartialEq};
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Value<'ctx> {
    pub handle: MlirValue,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> HandleWithContext<'ctx> for Value<'ctx> {
    type HandleTy = MlirValue;
    fn get_context_handle(&self) -> MlirContext {
        let ty = self.get_type();
        ty.get_context_handle()
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

// FIXME: remove this
impl<'ctx> From<MlirValue> for Value<'ctx> {
    fn from(value: MlirValue) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirValue> for Value<'ctx> {
    fn into(self) -> MlirValue {
        self.handle
    }
}

impl<'ctx> Value<'ctx> {
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    pub fn is_block_arg(self) -> bool {
        (unsafe { mlir_capi::IR::FFIVal_::<u8>::mlirValueIsABlockArgument(self) }) != 0
    }
    pub fn is_op_result(self) -> bool {
        (unsafe { mlir_capi::IR::FFIVal_::<u8>::mlirValueIsAOpResult(self) }) != 0
    }
    pub fn block_arg_get_owner(self) -> BlockRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirBlockArgumentGetOwner(self) };
        unsafe { BlockRef::wrap(handle, self.phantom) }
    }
    pub fn block_arg_get_arg_number(self) -> usize {
        (unsafe { mlir_capi::IR::FFIVal_::<i64>::mlirBlockArgumentGetArgNumber(self) }) as usize
    }
    pub fn block_arg_set_type(self, arg_type: Type<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirBlockArgumentSetType(self, arg_type);
        }
    }
    pub fn op_res_get_owner(self) -> OperationRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirOpResultGetOwner(self) };
        OperationRef::from_handle_same_context(handle, &self)
    }
    pub fn op_res_get_res_number(self) -> usize {
        (unsafe { mlir_capi::IR::FFIVal_::<i64>::mlirOpResultGetResultNumber(self) }) as usize
    }
    pub fn get_type(self) -> Type<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirValueGetType(self) };
        Type::from_handle_same_context(handle, &self)
    }
    pub fn set_type(self, new_type: Type<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirValueSetType(self, new_type);
        };
    }
    pub fn dump(self) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirValueDump(self);
        }
    }
    // TODO: print, print_as_operand
    pub fn get_first_use(self) -> OpOperand<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirValueGetFirstUse(self) }
    }
    pub fn replace_all_use_with(self, other: Value<'ctx>) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirValueReplaceAllUsesOfWith(self, other);
        }
    }
}

impl<'ctx> PartialEq for Value<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        (unsafe { mlir_capi::IR::FFIVal_::<u8>::mlirValueEqual(*self, *other) }) != 0
    }
}
impl<'ctx> Eq for Value<'ctx> {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct OpOperand<'ctx> {
    pub handle: MlirOpOperand,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> HandleWithContext<'ctx> for OpOperand<'ctx> {
    type HandleTy = MlirOpOperand;
    fn get_context_handle(&self) -> MlirContext {
        self.get_value().get_context_handle()
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> From<MlirOpOperand> for OpOperand<'ctx> {
    fn from(value: MlirOpOperand) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirOpOperand> for OpOperand<'ctx> {
    fn into(self) -> MlirOpOperand {
        self.handle
    }
}

impl<'ctx> OpOperand<'ctx> {
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null_mut()
    }
    pub fn get_value(self) -> Value<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirOpOperandGetValue(self) }
    }
    pub fn get_owner(self) -> OperationRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirOpOperandGetOwner(self) };
        OperationRef::from_handle_same_context(handle, &self)
    }
    pub fn get_number(self) -> usize {
        (unsafe { mlir_capi::IR::FFIVal_::<i64>::mlirOpOperandGetOperandNumber(self) }) as _
    }
    pub fn get_next_use(self) -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirOpOperandGetNextUse(self) }
    }
}
