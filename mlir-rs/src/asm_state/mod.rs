use mlir_capi::IR;
use mlir_capi::IR::MlirAsmState;

use crate::op_printing_flags::*;
use crate::operation::*;
use crate::value::*;

#[repr(C)]
pub struct AsmState {
    handle: MlirAsmState,
}

impl Into<MlirAsmState> for &AsmState {
    fn into(self) -> MlirAsmState {
        self.handle
    }
}

impl Drop for AsmState {
    fn drop(&mut self) {
        unsafe { IR::FFIVoid_::mlirAsmStateDestroy(&*self) }
    }
}

impl AsmState {
    pub fn create_for_op<'ctx>(op: &Operation<'ctx>, flags: &OpPrintingFlags) -> Self {
        Self {
            handle: unsafe { IR::FFIVal_::mlirAsmStateCreateForOperation(op, flags) },
        }
    }
    pub fn create_for_value<'ctx>(value: Value<'ctx>, flags: &OpPrintingFlags) -> Self {
        Self {
            handle: unsafe { IR::FFIVal_::mlirAsmStateCreateForValue(value, flags) },
        }
    }
}
