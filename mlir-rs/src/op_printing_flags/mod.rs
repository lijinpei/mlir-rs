use mlir_capi::IR;
use mlir_capi::IR::MlirOpPrintingFlags;

use crate::common::*;

#[repr(C)]
pub struct OpPrintingFlags {
    handle: MlirOpPrintingFlags,
}

impl Into<MlirOpPrintingFlags> for &OpPrintingFlags {
    fn into(self) -> MlirOpPrintingFlags {
        self.handle
    }
}

impl OpPrintingFlags {
    pub fn create() -> Self {
        Self {
            handle: unsafe { IR::FFIVal_::mlirOpPrintingFlagsCreate() },
        }
    }

    pub fn elide_large_elements_attr(&mut self, limit: usize) {
        unsafe {
            IR::FFIVoid_::mlirOpPrintingFlagsElideLargeElementsAttrs(&*self, limit as i64);
        }
    }

    pub fn elide_large_resource_string(&mut self, limit: usize) {
        unsafe { IR::FFIVoid_::mlirOpPrintingFlagsElideLargeResourceString(&*self, limit as i64) }
    }

    pub fn enable_debug_info(&mut self, enable: bool, pretty_form: bool) {
        unsafe {
            IR::FFIVoid_::mlirOpPrintingFlagsEnableDebugInfo(
                &*self,
                to_cbool(enable),
                to_cbool(pretty_form),
            );
        }
    }

    pub fn print_generic_op_form(&mut self) {
        unsafe {
            IR::FFIVoid_::mlirOpPrintingFlagsPrintGenericOpForm(&*self);
        }
    }

    pub fn use_local_scope(&mut self) {
        unsafe {
            IR::FFIVoid_::mlirOpPrintingFlagsUseLocalScope(&*self);
        }
    }

    pub fn assume_verified(&mut self) {
        unsafe {
            IR::FFIVoid_::mlirOpPrintingFlagsAssumeVerified(&*self);
        }
    }

    pub fn skip_regions(&mut self) {
        unsafe {
            IR::FFIVoid_::mlirOpPrintingFlagsSkipRegions(&*self);
        }
    }
}

impl Drop for OpPrintingFlags {
    fn drop(&mut self) {
        unsafe { IR::FFIVoid_::mlirOpPrintingFlagsDestroy(&*self) }
    }
}
