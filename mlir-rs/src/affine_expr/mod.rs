use crate::affine_map::*;
use crate::common::*;
use crate::context::*;
use crate::support::*;
use crate::type_cast::*;

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use mlir_capi::AffineExpr as MLIR_AffineExpr;
use mlir_capi::AffineExpr::*;
use mlir_capi::IR::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AffineExpr<'ctx> {
    pub handle: MlirAffineExpr,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirAffineExpr> for AffineExpr<'ctx> {
    fn into(self) -> MlirAffineExpr {
        self.handle
    }
}

impl<'ctx> HandleWithContext<'ctx> for AffineExpr<'ctx> {
    type HandleTy = MlirAffineExpr;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> PartialEq for AffineExpr<'ctx> {
    fn eq(&self, other: &AffineExpr<'ctx>) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprEqual(*self, *other) })
    }
}
impl<'ctx> Eq for AffineExpr<'ctx> {}

impl<'ctx> Debug for AffineExpr<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> Display for AffineExpr<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> NullableRef for AffineExpr<'ctx> {
    fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    fn create_null() -> Self {
        Self {
            handle: MlirAffineExpr {
                ptr: std::ptr::null_mut(),
            },
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> AffineExpr<'ctx> {
    pub fn print(self, callback: &mut dyn PrintCallback) {
        unsafe {
            MLIR_AffineExpr::FFIVoid_::mlirAffineExprPrint(
                self,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
    }
    fn print_to_formatter(
        self,
        formatter: &mut std::fmt::Formatter,
    ) -> Result<(), std::fmt::Error> {
        let mut printer = PrintToFormatter {
            formatter: formatter,
        };
        self.print(&mut printer);
        Ok(())
    }
    pub fn dump(self) {
        unsafe {
            MLIR_AffineExpr::FFIVoid_::mlirAffineExprDump(self);
        }
    }
    pub fn is_symbolic_or_constant(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsSymbolicOrConstant(self) })
    }
    pub fn is_pure_affine(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsPureAffine(self) })
    }
    pub fn get_largest_known_divisor(self) -> i64 {
        unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprGetLargestKnownDivisor(self) }
    }
    pub fn is_multiple_of(self, factor: i64) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsMultipleOf(self, factor) })
    }
    pub fn is_func_of_dim(self, dim: i64) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsFunctionOfDim(self, dim) })
    }
    pub fn compose(self, map: AffineMap<'ctx>) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprCompose(self, map) };
        Self::from_handle_same_context(handle, &self)
    }
    pub fn dim_expr_get(ctx: &'ctx Context, pos: usize) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineDimExprGet(ctx, pos as i64) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn is_a_dim(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsADim(self) })
    }
    pub fn symbol_expr_get(ctx: &'ctx Context, pos: usize) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineSymbolExprGet(ctx, pos as i64) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn symbol_expr_get_pos(self) -> i64 {
        unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineSymbolExprGetPosition(self) }
    }
    pub fn is_a_constant(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsAConstant(self) })
    }
    pub fn const_expr_get(ctx: &'ctx Context, value: i64) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineConstantExprGet(ctx, value) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn const_expr_get_value(self) -> i64 {
        unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineConstantExprGetValue(self) }
    }
    pub fn is_a_add(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsAAdd(self) })
    }
    pub fn add_expr_get(lhs: Self, rhs: Self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineAddExprGet(lhs, rhs) };
        Self::from_handle_same_context(handle, &lhs)
    }
    pub fn is_a_mul(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsAMul(self) })
    }
    pub fn mul_expr_get(lhs: Self, rhs: Self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineMulExprGet(lhs, rhs) };
        Self::from_handle_same_context(handle, &lhs)
    }
    pub fn is_a_mod(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsAMod(self) })
    }
    pub fn mod_expr_get(lhs: Self, rhs: Self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineModExprGet(lhs, rhs) };
        Self::from_handle_same_context(handle, &lhs)
    }
    pub fn is_a_floordiv(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsAFloorDiv(self) })
    }
    pub fn floordiv_expr_get(lhs: Self, rhs: Self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineFloorDivExprGet(lhs, rhs) };
        Self::from_handle_same_context(handle, &lhs)
    }
    pub fn is_a_ceildiv(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsACeilDiv(self) })
    }
    pub fn ceildiv_expr_get(lhs: Self, rhs: Self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineCeilDivExprGet(lhs, rhs) };
        Self::from_handle_same_context(handle, &lhs)
    }
    pub fn is_a_binary(self) -> bool {
        to_rbool(unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineExprIsABinary(self) })
    }
    pub fn binary_get_lhs(self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineBinaryOpExprGetLHS(self) };
        Self::from_handle_same_context(handle, &self)
    }
    pub fn binary_get_rhs(self) -> Self {
        let handle = unsafe { MLIR_AffineExpr::FFIVal_::mlirAffineBinaryOpExprGetRHS(self) };
        Self::from_handle_same_context(handle, &self)
    }
}
