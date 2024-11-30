use crate::affine_expr::*;
use crate::common::*;
use crate::context::*;
use crate::support::*;
use crate::type_cast::*;

use mlir_capi::IntegerSet as MLIR_IntegerSet;
use mlir_capi::IntegerSet::*;
use mlir_capi::IR::*;

use std::marker::PhantomData;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct IntegerSet<'ctx> {
    pub handle: MlirIntegerSet,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirIntegerSet> for IntegerSet<'ctx> {
    fn into(self) -> MlirIntegerSet {
        self.handle
    }
}

impl<'ctx> HandleWithContext<'ctx> for IntegerSet<'ctx> {
    type HandleTy = MlirIntegerSet;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { MLIR_IntegerSet::FFIVal_::mlirIntegerSetGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> NullableRef for IntegerSet<'ctx> {
    fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    fn create_null() -> Self {
        Self {
            handle: MlirIntegerSet {
                ptr: std::ptr::null_mut(),
            },
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> PartialEq for IntegerSet<'ctx> {
    fn eq(&self, other: &IntegerSet<'ctx>) -> bool {
        to_rbool(unsafe { MLIR_IntegerSet::FFIVal_::mlirIntegerSetEqual(*self, *other) })
    }
}
impl<'ctx> Eq for IntegerSet<'ctx> {}

impl<'ctx> IntegerSet<'ctx> {
    pub fn print(self, callback: &mut dyn PrintCallback) {
        unsafe {
            MLIR_IntegerSet::FFIVoid_::mlirIntegerSetPrint(
                self,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
    }
    pub fn dump(self) {
        unsafe {
            MLIR_IntegerSet::FFIVoid_::mlirIntegerSetDump(self);
        }
    }
    pub fn empty_get(ctx: &'ctx Context, num_dims: usize, num_symbols: usize) -> Self {
        let handle = unsafe {
            MLIR_IntegerSet::FFIVal_::mlirIntegerSetEmptyGet(
                ctx,
                num_dims as i64,
                num_symbols as i64,
            )
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn get(
        ctx: &'ctx Context,
        num_dims: usize,
        num_symbols: usize,
        equalities: &[AffineExpr<'ctx>],
        inequalities: &[AffineExpr<'ctx>],
    ) -> Self {
        let num_constraints = equalities.len() + inequalities.len();
        let mut constaints = Vec::new();
        constaints.reserve_exact(num_constraints);
        constaints.extend_from_slice(equalities);
        constaints.extend_from_slice(inequalities);
        let mut eq_flags = Vec::new();
        eq_flags.reserve_exact(num_constraints);
        eq_flags.resize(equalities.len(), to_cbool(true));
        eq_flags.resize(num_constraints, to_cbool(false));
        let handle = unsafe {
            MLIR_IntegerSet::FFIVal_::mlirIntegerSetGet(
                ctx,
                num_dims as i64,
                num_symbols as i64,
                num_constraints as i64,
                constaints.as_ptr() as *const _,
                eq_flags.as_ptr(),
            )
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn replace_get(
        self,
        dim_replacements: &[AffineExpr<'ctx>],
        symbol_replacements: &[AffineExpr<'ctx>],
        num_result_dims: usize,
        num_result_symbols: usize,
    ) -> Self {
        let handle = unsafe {
            MLIR_IntegerSet::FFIVal_::mlirIntegerSetReplaceGet(
                self,
                dim_replacements.as_ptr() as *const _,
                symbol_replacements.as_ptr() as *const _,
                num_result_dims as i64,
                num_result_symbols as i64,
            )
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn is_canonical_empty(self) -> bool {
        to_rbool(unsafe { MLIR_IntegerSet::FFIVal_::mlirIntegerSetIsCanonicalEmpty(self) })
    }
    pub fn get_num_dims(self) -> usize {
        (unsafe { MLIR_IntegerSet::FFIVal_::<i64>::mlirIntegerSetGetNumDims(self) }) as _
    }
    pub fn get_num_symbols(self) -> usize {
        (unsafe { MLIR_IntegerSet::FFIVal_::<i64>::mlirIntegerSetGetNumSymbols(self) }) as _
    }
    pub fn get_num_inputs(self) -> usize {
        (unsafe { MLIR_IntegerSet::FFIVal_::<i64>::mlirIntegerSetGetNumInputs(self) }) as _
    }
    pub fn get_num_constaints(self) -> usize {
        (unsafe { MLIR_IntegerSet::FFIVal_::<i64>::mlirIntegerSetGetNumConstraints(self) }) as _
    }
    pub fn get_num_equalities(self) -> usize {
        (unsafe { MLIR_IntegerSet::FFIVal_::<i64>::mlirIntegerSetGetNumEqualities(self) }) as _
    }
    pub fn get_num_inequalities(self) -> usize {
        (unsafe { MLIR_IntegerSet::FFIVal_::<i64>::mlirIntegerSetGetNumInequalities(self) }) as _
    }
    pub fn get_constraint(self, pos: usize) -> AffineExpr<'ctx> {
        let handle =
            unsafe { MLIR_IntegerSet::FFIVal_::mlirIntegerSetGetConstraint(self, pos as i64) };
        AffineExpr::from_handle_same_context(handle, &self)
    }
    pub fn is_constraint_equality(self, pos: usize) -> bool {
        to_rbool(unsafe {
            MLIR_IntegerSet::FFIVal_::mlirIntegerSetIsConstraintEq(self, pos as i64)
        })
    }
}
