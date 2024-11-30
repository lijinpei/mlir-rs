use crate::affine_expr::*;
use crate::common::*;
use crate::context::*;
use crate::support::*;
use crate::type_cast::*;

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use mlir_capi::AffineMap as MLIR_AffineMap;
use mlir_capi::AffineMap::*;
use mlir_capi::IR::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AffineMap<'ctx> {
    pub handle: MlirAffineMap,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirAffineMap> for AffineMap<'ctx> {
    fn into(self) -> MlirAffineMap {
        self.handle
    }
}

impl<'ctx> HandleWithContext<'ctx> for AffineMap<'ctx> {
    type HandleTy = MlirAffineMap;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> PartialEq for AffineMap<'ctx> {
    fn eq(&self, other: &AffineMap<'ctx>) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapEqual(*self, *other) })
    }
}
impl<'ctx> Eq for AffineMap<'ctx> {}

impl<'ctx> Debug for AffineMap<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> Display for AffineMap<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}
impl<'ctx> NullableRef for AffineMap<'ctx> {
    fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    fn create_null() -> Self {
        Self {
            handle: MlirAffineMap {
                ptr: std::ptr::null_mut(),
            },
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> AffineMap<'ctx> {
    pub fn print(self, callback: &mut dyn PrintCallback) {
        unsafe {
            MLIR_AffineMap::FFIVoid_::mlirAffineMapPrint(
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
            MLIR_AffineMap::FFIVoid_::mlirAffineMapDump(self);
        }
    }
    pub fn empty_get(ctx: &'ctx Context) -> Self {
        let handle = unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapEmptyGet(ctx) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn zero_result_get(ctx: &'ctx Context, dim_count: usize, symbol_count: usize) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapZeroResultGet(
                ctx,
                dim_count as i64,
                symbol_count as i64,
            )
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn get(
        ctx: &'ctx Context,
        dim_count: usize,
        symbol_count: usize,
        exprs: &[AffineExpr],
    ) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapGet(
                ctx,
                dim_count as i64,
                symbol_count as i64,
                exprs.len() as i64,
                exprs.as_ptr() as *const _ as *mut _,
            )
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn constant_get(ctx: &'ctx Context, val: i64) -> Self {
        let handle = unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapConstantGet(ctx, val) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn multi_dim_identity_get(ctx: &'ctx Context, num_dims: usize) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapMultiDimIdentityGet(ctx, num_dims as i64)
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn minor_identity_get(ctx: &'ctx Context, dims: usize, results: usize) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapMinorIdentityGet(ctx, dims as i64, results as i64)
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn permutation_get(ctx: &'ctx Context, permutation: &[usize]) -> Self {
        let perm: Vec<_> = permutation.iter().map(|x| *x as std::ffi::c_uint).collect();
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapPermutationGet(
                ctx,
                permutation.len() as i64,
                perm.as_ptr() as *mut _,
            )
        };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn is_identify(self) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapIsIdentity(self) })
    }
    pub fn is_minor_identity(self) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapIsMinorIdentity(self) })
    }
    pub fn is_empty(self) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapIsEmpty(self) })
    }
    pub fn is_single_constant(self) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapIsSingleConstant(self) })
    }
    pub fn get_single_constant_result(self) -> i64 {
        unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapGetSingleConstantResult(self) }
    }
    pub fn get_num_dims(self) -> usize {
        (unsafe { MLIR_AffineMap::FFIVal_::<i64>::mlirAffineMapGetNumDims(self) }) as _
    }
    pub fn get_num_symbols(self) -> usize {
        (unsafe { MLIR_AffineMap::FFIVal_::<i64>::mlirAffineMapGetNumSymbols(self) }) as _
    }
    pub fn get_num_results(self) -> usize {
        (unsafe { MLIR_AffineMap::FFIVal_::<i64>::mlirAffineMapGetNumResults(self) }) as _
    }
    pub fn get_result(self, pos: usize) -> AffineExpr<'ctx> {
        let handle = unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapGetResult(self, pos as i64) };
        AffineExpr::from_handle_same_context(handle, &self)
    }
    pub fn get_num_inputs(self) -> usize {
        (unsafe { MLIR_AffineMap::FFIVal_::<i64>::mlirAffineMapGetNumInputs(self) }) as _
    }
    pub fn is_projected_permutation(self) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapIsProjectedPermutation(self) })
    }
    pub fn is_permutation(self) -> bool {
        to_rbool(unsafe { MLIR_AffineMap::FFIVal_::mlirAffineMapIsPermutation(self) })
    }
    pub fn get_sub_map(self, result_pos: &[usize]) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapGetSubMap(
                self,
                result_pos.len() as i64,
                result_pos.as_ptr() as *mut _,
            )
        };
        AffineMap::from_handle_same_context(handle, &self)
    }
    pub fn get_major_sub_map(self, num_results: usize) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapGetMajorSubMap(self, num_results as i64)
        };
        AffineMap::from_handle_same_context(handle, &self)
    }
    pub fn get_minor_sub_map(self, num_results: usize) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapGetMinorSubMap(self, num_results as i64)
        };
        AffineMap::from_handle_same_context(handle, &self)
    }
    pub fn replace(
        self,
        expr: AffineExpr<'ctx>,
        replacement: AffineExpr<'ctx>,
        num_result_dims: usize,
        num_results_syms: usize,
    ) -> Self {
        let handle = unsafe {
            MLIR_AffineMap::FFIVal_::mlirAffineMapReplace(
                self,
                expr,
                replacement,
                num_result_dims as i64,
                num_results_syms as i64,
            )
        };
        AffineMap::from_handle_same_context(handle, &self)
    }
    pub fn compress_unused_combols(affine_maps: &[AffineMap<'ctx>]) -> Vec<AffineMap<'ctx>> {
        let mut res = vec![AffineMap::create_null(); affine_maps.len()];
        extern "C" fn compress_helper(res: *mut std::ffi::c_void, idx: i64, m: MlirAffineMap) {
            let map = AffineMap {
                handle: m,
                phantom: PhantomData::default(),
            };
            let res_vec: &mut Vec<_> = unsafe { (res as *mut Vec<_>).as_mut_unchecked() };
            res_vec[idx as usize] = map;
        }
        unsafe {
            MLIR_AffineMap::FFIVoid_::mlirAffineMapCompressUnusedSymbols(
                affine_maps.as_ptr() as *mut _,
                affine_maps.len() as i64,
                (&mut res).as_mut_ptr() as *mut _,
                compress_helper as *mut _,
            );
        }
        res
    }
}

#[cfg(test)]
pub mod affine_map_test {
    use super::*;
    use crate::r#type::*;

    pub fn create_affine_layout<'ctx>(
        ctx: &'ctx Context,
        shape: &[i64],
        k_dynamic: i64,
        col_major: bool,
    ) -> AffineMap<'ctx> {
        let num_dims = shape.len();
        let mut stride = 1;
        let mut affine_expr = AffineExpr::const_expr_get(ctx, 0);
        let mut num_symbols = 0;
        for i in 0..num_dims {
            let i = if col_major { i } else { num_dims - 1 - i };
            let this_stride;
            if stride == k_dynamic {
                this_stride = AffineExpr::symbol_expr_get(ctx, num_symbols);
                num_symbols += 1;
            } else {
                this_stride = AffineExpr::const_expr_get(ctx, stride);
                if shape[i] == k_dynamic {
                    stride = k_dynamic
                } else {
                    stride *= shape[i];
                }
            };
            let dim = AffineExpr::dim_expr_get(ctx, i);
            let this_expr = AffineExpr::mul_expr_get(dim, this_stride);
            affine_expr = AffineExpr::add_expr_get(affine_expr, this_expr);
        }
        AffineMap::get(ctx, num_dims, num_symbols, &[affine_expr])
    }

    pub fn generate_some_affine_maps<'ctx>(ctx: &'ctx Context) -> Vec<AffineMap<'ctx>> {
        let k_dynamic = RankedTensorType::get_dynamic_size();
        let shapes: &[&[i64]] = &[
            &[10],
            &[k_dynamic],
            &[k_dynamic, 5],
            &[7, k_dynamic],
            &[11, 13, 15],
            &[23, k_dynamic, 29],
        ];
        let mut res = Vec::new();
        for col_major in [true, false] {
            for shape in shapes {
                res.push(create_affine_layout(ctx, shape, k_dynamic, col_major));
            }
        }
        res
    }
}
