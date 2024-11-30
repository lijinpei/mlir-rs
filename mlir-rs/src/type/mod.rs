use crate::attribute::Attr;
use crate::common::*;
use crate::context::*;
use crate::location::Location;
use crate::support::*;
use mlir_capi::IR::*;
use mlir_capi::{BuiltinTypes, IR};
use std::marker::PhantomData;
use strum::EnumIter;
#[allow(unused_imports)]
use strum::IntoEnumIterator;

use mlir_impl_macros;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Type<'ctx> {
    pub handle: MlirType,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Type<'ctx> {
    pub fn parse(ctx: &'ctx Context, s: &str) -> Self {
        unsafe { IR::FFIVal_::mlirTypeParseGet(ctx, StrRef::from(s)) }
    }
    pub fn get_context(self) -> ContextRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirTypeGetContext(self) };
        unsafe { ContextRef::construct(handle, self.phantom) }
    }
    pub fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    pub fn print(self, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirTypePrint(
                self,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
    }
    pub fn print_to_formatter(
        &self,
        formatter: &mut std::fmt::Formatter,
    ) -> Result<(), std::fmt::Error> {
        struct PrintHelper<'a, 'b> {
            pub formatter: &'a mut std::fmt::Formatter<'b>,
        }
        impl<'a, 'b> PrintCallback for PrintHelper<'a, 'b> {
            fn print(&mut self, s: &str) {
                self.formatter.write_str(s).unwrap();
            }
        }
        let mut printer = PrintHelper {
            formatter: formatter,
        };
        self.print(&mut printer);
        Ok(())
    }
    pub fn dump(self) {
        unsafe {
            IR::FFIVoid_::mlirTypeDump(self);
        }
    }
}

impl<'ctx> PartialEq for Type<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        to_rbool(unsafe { FFIVal_::mlirTypeEqual(*self, *other) })
    }
}
impl<'ctx> Eq for Type<'ctx> {}

impl<'ctx> From<MlirType> for Type<'ctx> {
    fn from(value: MlirType) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirType> for Type<'ctx> {
    fn into(self) -> MlirType {
        self.handle
    }
}

impl<'ctx> std::fmt::Debug for Type<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> std::fmt::Display for Type<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

mlir_impl_macros::define_builtin_types!(
    Integer,
    Float,
    Index,
    None,
    Complex,
    Shaped,
    (Vector, Shaped),
    (Tensor, Shaped),
    (MemRef, Shaped),
    Tuple,
    Function,
    Opaque
);

impl<'ctx> IntegerType<'ctx> {
    pub fn get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        let ty: Type = unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeGet(ctx, bitwidth) };
        unsafe { Self::from_type_unchecked(ty) }
    }

    pub fn signed_get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        let ty: Type = unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeSignedGet(ctx, bitwidth) };
        unsafe { Self::from_type_unchecked(ty) }
    }

    pub fn unsigned_get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        let ty: Type = unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeUnsignedGet(ctx, bitwidth) };
        unsafe { Self::from_type_unchecked(ty) }
    }

    pub fn get_width(self) -> u32 {
        unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeGetWidth(self) }
    }

    pub fn is_signless(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeIsSignless(self) })
    }
    pub fn is_signed(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeIsSigned(self) })
    }
    pub fn is_unsigned(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeIsUnsigned(self) })
    }
}

#[cfg(test)]
mod integer_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let bits = [1, 2, 5, 8, 16, 32, 64, 128];
        let mut int_tys = Vec::new();
        for i in bits {
            let in_ty = IntegerType::get(&ctx, i);
            let un_ty = IntegerType::unsigned_get(&ctx, i);
            let sn_ty = IntegerType::signed_get(&ctx, i);
            assert_eq!(in_ty.get_width(), i);
            assert_eq!(un_ty.get_width(), i);
            assert_eq!(sn_ty.get_width(), i);

            assert!(!in_ty.is_signed());
            assert!(in_ty.is_signless());

            assert!(!un_ty.is_signed());
            assert!(!un_ty.is_signless());

            assert!(sn_ty.is_signed());
            assert!(!sn_ty.is_signless());
            int_tys.push(in_ty);
            int_tys.push(un_ty);
            int_tys.push(sn_ty);
        }
        let num_int_tys = int_tys.len();
        for i in 0..num_int_tys {
            for j in 0..num_int_tys {
                assert_eq!(i == j, int_tys[i] == int_tys[j]);
            }
        }
    }
}

impl<'ctx> IndexType<'ctx> {
    pub fn get(ctx: &'ctx Context) -> Self {
        let ty: Type = unsafe { BuiltinTypes::FFIVal_::mlirIndexTypeGet(ctx) };
        Self { ty }
    }
}

#[cfg(test)]
mod index_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let index_ty = IndexType::get(&ctx);
        assert!(IndexType::is_a(index_ty.into()));
    }
}

mlir_impl_macros::define_float_kind! {
F4E2M1FN,
F6E2M3FN,
F6E3M2FN,
F8E5M2,
F8E4M3,
F8E4M3FN,
F8E5M2FNUZ,
F8E4M3FNUZ,
F8E4M3B11FNUZ,
F8E3M4,
F8E8M0FNU,
BF16,
F16,
F32,
F64,
TF32,}

impl<'ctx> FloatType<'ctx> {
    pub fn get_width(self) -> u32 {
        unsafe { BuiltinTypes::FFIVal_::mlirFloatTypeGetWidth(self) }
    }
}

#[cfg(test)]
mod float_type_test {
    use super::*;

    fn get_fp_bitwidth(fp_kind: FloatKind) -> u32 {
        match fp_kind {
            FloatKind::F4E2M1FN => 4,
            FloatKind::F6E2M3FN | FloatKind::F6E3M2FN => 6,
            FloatKind::F8E5M2
            | FloatKind::F8E4M3
            | FloatKind::F8E4M3FN
            | FloatKind::F8E5M2FNUZ
            | FloatKind::F8E4M3FNUZ
            | FloatKind::F8E4M3B11FNUZ
            | FloatKind::F8E3M4
            | FloatKind::F8E8M0FNU => 8,
            FloatKind::BF16 | FloatKind::F16 => 16,
            FloatKind::TF32 => 19,
            FloatKind::F32 => 32,
            FloatKind::F64 => 64,
        }
    }

    #[test]
    fn create() {
        let all_fp_kinds: Vec<_> = FloatKind::iter().collect();
        let ctx = Context::create();
        let all_fp_types: Vec<_> = all_fp_kinds
            .iter()
            .map(|x| FloatType::get(&ctx, *x))
            .collect();
        let all_fp_typeids: Vec<_> = all_fp_kinds
            .iter()
            .map(|x| FloatType::get_typeid(*x))
            .collect();
        let num_fp_kinds = all_fp_kinds.len();
        for i in 0..num_fp_kinds {
            let fp_i = all_fp_types[i];
            assert_eq!(fp_i.get_width(), get_fp_bitwidth(all_fp_kinds[i]));
            for j in 0..num_fp_kinds {
                let fp_j = all_fp_types[j];
                assert_eq!(i == j, fp_i == fp_j);
                assert_eq!(i == j, all_fp_typeids[i] == all_fp_typeids[j]);
                assert_eq!(i == j, fp_i.is_a_fp(all_fp_kinds[j]));
            }
        }
    }
}

impl<'ctx> ComplexType<'ctx> {
    pub fn get(elem: Type<'ctx>) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirComplexTypeGet(elem);
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_element_type(self) -> Type<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirComplexTypeGetElementType(self) }
    }
}

#[cfg(test)]
mod complex_type_test {
    use super::*;
    #[test]
    fn create() {
        let ctx = Context::create();
        let i32_ty = IntegerType::get(&ctx, 32);
        let u48_ty = IntegerType::unsigned_get(&ctx, 48);
        let f16_ty = FloatType::get(&ctx, FloatKind::F16);
        let elem_types: &[Type] = &[i32_ty.into(), f16_ty.into(), u48_ty.into()];
        let complex_types: Vec<_> = elem_types.iter().map(|x| ComplexType::get(*x)).collect();
        let num = complex_types.len();
        for i in 0..num {
            let ty = complex_types[i];
            let elem_ty = elem_types[i];
            assert!(ComplexType::is_a(ty.into()));
            assert!(elem_ty == ty.get_element_type());
            for j in 0..num {
                assert_eq!(i == j, ty == complex_types[j]);
            }
        }
    }
    //#[test]
    //#[should_panic]
    //fn create_with_index() {
    //    let ctx = Context::create();
    //    let idx_ty = IndexType::get(&ctx);
    //    ComplexType::get(idx_ty.into());
    //}
}

impl<'ctx> ShapedType<'ctx> {
    pub fn get_element_type(self) -> Type<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetElementType(self) }
    }
    pub fn has_rank(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeHasRank(self) })
    }
    pub fn get_rank(self) -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetRank(self) }
    }
    pub fn has_static_shape(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeHasStaticShape(self) })
    }
    pub fn is_dynamic_dim(self, dim: usize) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeIsDynamicDim(self, dim as i64) })
    }
    pub fn get_dim_size(self, dim: usize) -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetDimSize(self, dim as i64) }
    }
    pub fn is_dynamic_size(size: i64) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeIsDynamicSize(size) })
    }
    pub fn get_dynamic_size() -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetDynamicSize() }
    }
    pub fn is_dynamic_stride_or_offset(s_or_o: i64) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeIsDynamicStrideOrOffset(s_or_o) })
    }
    pub fn get_dynamic_stride_or_offset() -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetDynamicStrideOrOffset() }
    }
}

impl<'ctx> VectorType<'ctx> {
    pub fn get(shape: &[i64], elem_type: Type<'ctx>) -> Self {
        unsafe {
            let ty: Type = BuiltinTypes::FFIVal_::mlirVectorTypeGet(
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_checked(loc: Location<'ctx>, shape: &[i64], elem_type: Type) -> Self {
        unsafe {
            let ty: Type = BuiltinTypes::FFIVal_::mlirVectorTypeGetChecked(
                &loc,
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_scalable(shape: &[i64], scalable: &[bool], elem_type: Type<'ctx>) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirVectorTypeGetScalable(
                shape.len() as i64,
                shape.as_ptr(),
                scalable.as_ptr() as *const _,
                elem_type,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_scalable_checked(
        loc: Location<'ctx>,
        shape: &[i64],
        scalable: &[bool],
        elem_type: Type<'ctx>,
    ) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirVectorTypeGetScalableChecked(
                &loc,
                shape.len() as i64,
                shape.as_ptr(),
                scalable.as_ptr() as *const _,
                elem_type,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn is_scalable(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirVectorTypeIsScalable(self) })
    }
    pub fn is_dim_scalable(self, dim: usize) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirVectorTypeIsDimScalable(self, dim as i64) })
    }
}

#[cfg(test)]
mod vector_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let f4_ty = FloatType::get(&ctx, FloatKind::F4E2M1FN);
        let s8_ty = IntegerType::signed_get(&ctx, 8);
        let idx_ty = IndexType::get(&ctx);
        let elem_tys: &[Type] = &[f4_ty.into(), s8_ty.into(), idx_ty.into()];
        let shapes: &[&[i64]] = &[&[1], &[2, 5], &[10, 7, 3]];
        for elem in elem_tys {
            for shape in shapes {
                let vec_ty = VectorType::get(&shape, *elem);
                let shaped_ty: ShapedType = vec_ty.into();
                let ty: Type = shaped_ty.into();
                assert!(VectorType::is_a(ty));
                assert!(!VectorType::is_scalable(vec_ty));
                assert_eq!(shaped_ty.get_element_type(), *elem);
                assert!(shaped_ty.has_rank());
                let rank = shape.len();
                assert_eq!(shaped_ty.get_rank(), rank.try_into().unwrap());
                assert!(shaped_ty.has_static_shape());
                for i in 0..rank {
                    assert!(!shaped_ty.is_dynamic_dim(i));
                    assert_eq!(shape[i], shaped_ty.get_dim_size(i));
                }
            }
        }
    }
}

impl<'ctx> TensorType<'ctx> {
    pub fn ranked_tensor_type_get(
        shape: &[i64],
        elem_type: Type<'ctx>,
        encoding: Attr<'ctx>,
    ) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirRankedTensorTypeGet(
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
                encoding,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn ranked_tensor_type_get_checked(
        loc: Location<'ctx>,
        shape: &[i64],
        elem_type: Type<'ctx>,
        encoding: Attr<'ctx>,
    ) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirRankedTensorTypeGetChecked(
                &loc,
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
                encoding,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_ranked_tensor_encoding(self) -> Attr<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirRankedTensorTypeGetEncoding(self) }
    }
    pub fn unranked_tensor_type_get(elem_ty: Type<'ctx>) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirUnrankedTensorTypeGet(elem_ty);
            Self::from_type_unchecked(ty)
        }
    }
    pub fn unranked_tensor_type_get_checked(loc: Location<'ctx>, elem_ty: Type<'ctx>) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirUnrankedTensorTypeGetChecked(&loc, elem_ty);
            Self::from_type_unchecked(ty)
        }
    }
}

impl<'ctx> MemRefType<'ctx> {
    pub fn ranked_get(elem_type: Type<'ctx>, shape: &[i64], layout: Attr, mem_space: Attr) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirMemRefTypeGet(
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                layout,
                mem_space,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn ranked_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        shape: &[i64],
        layout: Attr,
        mem_space: Attr,
    ) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirMemRefTypeGetChecked(
                &loc,
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                layout,
                mem_space,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn ranked_contiguous_get(elem_type: Type<'ctx>, shape: &[i64], mem_space: Attr) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirMemRefTypeContiguousGet(
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                mem_space,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn ranked_contiguous_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        shape: &[i64],
        mem_space: Attr,
    ) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirMemRefTypeContiguousGetChecked(
                &loc,
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                mem_space,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn unranked_get(elem_type: Type<'ctx>, mem_space: Attr) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirUnrankedMemRefTypeGet(elem_type, mem_space);
            Self::from_type_unchecked(ty)
        }
    }
    pub fn unranked_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        mem_space: Attr,
    ) -> Self {
        unsafe {
            let ty =
                BuiltinTypes::FFIVal_::mlirUnrankedMemRefTypeGetChecked(&loc, elem_type, mem_space);
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_layout(self) -> Attr<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirMemRefTypeGetLayout(self) }
    }
    // FIXME: get affine map
    pub fn get_memory_space(self) -> Attr<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirMemRefTypeGetMemorySpace(self) }
    }
    pub fn get_strides_and_offset(self) -> (Vec<i64>, i64) {
        let mut strides = Vec::<i64>::new();
        strides.resize(self.ty.get_rank() as _, 0);
        let mut offset = 0;
        // FIXME: check logical result
        let _: LogicalResult = unsafe {
            BuiltinTypes::FFIVal_::mlirMemRefTypeGetStridesAndOffset(
                self,
                strides.as_mut_ptr(),
                &mut offset as *mut _,
            )
        };
        (strides, offset)
    }
    // FIXME: unranked memory space
}

impl<'ctx> TupleType<'ctx> {
    pub fn get(ctx: &'ctx Context, elements: &[Type<'ctx>]) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirTupleTypeGet(
                ctx,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_num_types(self) -> usize {
        (unsafe { BuiltinTypes::FFIVal_::<i64>::mlirTupleTypeGetNumTypes(self) }) as _
    }
    pub fn get_type(self, pos: usize) -> Type<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirTupleTypeGetType(self, pos as i64) }
    }
}

impl<'ctx> FunctionType<'ctx> {
    pub fn get(ctx: &'ctx Context, inputs: &[Type<'ctx>], results: &[Type<'ctx>]) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirFunctionTypeGet(
                ctx,
                inputs.len() as i64,
                inputs.as_ptr() as *const _,
                results.len() as i64,
                results.as_ptr() as *const _,
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_num_inputs(self) -> usize {
        (unsafe { BuiltinTypes::FFIVal_::<i64>::mlirFunctionTypeGetNumInputs(self) }) as _
    }
    pub fn get_num_results(self) -> usize {
        (unsafe { BuiltinTypes::FFIVal_::<i64>::mlirFunctionTypeGetNumResults(self) }) as _
    }
    pub fn get_input(self, pos: usize) -> Type<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirFunctionTypeGetInput(self, pos as i64) }
    }
    pub fn get_result(self, pos: usize) -> Type<'ctx> {
        unsafe { BuiltinTypes::FFIVal_::mlirFunctionTypeGetResult(self, pos as i64) }
    }
}

impl<'ctx> OpaqueType<'ctx> {
    pub fn get(ctx: &'ctx Context, ns: &str, type_data: &str) -> Self {
        unsafe {
            let ty = BuiltinTypes::FFIVal_::mlirOpaqueTypeGet(
                ctx,
                to_string_ref(ns),
                to_string_ref(type_data),
            );
            Self::from_type_unchecked(ty)
        }
    }
    pub fn get_dialect_namespace(self) -> &'ctx str {
        let str_ref: StrRef =
            unsafe { BuiltinTypes::FFIVal_::mlirOpaqueTypeGetDialectNamespace(self) };
        str_ref.into()
    }
    pub fn get_data(self) -> &'ctx str {
        let str_ref: StrRef = unsafe { BuiltinTypes::FFIVal_::mlirOpaqueTypeGetData(self) };
        str_ref.into()
    }
}
