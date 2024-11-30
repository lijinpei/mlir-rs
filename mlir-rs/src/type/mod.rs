use crate::affine_map::*;
use crate::attribute::*;
use crate::common::*;
use crate::context::*;
use crate::location::*;
use crate::support::*;
use crate::type_cast::*;

use mlir_capi::IR::*;
use mlir_capi::{BuiltinTypes, IR};

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use strum::EnumIter;
#[allow(unused_imports)]
use strum::IntoEnumIterator;

use mlir_impl_macros;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Type<'ctx> {
    pub handle: MlirType,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> HandleWithContext<'ctx> for Type<'ctx> {
    type HandleTy = MlirType;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { IR::FFIVal_::mlirTypeGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> Into<MlirType> for Type<'ctx> {
    fn into(self) -> MlirType {
        self.handle
    }
}

impl<'ctx> PartialEq for Type<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        to_rbool(unsafe { FFIVal_::mlirTypeEqual(*self, *other) })
    }
}
impl<'ctx> Eq for Type<'ctx> {}

impl<'ctx> Debug for Type<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> Display for Type<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> NullableRef for Type<'ctx> {
    fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    fn create_null() -> Self {
        Type {
            handle: MlirType {
                ptr: std::ptr::null(),
            },
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Type<'ctx> {
    pub fn parse(ctx: &'ctx Context, s: &str) -> Self {
        let handle = unsafe { IR::FFIVal_::mlirTypeParseGet(ctx, StrRef::from(s)) };
        unsafe { Type::from_handle_and_phantom(handle, PhantomData::default()) }
    }
}

pub trait TypeTrait<'ctx>: Into<MlirType> + HandleWithContext<'ctx> + Copy {
    fn print(self, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirTypePrint(
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
    fn dump(self) {
        unsafe {
            IR::FFIVoid_::mlirTypeDump(self);
        }
    }
}

impl<'ctx> TypeTrait<'ctx> for Type<'ctx> {}

#[cfg(test)]
mod type_test {
    use super::*;

    #[test]
    fn test_null() {
        let t = Type::create_null();
        assert!(t.is_null());
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
    (RankedTensor, Shaped),
    (UnrankedTensor, Shaped),
    (MemRef, Shaped),
    (UnrankedMemRef, Shaped),
    Tuple,
    Function,
    Opaque
);

impl<'ctx> IntegerType<'ctx> {
    pub fn get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeGet(ctx, bitwidth) };
        let ty = unsafe { Type::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(ty) }
    }

    pub fn signed_get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeSignedGet(ctx, bitwidth) };
        let ty = unsafe { Type::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(ty) }
    }

    pub fn unsigned_get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirIntegerTypeUnsignedGet(ctx, bitwidth) };
        let ty = unsafe { Type::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(ty) }
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

            let in_ty_str = format!("i{}", i);
            let un_ty_str = format!("ui{}", i);
            let sn_ty_str = format!("si{}", i);

            let in_ty_str_1 = format!("{}", in_ty);
            let un_ty_str_1 = format!("{}", un_ty);
            let sn_ty_str_1 = format!("{}", sn_ty);

            assert_eq!(in_ty_str, in_ty_str_1);
            assert_eq!(un_ty_str, un_ty_str_1);
            assert_eq!(sn_ty_str, sn_ty_str_1);

            let in_ty_1 = Type::parse(&ctx, &in_ty_str);
            let un_ty_1 = Type::parse(&ctx, &un_ty_str);
            let sn_ty_1 = Type::parse(&ctx, &sn_ty_str);

            assert_eq!(in_ty, in_ty_1);
            assert_eq!(un_ty, un_ty_1);
            assert_eq!(sn_ty, sn_ty_1);
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
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirIndexTypeGet(ctx) };
        let ty = unsafe { Type::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(ty) }
    }
}

#[cfg(test)]
mod index_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let index_ty = IndexType::get(&ctx);
        assert!(IndexType::is_a(index_ty));

        let str_1 = format!("{}", index_ty);
        let str_2 = "index";
        assert_eq!(str_1, str_2);

        let index_ty_1 = Type::parse(&ctx, &str_2);
        assert_eq!(index_ty, index_ty_1);
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
            let handle = BuiltinTypes::FFIVal_::mlirComplexTypeGet(elem);
            let ty = Type::from_handle_same_context(handle, &elem);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_element_type(self) -> Type<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirComplexTypeGetElementType(self) };
        Type::from_handle_same_context(handle, &self)
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

    // FIXME:
    //#[test]
    //#[should_panic]
    //fn create_with_index() {
    //    let ctx = Context::create();
    //    let idx_ty = IndexType::get(&ctx);
    //    ComplexType::get(idx_ty.into());
    //}
}

pub trait ShapedTypeTrait<'ctx>: TypeTrait<'ctx> {
    fn get_element_type(self) -> Type<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetElementType(self) };
        Type::from_handle_same_context(handle, &self)
    }
    fn has_rank(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeHasRank(self) })
    }
    fn get_rank(self) -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetRank(self) }
    }
    fn has_static_shape(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeHasStaticShape(self) })
    }
    fn is_dynamic_dim(self, dim: usize) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeIsDynamicDim(self, dim as i64) })
    }
    fn get_dim_size(self, dim: usize) -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetDimSize(self, dim as i64) }
    }
    fn is_dynamic_size(size: i64) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeIsDynamicSize(size) })
    }
    fn get_dynamic_size() -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetDynamicSize() }
    }
    fn is_dynamic_stride_or_offset(s_or_o: i64) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeIsDynamicStrideOrOffset(s_or_o) })
    }
    fn get_dynamic_stride_or_offset() -> i64 {
        unsafe { BuiltinTypes::FFIVal_::mlirShapedTypeGetDynamicStrideOrOffset() }
    }
}

impl<'ctx> VectorType<'ctx> {
    pub fn get(shape: &[i64], elem_type: Type<'ctx>) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirVectorTypeGet(
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_checked(loc: Location<'ctx>, shape: &[i64], elem_type: Type<'ctx>) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirVectorTypeGetChecked(
                loc,
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_scalable(shape: &[i64], scalable: &[bool], elem_type: Type<'ctx>) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirVectorTypeGetScalable(
                shape.len() as i64,
                shape.as_ptr(),
                scalable.as_ptr() as *const _,
                elem_type,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_scalable_checked(
        loc: Location<'ctx>,
        shape: &[i64],
        scalable: &[bool],
        elem_type: Type<'ctx>,
    ) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirVectorTypeGetScalableChecked(
                loc,
                shape.len() as i64,
                shape.as_ptr(),
                scalable.as_ptr() as *const _,
                elem_type,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn is_scalable(self) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirVectorTypeIsScalable(self) })
    }
    pub fn is_dim_scalable(self, dim: usize) -> bool {
        to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirVectorTypeIsDimScalable(self, dim as i64) })
    }
}
impl<'ctx> ShapedTypeTrait<'ctx> for VectorType<'ctx> {}

#[cfg(test)]
mod vector_type_test {
    use super::*;

    // FIXME negative test for dynamic
    #[test]
    fn create() {
        let ctx = Context::create();
        let loc = Location::unknown_get(&ctx);
        let f4_ty = FloatType::get(&ctx, FloatKind::F4E2M1FN);
        let s8_ty = IntegerType::signed_get(&ctx, 8);
        let idx_ty = IndexType::get(&ctx);
        let elem_tys: &[Type] = &[f4_ty.into(), s8_ty.into(), idx_ty.into()];
        let shapes: &[&[i64]] = &[&[1], &[2, 5], &[10, 7, 3]];
        let scalables: &[&[bool]] = &[&[true], &[false, false], &[false, true, false]];
        for elem in elem_tys {
            for (shape, scalable) in shapes.iter().zip(scalables.iter()) {
                let vec_ty = VectorType::get(&shape, *elem);
                assert!(!vec_ty.is_scalable());
                assert_eq!(vec_ty, VectorType::get_checked(loc, &shape, *elem));

                let sca_vec_ty = VectorType::get_scalable(&shape, &scalable, *elem);
                assert!(IsA::<VectorType>::is_a(sca_vec_ty));
                assert_eq!(sca_vec_ty.is_scalable(), scalable.contains(&true));
                assert_eq!(
                    sca_vec_ty,
                    VectorType::get_scalable_checked(loc, &shape, &scalable, *elem)
                );
                for vt in [vec_ty, sca_vec_ty] {
                    assert!(IsA::<VectorType>::is_a(vt));
                    assert_eq!(vt.get_element_type(), *elem);
                    assert!(vt.has_rank());
                    let rank = shape.len();
                    assert_eq!(vt.get_rank(), rank.try_into().unwrap());
                    assert!(vt.has_static_shape());
                    for i in 0..rank {
                        assert!(!vt.is_dynamic_dim(i));
                        assert_eq!(shape[i], vt.get_dim_size(i));
                    }
                }
                for dim in 0..shape.len() {
                    assert_eq!(vec_ty.is_dim_scalable(dim), false);
                    assert_eq!(sca_vec_ty.is_dim_scalable(dim), scalable[dim]);
                }
            }
        }
    }
}

impl<'ctx> RankedTensorType<'ctx> {
    pub fn get(shape: &[i64], elem_type: Type<'ctx>, encoding: Attr<'ctx>) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirRankedTensorTypeGet(
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
                encoding,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_checked(
        loc: Location<'ctx>,
        shape: &[i64],
        elem_type: Type<'ctx>,
        encoding: Attr<'ctx>,
    ) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirRankedTensorTypeGetChecked(
                loc,
                shape.len() as i64,
                shape.as_ptr(),
                elem_type,
                encoding,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_encoding(self) -> Attr<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirRankedTensorTypeGetEncoding(self) };
        Attr::from_handle_same_context(handle, &self)
    }
}
impl<'ctx> ShapedTypeTrait<'ctx> for RankedTensorType<'ctx> {}

#[cfg(test)]
mod ranked_tensor_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let i32_ty = IntegerType::get(&ctx, 32);
        let tf32_ty = FloatType::get(&ctx, FloatKind::TF32);
        let idx_ty = IndexType::get(&ctx);
        // FIXME: remove this into
        let c_i32_ty = ComplexType::get(i32_ty.into());
        let f64_ty = FloatType::get(&ctx, FloatKind::F64);
        let elem_tys: &[Type] = &[
            i32_ty.into(),
            tf32_ty.into(),
            idx_ty.into(),
            c_i32_ty.into(),
            f64_ty.into(),
        ];
        let k_dynamic = RankedTensorType::get_dynamic_size();
        let shapes: &[&[i64]] = &[
            &[1],
            &[2, 3],
            &[5, 7, 11],
            &[k_dynamic],
            &[2, k_dynamic, 3],
            &[5, 7, 11, k_dynamic],
        ];
        let attrs: &[Attr] = &[
            UnitAttr::get(&ctx).into(),
            FloatAttr::f64_get(&ctx, f64_ty.into(), 3.14).into(),
            IntegerAttr::get(i32_ty.into(), 13).into(),
            Attr::create_null().into(),
        ];
        let d_s_o = RankedTensorType::get_dynamic_stride_or_offset();
        assert!(RankedTensorType::is_dynamic_stride_or_offset(d_s_o));
        let loc = Location::unknown_get(&ctx);
        for elem in elem_tys {
            for shape in shapes {
                for i in 0..shape.len() {
                    assert_eq!(
                        RankedTensorType::is_dynamic_size(shape[i]),
                        shape[i] == k_dynamic
                    );
                }
                let has_static_shape = !shape.contains(&k_dynamic);
                for attr in attrs {
                    let tensor_ty = RankedTensorType::get(&shape, *elem, *attr);
                    assert!(RankedTensorType::is_a(tensor_ty));
                    assert_eq!(
                        tensor_ty,
                        RankedTensorType::get_checked(loc, &shape, *elem, *attr)
                    );
                    assert_eq!(tensor_ty.get_element_type(), *elem);
                    assert!(tensor_ty.has_rank());
                    assert_eq!(tensor_ty.get_rank(), shape.len().try_into().unwrap());
                    assert_eq!(tensor_ty.has_static_shape(), has_static_shape);
                    for i in 0..shape.len() {
                        assert_eq!(tensor_ty.get_dim_size(i), shape[i]);
                        assert_eq!(tensor_ty.is_dynamic_dim(i), shape[i] == k_dynamic);
                    }
                    assert_eq!(tensor_ty.get_encoding(), *attr);
                }
            }
        }
    }
}

impl<'ctx> UnrankedTensorType<'ctx> {
    pub fn get(elem_ty: Type<'ctx>) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirUnrankedTensorTypeGet(elem_ty);
            let ty = Type::from_handle_same_context(handle, &elem_ty);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_checked(loc: Location<'ctx>, elem_ty: Type<'ctx>) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirUnrankedTensorTypeGetChecked(loc, elem_ty);
            let ty = Type::from_handle_same_context(handle, &elem_ty);
            IsA::<Self>::cast(ty)
        }
    }
}
impl<'ctx> ShapedTypeTrait<'ctx> for UnrankedTensorType<'ctx> {}

#[cfg(test)]
mod unranked_tensor_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let i32_ty = IntegerType::get(&ctx, 32);
        let tf32_ty = FloatType::get(&ctx, FloatKind::TF32);
        let idx_ty = IndexType::get(&ctx);
        // FIXME: remove this into
        let c_i32_ty = ComplexType::get(i32_ty.into());
        let f64_ty = FloatType::get(&ctx, FloatKind::F64);
        let elem_tys: &[Type] = &[
            i32_ty.into(),
            tf32_ty.into(),
            idx_ty.into(),
            c_i32_ty.into(),
            f64_ty.into(),
        ];
        let d_s_o = UnrankedTensorType::get_dynamic_stride_or_offset();
        assert!(UnrankedTensorType::is_dynamic_stride_or_offset(d_s_o));
        let loc = Location::unknown_get(&ctx);
        for elem in elem_tys {
            assert!(!IsA::<UnrankedTensorType>::is_a(*elem));
            let un_tensor_ty = UnrankedTensorType::get(*elem);
            assert!(UnrankedTensorType::is_a(un_tensor_ty));
            assert_eq!(un_tensor_ty, UnrankedTensorType::get_checked(loc, *elem));
            assert_eq!(un_tensor_ty.get_element_type(), *elem);
            assert!(!un_tensor_ty.has_rank());
            assert!(!un_tensor_ty.has_static_shape());
        }
    }
}

impl<'ctx> MemRefType<'ctx> {
    pub fn get(elem_type: Type<'ctx>, shape: &[i64], layout: Attr, mem_space: Attr) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirMemRefTypeGet(
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                layout,
                mem_space,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        shape: &[i64],
        layout: Attr,
        mem_space: Attr,
    ) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirMemRefTypeGetChecked(
                loc,
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                layout,
                mem_space,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn contiguous_get(elem_type: Type<'ctx>, shape: &[i64], mem_space: Attr) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirMemRefTypeContiguousGet(
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                mem_space,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn contiguous_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        shape: &[i64],
        mem_space: Attr,
    ) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirMemRefTypeContiguousGetChecked(
                loc,
                elem_type,
                shape.len() as i64,
                shape.as_ptr(),
                mem_space,
            );
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_layout(self) -> Attr<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirMemRefTypeGetLayout(self) };
        Attr::from_handle_same_context(handle, &self)
    }
    pub fn get_affine_map(self) -> AffineMap<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirMemRefTypeGetAffineMap(self) };
        AffineMap::from_handle_same_context(handle, &self)
    }
    pub fn get_memory_space(self) -> Attr<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirMemRefTypeGetMemorySpace(self) };
        Attr::from_handle_same_context(handle, &self)
    }
    pub fn get_strides_and_offset(self) -> Option<(Vec<i64>, i64)> {
        let mut strides = Vec::<i64>::new();
        strides.resize(self.get_rank() as _, 0);
        let mut offset = 0;
        let res: LogicalResult = unsafe {
            BuiltinTypes::FFIVal_::mlirMemRefTypeGetStridesAndOffset(
                self,
                strides.as_mut_ptr(),
                &mut offset as *mut _,
            )
        };
        if res.is_success() {
            Some((strides, offset))
        } else {
            None
        }
    }
}
impl<'ctx> ShapedTypeTrait<'ctx> for MemRefType<'ctx> {}

#[cfg(test)]
mod memref_type_test {
    use super::*;
    use crate::attribute::affine_map_attr_test::*;

    fn create_strided_layout<'ctx>(shape: &[i64], k_dynamic: i64, col_major: bool) -> Vec<i64> {
        let num_dims = shape.len();
        let mut strides = vec![0 as i64; num_dims];
        let mut stride = 1;
        for i in 0..num_dims {
            let i = if col_major { i } else { num_dims - 1 - i };
            let new_stride = if shape[i] == k_dynamic || stride == k_dynamic {
                k_dynamic
            } else {
                shape[i] * stride
            };
            strides[i] = stride;
            stride = new_stride;
        }
        strides
    }

    #[test]
    fn create() {
        let ctx = Context::create();
        let i32_ty = IntegerType::get(&ctx, 32);
        let tf32_ty = FloatType::get(&ctx, FloatKind::TF32);
        let idx_ty = IndexType::get(&ctx);
        // FIXME: remove this into
        let c_i32_ty = ComplexType::get(i32_ty.into());
        let f64_ty = FloatType::get(&ctx, FloatKind::F64);
        let elem_tys: &[Type] = &[
            i32_ty.into(),
            tf32_ty.into(),
            idx_ty.into(),
            c_i32_ty.into(),
            f64_ty.into(),
        ];
        let k_dynamic = MemRefType::get_dynamic_size();
        let shapes: &[&[i64]] = &[
            &[1],
            &[2, 3],
            &[5, 7, 11],
            &[k_dynamic],
            &[2, k_dynamic, 3],
            &[5, 7, 11, k_dynamic],
        ];
        let i32_zero_attr = IntegerAttr::get(i32_ty.into(), 0);
        let mem_spaces: &[Attr] = &[
            i32_zero_attr.into(),
            IntegerAttr::get(i32_ty.into(), 1).into(),
            Attr::create_null(),
        ];
        let d_s_o = MemRefType::get_dynamic_stride_or_offset();
        assert!(MemRefType::is_dynamic_stride_or_offset(d_s_o));
        let loc = Location::unknown_get(&ctx);
        for shape in shapes {
            let affine_col_major = create_affine_layout_attr(&ctx, &shape, k_dynamic, true);
            let affine_row_major = create_affine_layout_attr(&ctx, &shape, k_dynamic, false);
            // There is more than one structural representation of affine-map for the same affine-map.
            // let col_major_affine_map = affine_col_major.get_value();
            // let row_major_affine_map = affine_row_major.get_value();
            let col_major_strides = create_strided_layout(&shape, k_dynamic, true);
            let row_major_strides = create_strided_layout(&shape, k_dynamic, false);
            let strided_col_major = StridedLayoutAttr::get(&ctx, 0, &col_major_strides);
            let strided_row_major = StridedLayoutAttr::get(&ctx, 0, &row_major_strides);
            let layouts: &[(Attr, i32)] = &[
                (affine_col_major.into(), 0),
                (affine_row_major.into(), 1),
                (strided_col_major.into(), 2),
                (strided_row_major.into(), 3),
                (Attr::create_null(), 4),
            ];
            for (layout, layout_kind) in layouts {
                for mem_space in mem_spaces {
                    for elem in elem_tys {
                        let memref_ty = MemRefType::get(*elem, *shape, *layout, *mem_space);
                        assert_eq!(
                            memref_ty,
                            MemRefType::get_checked(loc, *elem, *shape, *layout, *mem_space)
                        );
                        assert_eq!(*elem, memref_ty.get_element_type());
                        assert!(memref_ty.has_rank());
                        assert_eq!(memref_ty.get_rank(), shape.len().try_into().unwrap());
                        let has_static_shape = !shape.contains(&k_dynamic);
                        assert_eq!(has_static_shape, memref_ty.has_static_shape());
                        for dim in 0..shape.len() {
                            assert_eq!(memref_ty.is_dynamic_dim(dim), shape[dim] == k_dynamic);
                            assert_eq!(memref_ty.get_dim_size(dim), shape[dim]);
                        }
                        if *mem_space == i32_zero_attr {
                            assert!(memref_ty.get_memory_space().is_null());
                        } else {
                            assert_eq!(memref_ty.get_memory_space(), *mem_space);
                        }
                        let (get_strides, get_offset) = memref_ty.get_strides_and_offset().unwrap();
                        assert_eq!(get_offset, 0);
                        let get_layout = memref_ty.get_layout();
                        if !layout.is_null() {
                            assert_eq!(get_layout, *layout);
                        }
                        let get_affine_map = memref_ty.get_affine_map();
                        assert!(!get_affine_map.is_null());
                        match layout_kind {
                            0 => {
                                assert_eq!(get_strides, col_major_strides);
                                // assert_eq!(get_affine_map, col_major_affine_map);
                            }
                            1 => {
                                assert_eq!(get_strides, row_major_strides);
                                // assert_eq!(get_affine_map, row_major_affine_map);
                            }
                            2 => {
                                assert_eq!(get_strides, col_major_strides);
                                // assert_eq!(get_affine_map, col_major_affine_map);
                            }
                            3 => {
                                assert_eq!(get_strides, row_major_strides);
                                // assert_eq!(get_affine_map, row_major_affine_map);
                            }
                            4 => {
                                assert_eq!(get_strides, row_major_strides);
                                // assert_eq!(get_affine_map, row_major_affine_map);
                            }
                            _ => {
                                assert!(false);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<'ctx> UnrankedMemRefType<'ctx> {
    pub fn get(elem_type: Type<'ctx>, mem_space: Attr) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirUnrankedMemRefTypeGet(elem_type, mem_space);
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_checked(loc: Location<'ctx>, elem_type: Type<'ctx>, mem_space: Attr) -> Self {
        unsafe {
            let handle =
                BuiltinTypes::FFIVal_::mlirUnrankedMemRefTypeGetChecked(loc, elem_type, mem_space);
            let ty = Type::from_handle_same_context(handle, &elem_type);
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_memory_space(self) -> Attr<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirUnrankedMemrefGetMemorySpace(self) };
        Attr::from_handle_same_context(handle, &self)
    }
}
impl<'ctx> ShapedTypeTrait<'ctx> for UnrankedMemRefType<'ctx> {}

impl<'ctx> TupleType<'ctx> {
    pub fn get(ctx: &'ctx Context, elements: &[Type<'ctx>]) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirTupleTypeGet(
                ctx,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            );
            let ty = Type::from_handle_and_phantom(handle, PhantomData::default());
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_num_types(self) -> usize {
        (unsafe { BuiltinTypes::FFIVal_::<i64>::mlirTupleTypeGetNumTypes(self) }) as _
    }
    pub fn get_type(self, pos: usize) -> Type<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirTupleTypeGetType(self, pos as i64) };
        Type::from_handle_same_context(handle, &self)
    }
}
#[cfg(test)]
mod unranked_memref_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let i32_ty = IntegerType::get(&ctx, 32);
        let tf32_ty = FloatType::get(&ctx, FloatKind::TF32);
        let idx_ty = IndexType::get(&ctx);
        // FIXME: remove this into
        let c_i32_ty = ComplexType::get(i32_ty.into());
        let f64_ty = FloatType::get(&ctx, FloatKind::F64);
        let elem_tys: &[Type] = &[
            i32_ty.into(),
            tf32_ty.into(),
            idx_ty.into(),
            c_i32_ty.into(),
            f64_ty.into(),
        ];
        let i32_zero_attr = IntegerAttr::get(i32_ty.into(), 0);
        let mem_spaces: &[Attr] = &[
            i32_zero_attr.into(),
            IntegerAttr::get(i32_ty.into(), 1).into(),
            Attr::create_null(),
        ];
        let loc = Location::unknown_get(&ctx);
        for elem in elem_tys {
            for mem_space in mem_spaces {
                let u_memref_ty = UnrankedMemRefType::get(*elem, *mem_space);
                assert!(UnrankedMemRefType::is_a(u_memref_ty));
                assert_eq!(
                    u_memref_ty,
                    UnrankedMemRefType::get_checked(loc, *elem, *mem_space)
                );
                assert_eq!(u_memref_ty.get_element_type(), *elem);
                if i32_zero_attr == *mem_space {
                    assert!(u_memref_ty.get_memory_space().is_null());
                } else {
                    assert_eq!(u_memref_ty.get_memory_space(), *mem_space);
                }

                assert_eq!(u_memref_ty.get_element_type(), *elem);
                assert!(!u_memref_ty.has_rank());
                assert!(!u_memref_ty.has_static_shape());
            }
        }
    }
}

impl<'ctx> FunctionType<'ctx> {
    pub fn get(ctx: &'ctx Context, inputs: &[Type<'ctx>], results: &[Type<'ctx>]) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirFunctionTypeGet(
                ctx,
                inputs.len() as i64,
                inputs.as_ptr() as *const _,
                results.len() as i64,
                results.as_ptr() as *const _,
            );
            let ty = Type::from_handle_and_phantom(handle, PhantomData::default());
            IsA::<Self>::cast(ty)
        }
    }
    pub fn get_num_inputs(self) -> usize {
        (unsafe { BuiltinTypes::FFIVal_::<i64>::mlirFunctionTypeGetNumInputs(self) }) as _
    }
    pub fn get_num_results(self) -> usize {
        (unsafe { BuiltinTypes::FFIVal_::<i64>::mlirFunctionTypeGetNumResults(self) }) as _
    }
    pub fn get_input(self, pos: usize) -> Type<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirFunctionTypeGetInput(self, pos as i64) };
        Type::from_handle_same_context(handle, &self)
    }
    pub fn get_result(self, pos: usize) -> Type<'ctx> {
        let handle = unsafe { BuiltinTypes::FFIVal_::mlirFunctionTypeGetResult(self, pos as i64) };
        Type::from_handle_same_context(handle, &self)
    }
}

#[cfg(test)]
mod function_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let i32_ty: Type = IntegerType::get(&ctx, 32).into();
        let tf32_ty: Type = FloatType::get(&ctx, FloatKind::TF32).into();
        let idx_ty: Type = IndexType::get(&ctx).into();
        let c_i32_ty: Type = ComplexType::get(i32_ty).into();
        let f64_ty: Type = FloatType::get(&ctx, FloatKind::F64).into();
        let elem_tys: &[&[Type]] = &[
            &[],
            &[i32_ty],
            &[tf32_ty, idx_ty],
            &[c_i32_ty, f64_ty, tf32_ty],
        ];
        for input_tys in elem_tys {
            for output_tys in elem_tys {
                let func_ty = FunctionType::get(&ctx, input_tys, output_tys);
                assert!(FunctionType::is_a(func_ty));
                assert_eq!(func_ty.get_num_inputs(), input_tys.len());
                assert_eq!(func_ty.get_num_results(), output_tys.len());
                for i in 0..input_tys.len() {
                    assert_eq!(func_ty.get_input(i), input_tys[i]);
                }
                for i in 0..output_tys.len() {
                    assert_eq!(func_ty.get_result(i), output_tys[i]);
                }
            }
        }
    }
}

impl<'ctx> OpaqueType<'ctx> {
    pub fn get(ctx: &'ctx Context, ns: &str, type_data: &str) -> Self {
        unsafe {
            let handle = BuiltinTypes::FFIVal_::mlirOpaqueTypeGet(
                ctx,
                to_string_ref(ns),
                to_string_ref(type_data),
            );
            let ty = Type::from_handle_and_phantom(handle, PhantomData::default());
            IsA::<Self>::cast(ty)
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

#[cfg(test)]
mod opaque_type_test {
    use super::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        ctx.set_allow_unregistered_dialects(true);
        let d_ns = "foobar";
        let data = "2000";
        let opaque_ty = OpaqueType::get(&ctx, d_ns, data);
        assert!(OpaqueType::is_a(opaque_ty));
        assert_eq!(opaque_ty.get_dialect_namespace(), d_ns);
        assert_eq!(opaque_ty.get_data(), data);
    }
}
