//use crate::attribute::Attr;
use crate::context::*;
//use crate::location::Location;
use crate::support::*;
//use mlir_capi::BuiltinTypes::*;
use mlir_capi::IR::*;
use mlir_capi::{BuiltinTypes, IR};
//use std::cmp::{Eq, PartialEq};
use crate::common::*;
use std::marker::PhantomData;

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
        extern "C" fn print_helper(
            s: mlir_capi::Support::MlirStringRef,
            ptr: *mut std::ffi::c_void,
        ) {
            let ptr_to_callback = ptr as *mut &mut dyn PrintCallback;
            let callback: &mut dyn PrintCallback = unsafe { *ptr_to_callback };
            let str_ref = StrRef::from_ffi(s);
            callback.print(str_ref.to_str());
        }
        unsafe {
            IR::FFIVoid_::mlirTypePrint(
                self,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
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

//#[cfg(test)]
//mod type_test {
//    use super::*;
//    use crate::context::context_test::*;
//
//    #[test]
//    fn create() {
//        let ctx = create_context_with_all_upstream_dialects();
//    }
//}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct IntegerType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> IntegerType<'ctx> {
    pub unsafe fn from_type_unchecked(ty: Type<'ctx>) -> Self {
        debug_assert!(to_rbool(mlir_capi_extra::mlirTypeIsIntegerType(ty.handle)));
        Self { ty }
    }

    pub fn from_type(ty: Type<'ctx>) -> Option<Self> {
        if to_rbool(unsafe { BuiltinTypes::FFIVal_::mlirTypeIsAInteger(ty) }) {
            Some(Self { ty })
        } else {
            None
        }
    }

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

impl<'ctx> TryFrom<Type<'ctx>> for IntegerType<'ctx> {
    type Error = &'static str;

    fn try_from(ty: Type<'ctx>) -> Result<Self, Self::Error> {
        Self::from_type(ty).ok_or("not a integer-type")
    }
}

impl<'ctx> Into<MlirType> for IntegerType<'ctx> {
    fn into(self) -> MlirType {
        Type::into(self.ty)
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
#[repr(C)]
#[derive(Copy, Clone)]
pub struct IndexType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> IndexType<'ctx> {
    pub fn get(ctx: &'ctx Context) -> Self {
        let ty: Type = unsafe { BuiltinTypes::FFIVal_::mlirIndexTypeGet(ctx) };
        Self { ty }
    }
}

//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct FloatType<'ctx> {
//    pub ty: Type<'ctx>,
//}
//
//impl<'ctx> FloatType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: Type::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    pub fn get_width(self) -> u32 {
//        unsafe { mlirFloatTypeGetWidth(self.to_ffi()) }
//    }
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct NoneType<'ctx> {
//    pub ty: Type<'ctx>,
//}
//
//impl<'ctx> NoneType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: Type::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    // FIXME
//    //pub fn get(ctx: &'ctx Context) -> Self {
//    //    Self::from_ffi(unsafe { mlirNoneTypeGet(ctx.to_ffi()) })
//    //}
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct ComplexType<'ctx> {
//    pub ty: Type<'ctx>,
//}
//
//impl<'ctx> ComplexType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: Type::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    pub fn get(elem_ty: Type<'ctx>) -> Self {
//        Self::from_ffi(unsafe { mlirComplexTypeGet(elem_ty.to_ffi()) })
//    }
//    pub fn get_element_type(self) -> Type<'ctx> {
//        Type::from_ffi(unsafe { mlirComplexTypeGetElementType(self.to_ffi()) })
//    }
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct ShapedType<'ctx> {
//    pub ty: Type<'ctx>,
//}
//
//impl<'ctx> ShapedType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: Type::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    pub fn get_element_type(self) -> Type<'ctx> {
//        Type::from_ffi(unsafe { mlirShapedTypeGetElementType(self.to_ffi()) })
//    }
//    pub fn has_rank(self) -> bool {
//        (unsafe { mlirShapedTypeHasRank(self.to_ffi()) }) != 0
//    }
//    pub fn get_rank(self) -> i64 {
//        unsafe { mlirShapedTypeGetRank(self.to_ffi()) }
//    }
//    pub fn has_static_shape(self) -> bool {
//        (unsafe { mlirShapedTypeHasStaticShape(self.to_ffi()) }) != 0
//    }
//    pub fn is_dynamic_dim(self, dim: usize) -> bool {
//        (unsafe { mlirShapedTypeIsDynamicDim(self.to_ffi(), dim as _) }) != 0
//    }
//    pub fn get_dim_size(self, dim: usize) -> i64 {
//        unsafe { mlirShapedTypeGetDimSize(self.to_ffi(), dim as _) }
//    }
//    pub fn is_dynamic_size(size: i64) -> bool {
//        (unsafe { mlirShapedTypeIsDynamicSize(size) }) != 0
//    }
//    pub fn get_dynamic_size() -> i64 {
//        unsafe { mlirShapedTypeGetDynamicSize() }
//    }
//    pub fn is_dynamic_stride_or_offset(s_or_o: i64) -> bool {
//        (unsafe { mlirShapedTypeIsDynamicStrideOrOffset(s_or_o) }) != 0
//    }
//    pub fn get_dynamic_stride_or_offset() -> i64 {
//        unsafe { mlirShapedTypeGetDynamicStrideOrOffset() }
//    }
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct VectorType<'ctx> {
//    pub ty: ShapedType<'ctx>,
//}
//
//impl<'ctx> VectorType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: ShapedType::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    pub fn get(shape: &[i64], elem_type: Type<'ctx>) -> Self {
//        Self::from_ffi(unsafe {
//            mlirVectorTypeGet(shape.len() as _, shape.as_ptr() as _, elem_type.to_ffi())
//        })
//    }
//    pub fn get_checked(loc: Location<'ctx>, shape: &[i64], elem_type: Type) -> Self {
//        Self::from_ffi(unsafe {
//            mlirVectorTypeGetChecked(
//                loc.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                elem_type.to_ffi(),
//            )
//        })
//    }
//    pub fn get_scalable(shape: &[i64], scalable: &[bool], elem_type: Type<'ctx>) -> Self {
//        Self::from_ffi(unsafe {
//            mlirVectorTypeGetScalable(
//                shape.len() as _,
//                shape.as_ptr() as _,
//                scalable.as_ptr() as _,
//                elem_type.to_ffi(),
//            )
//        })
//    }
//    pub fn get_scalable_checked(
//        loc: Location<'ctx>,
//        shape: &[i64],
//        scalable: &[bool],
//        elem_type: Type<'ctx>,
//    ) -> Self {
//        Self::from_ffi(unsafe {
//            mlirVectorTypeGetScalableChecked(
//                loc.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                scalable.as_ptr() as _,
//                elem_type.to_ffi(),
//            )
//        })
//    }
//    pub fn is_scalable(self) -> bool {
//        (unsafe { mlirVectorTypeIsScalable(self.to_ffi()) }) != 0
//    }
//    pub fn is_dim_scalable(self, dim: usize) -> bool {
//        (unsafe { mlirVectorTypeIsDimScalable(self.to_ffi(), dim as _) }) != 0
//    }
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct TensorType<'ctx> {
//    pub ty: ShapedType<'ctx>,
//}
//
//impl<'ctx> TensorType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: ShapedType::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    pub fn ranked_tensor_type_get(
//        shape: &[i64],
//        elem_type: Type<'ctx>,
//        encoding: Attr<'ctx>,
//    ) -> Self {
//        Self::from_ffi(unsafe {
//            mlirRankedTensorTypeGet(
//                shape.len() as _,
//                shape.as_ptr() as _,
//                elem_type.to_ffi(),
//                encoding.to_ffi(),
//            )
//        })
//    }
//    pub fn ranked_tensor_type_get_checked(
//        loc: Location<'ctx>,
//        shape: &[i64],
//        elem_type: Type<'ctx>,
//        encoding: Attr<'ctx>,
//    ) -> Self {
//        Self::from_ffi(unsafe {
//            mlirRankedTensorTypeGetChecked(
//                loc.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                elem_type.to_ffi(),
//                encoding.to_ffi(),
//            )
//        })
//    }
//    pub fn get_ranked_tensor_encoding(self) -> Attr<'ctx> {
//        Attr::from_ffi(unsafe { mlirRankedTensorTypeGetEncoding(self.to_ffi()) })
//    }
//    pub fn unranked_tensor_type_get(elem_ty: Type<'ctx>) -> Self {
//        Self::from_ffi(unsafe { mlirUnrankedTensorTypeGet(elem_ty.to_ffi()) })
//    }
//    pub fn unranked_tensor_type_get_checked(loc: Location<'ctx>, elem_ty: Type<'ctx>) -> Self {
//        Self::from_ffi(unsafe { mlirUnrankedTensorTypeGetChecked(loc.to_ffi(), elem_ty.to_ffi()) })
//    }
//}
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct MemRefType<'ctx> {
//    pub ty: ShapedType<'ctx>,
//}
//
//impl<'ctx> MemRefType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: ShapedType::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    pub fn ranked_get(elem_type: Type<'ctx>, shape: &[i64], layout: Attr, mem_space: Attr) -> Self {
//        Self::from_ffi(unsafe {
//            mlirMemRefTypeGet(
//                elem_type.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                layout.to_ffi(),
//                mem_space.to_ffi(),
//            )
//        })
//    }
//    pub fn ranked_get_checked(
//        loc: Location<'ctx>,
//        elem_type: Type<'ctx>,
//        shape: &[i64],
//        layout: Attr,
//        mem_space: Attr,
//    ) -> Self {
//        Self::from_ffi(unsafe {
//            mlirMemRefTypeGetChecked(
//                loc.to_ffi(),
//                elem_type.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                layout.to_ffi(),
//                mem_space.to_ffi(),
//            )
//        })
//    }
//    pub fn ranked_contiguous_get(elem_type: Type<'ctx>, shape: &[i64], mem_space: Attr) -> Self {
//        Self::from_ffi(unsafe {
//            mlirMemRefTypeContiguousGet(
//                elem_type.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                mem_space.to_ffi(),
//            )
//        })
//    }
//    pub fn ranked_contiguous_get_checked(
//        loc: Location<'ctx>,
//        elem_type: Type<'ctx>,
//        shape: &[i64],
//        mem_space: Attr,
//    ) -> Self {
//        Self::from_ffi(unsafe {
//            mlirMemRefTypeContiguousGetChecked(
//                loc.to_ffi(),
//                elem_type.to_ffi(),
//                shape.len() as _,
//                shape.as_ptr() as _,
//                mem_space.to_ffi(),
//            )
//        })
//    }
//    pub fn unranked_get(elem_type: Type<'ctx>, mem_space: Attr) -> Self {
//        Self::from_ffi(unsafe { mlirUnrankedMemRefTypeGet(elem_type.to_ffi(), mem_space.to_ffi()) })
//    }
//    pub fn unranked_get_checked(
//        loc: Location<'ctx>,
//        elem_type: Type<'ctx>,
//        mem_space: Attr,
//    ) -> Self {
//        Self::from_ffi(unsafe {
//            mlirUnrankedMemRefTypeGetChecked(loc.to_ffi(), elem_type.to_ffi(), mem_space.to_ffi())
//        })
//    }
//    pub fn get_layout(self) -> Attr<'ctx> {
//        Attr::from_ffi(unsafe { mlirMemRefTypeGetLayout(self.to_ffi()) })
//    }
//    // FIXME: get affine map
//    pub fn get_memory_space(self) -> Attr<'ctx> {
//        Attr::from_ffi(unsafe { mlirMemRefTypeGetMemorySpace(self.to_ffi()) })
//    }
//    pub fn get_strides_and_offset(self) -> (Vec<i64>, i64) {
//        let mut strides = Vec::<i64>::new();
//        strides.resize(self.ty.get_rank() as _, 0);
//        let mut offset = 0;
//        unsafe {
//            mlirMemRefTypeGetStridesAndOffset(self.to_ffi(), strides.as_mut_ptr(), &mut offset);
//        }
//        (strides, offset)
//    }
//    // FIXME: unranked memory space
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct TupleType<'ctx> {
//    pub ty: Type<'ctx>,
//}
//
//impl<'ctx> TupleType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: Type::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    // FIXME:
//    //pub fn get(ctx: &'ctx Context, elements: &[Type<'ctx>]) -> Self {
//    //    Self::from_ffi(unsafe {
//    //        mlirTupleTypeGet(ctx.to_ffi(), elements.len() as _, elements.as_ptr() as _)
//    //    })
//    //}
//    pub fn get_num_types(self) -> usize {
//        (unsafe { mlirTupleTypeGetNumTypes(self.to_ffi()) }) as _
//    }
//    pub fn get_type(self, pos: usize) -> Type<'ctx> {
//        Type::from_ffi(unsafe { mlirTupleTypeGetType(self.to_ffi(), pos as _) })
//    }
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct FunctionType<'ctx> {
//    pub ty: Type<'ctx>,
//}
//
//impl<'ctx> FunctionType<'ctx> {
//    pub fn from_ffi(handle: MlirType) -> Self {
//        Self {
//            ty: Type::from_ffi(handle),
//        }
//    }
//    pub fn to_ffi(self) -> MlirType {
//        self.ty.to_ffi()
//    }
//    // FIXME
//    //pub fn get(ctx: &'ctx Context, inputs: &[Type<'ctx>], results: &[Type<'ctx>]) -> Self {
//    //    FunctionType::from_ffi(unsafe {
//    //        mlirFunctionTypeGet(
//    //            ctx.to_ffi(),
//    //            inputs.len() as _,
//    //            inputs.as_ptr() as _,
//    //            results.len() as _,
//    //            results.as_ptr() as _,
//    //        )
//    //    })
//    //}
//    pub fn get_num_inputs(self) -> usize {
//        (unsafe { mlirFunctionTypeGetNumInputs(self.to_ffi()) }) as _
//    }
//    pub fn get_num_results(self) -> usize {
//        (unsafe { mlirFunctionTypeGetNumResults(self.to_ffi()) }) as _
//    }
//    pub fn get_input(self, pos: usize) -> Type<'ctx> {
//        Type::from_ffi(unsafe { mlirFunctionTypeGetInput(self.to_ffi(), pos as _) })
//    }
//    pub fn get_result(self, pos: usize) -> Type<'ctx> {
//        Type::from_ffi(unsafe { mlirFunctionTypeGetResult(self.to_ffi(), pos as _) })
//    }
//}
//
//#[repr(C)]
//#[derive(Copy, Clone)]
//pub struct OpaqueType<'ctx> {
//    pub ty: Type<'ctx>,
//}
