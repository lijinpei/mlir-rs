use crate::attribute::Attr;
use crate::context::*;
use crate::location::Location;
use crate::support::*;
use mlir_capi::BuiltinTypes::*;
use mlir_capi::IR::*;
use std::cmp::{Eq, PartialEq};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Type<'ctx> {
    pub handle: MlirType,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Type<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            handle: handle,
            phantom: PhantomData::default(),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.handle
    }
    pub fn parse(ctx: &'ctx Context, s: &str) -> Self {
        Self::from_ffi(unsafe { mlirTypeParseGet(ctx.to_ffi(), StrRef::from_str(s).to_ffi()) })
    }
    // FIXME: re-think about get_context api
    pub fn get_context(self) -> Context {
        Context::from_ffi(unsafe { mlirTypeGetContext(self.to_ffi()) })
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
            mlirTypePrint(
                self.to_ffi(),
                print_helper as _,
                &callback as *const &mut dyn PrintCallback as _,
            );
        }
    }
    pub fn dump(self) {
        unsafe {
            mlirTypeDump(self.to_ffi());
        }
    }
}

impl<'ctx> PartialEq for Type<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        (unsafe { mlirTypeEqual(self.to_ffi(), other.to_ffi()) }) != 0
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

impl<'ctx> Into<MlirType> for &Type<'ctx> {
    fn into(self) -> MlirType {
        self.handle
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IntegerType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> IntegerType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        IntegerType::from_ffi(unsafe { mlirIntegerTypeGet(ctx.to_ffi(), bitwidth) })
    }
    pub fn signed_get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        IntegerType::from_ffi(unsafe { mlirIntegerTypeSignedGet(ctx.to_ffi(), bitwidth) })
    }
    pub fn unsigned_get(ctx: &'ctx Context, bitwidth: u32) -> Self {
        IntegerType::from_ffi(unsafe { mlirIntegerTypeUnsignedGet(ctx.to_ffi(), bitwidth) })
    }
    pub fn get_width(self) -> u32 {
        unsafe { mlirIntegerTypeGetWidth(self.to_ffi()) }
    }
    pub fn is_signless(self) -> bool {
        (unsafe { mlirIntegerTypeIsSignless(self.to_ffi()) }) != 0
    }
    pub fn is_signed(self) -> bool {
        (unsafe { mlirIntegerTypeIsSigned(self.to_ffi()) }) != 0
    }
    pub fn is_unsigned(self) -> bool {
        (unsafe { mlirIntegerTypeIsUnsigned(self.to_ffi()) }) != 0
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IndexType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> IndexType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(ctx: &'ctx Context) -> Self {
        Self::from_ffi(unsafe { mlirIndexTypeGet(ctx.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FloatType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> FloatType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get_width(self) -> u32 {
        unsafe { mlirFloatTypeGetWidth(self.to_ffi()) }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NoneType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> NoneType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(ctx: &'ctx Context) -> Self {
        Self::from_ffi(unsafe { mlirNoneTypeGet(ctx.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ComplexType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> ComplexType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(elem_ty: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe { mlirComplexTypeGet(elem_ty.to_ffi()) })
    }
    pub fn get_element_type(self) -> Type<'ctx> {
        Type::from_ffi(unsafe { mlirComplexTypeGetElementType(self.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ShapedType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> ShapedType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get_element_type(self) -> Type<'ctx> {
        Type::from_ffi(unsafe { mlirShapedTypeGetElementType(self.to_ffi()) })
    }
    pub fn has_rank(self) -> bool {
        (unsafe { mlirShapedTypeHasRank(self.to_ffi()) }) != 0
    }
    pub fn get_rank(self) -> i64 {
        unsafe { mlirShapedTypeGetRank(self.to_ffi()) }
    }
    pub fn has_static_shape(self) -> bool {
        (unsafe { mlirShapedTypeHasStaticShape(self.to_ffi()) }) != 0
    }
    pub fn is_dynamic_dim(self, dim: usize) -> bool {
        (unsafe { mlirShapedTypeIsDynamicDim(self.to_ffi(), dim as _) }) != 0
    }
    pub fn get_dim_size(self, dim: usize) -> i64 {
        unsafe { mlirShapedTypeGetDimSize(self.to_ffi(), dim as _) }
    }
    pub fn is_dynamic_size(size: i64) -> bool {
        (unsafe { mlirShapedTypeIsDynamicSize(size) }) != 0
    }
    pub fn get_dynamic_size() -> i64 {
        unsafe { mlirShapedTypeGetDynamicSize() }
    }
    pub fn is_dynamic_stride_or_offset(s_or_o: i64) -> bool {
        (unsafe { mlirShapedTypeIsDynamicStrideOrOffset(s_or_o) }) != 0
    }
    pub fn get_dynamic_stride_or_offset() -> i64 {
        unsafe { mlirShapedTypeGetDynamicStrideOrOffset() }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VectorType<'ctx> {
    pub ty: ShapedType<'ctx>,
}

impl<'ctx> VectorType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: ShapedType::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(shape: &[i64], elem_type: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe {
            mlirVectorTypeGet(shape.len() as _, shape.as_ptr() as _, elem_type.to_ffi())
        })
    }
    pub fn get_checked(loc: Location<'ctx>, shape: &[i64], elem_type: Type) -> Self {
        Self::from_ffi(unsafe {
            mlirVectorTypeGetChecked(
                loc.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                elem_type.to_ffi(),
            )
        })
    }
    pub fn get_scalable(shape: &[i64], scalable: &[bool], elem_type: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe {
            mlirVectorTypeGetScalable(
                shape.len() as _,
                shape.as_ptr() as _,
                scalable.as_ptr() as _,
                elem_type.to_ffi(),
            )
        })
    }
    pub fn get_scalable_checked(
        loc: Location<'ctx>,
        shape: &[i64],
        scalable: &[bool],
        elem_type: Type<'ctx>,
    ) -> Self {
        Self::from_ffi(unsafe {
            mlirVectorTypeGetScalableChecked(
                loc.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                scalable.as_ptr() as _,
                elem_type.to_ffi(),
            )
        })
    }
    pub fn is_scalable(self) -> bool {
        (unsafe { mlirVectorTypeIsScalable(self.to_ffi()) }) != 0
    }
    pub fn is_dim_scalable(self, dim: usize) -> bool {
        (unsafe { mlirVectorTypeIsDimScalable(self.to_ffi(), dim as _) }) != 0
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TensorType<'ctx> {
    pub ty: ShapedType<'ctx>,
}

impl<'ctx> TensorType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: ShapedType::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn ranked_tensor_type_get(
        shape: &[i64],
        elem_type: Type<'ctx>,
        encoding: Attr<'ctx>,
    ) -> Self {
        Self::from_ffi(unsafe {
            mlirRankedTensorTypeGet(
                shape.len() as _,
                shape.as_ptr() as _,
                elem_type.to_ffi(),
                encoding.to_ffi(),
            )
        })
    }
    pub fn ranked_tensor_type_get_checked(
        loc: Location<'ctx>,
        shape: &[i64],
        elem_type: Type<'ctx>,
        encoding: Attr<'ctx>,
    ) -> Self {
        Self::from_ffi(unsafe {
            mlirRankedTensorTypeGetChecked(
                loc.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                elem_type.to_ffi(),
                encoding.to_ffi(),
            )
        })
    }
    pub fn get_ranked_tensor_encoding(self) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirRankedTensorTypeGetEncoding(self.to_ffi()) })
    }
    pub fn unranked_tensor_type_get(elem_ty: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe { mlirUnrankedTensorTypeGet(elem_ty.to_ffi()) })
    }
    pub fn unranked_tensor_type_get_checked(loc: Location<'ctx>, elem_ty: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe { mlirUnrankedTensorTypeGetChecked(loc.to_ffi(), elem_ty.to_ffi()) })
    }
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MemRefType<'ctx> {
    pub ty: ShapedType<'ctx>,
}

impl<'ctx> MemRefType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: ShapedType::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn ranked_get(elem_type: Type<'ctx>, shape: &[i64], layout: Attr, mem_space: Attr) -> Self {
        Self::from_ffi(unsafe {
            mlirMemRefTypeGet(
                elem_type.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                layout.to_ffi(),
                mem_space.to_ffi(),
            )
        })
    }
    pub fn ranked_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        shape: &[i64],
        layout: Attr,
        mem_space: Attr,
    ) -> Self {
        Self::from_ffi(unsafe {
            mlirMemRefTypeGetChecked(
                loc.to_ffi(),
                elem_type.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                layout.to_ffi(),
                mem_space.to_ffi(),
            )
        })
    }
    pub fn ranked_contiguous_get(elem_type: Type<'ctx>, shape: &[i64], mem_space: Attr) -> Self {
        Self::from_ffi(unsafe {
            mlirMemRefTypeContiguousGet(
                elem_type.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                mem_space.to_ffi(),
            )
        })
    }
    pub fn ranked_contiguous_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        shape: &[i64],
        mem_space: Attr,
    ) -> Self {
        Self::from_ffi(unsafe {
            mlirMemRefTypeContiguousGetChecked(
                loc.to_ffi(),
                elem_type.to_ffi(),
                shape.len() as _,
                shape.as_ptr() as _,
                mem_space.to_ffi(),
            )
        })
    }
    pub fn unranked_get(elem_type: Type<'ctx>, mem_space: Attr) -> Self {
        Self::from_ffi(unsafe { mlirUnrankedMemRefTypeGet(elem_type.to_ffi(), mem_space.to_ffi()) })
    }
    pub fn unranked_get_checked(
        loc: Location<'ctx>,
        elem_type: Type<'ctx>,
        mem_space: Attr,
    ) -> Self {
        Self::from_ffi(unsafe {
            mlirUnrankedMemRefTypeGetChecked(loc.to_ffi(), elem_type.to_ffi(), mem_space.to_ffi())
        })
    }
    pub fn get_layout(self) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirMemRefTypeGetLayout(self.to_ffi()) })
    }
    // FIXME: get affine map
    pub fn get_memory_space(self) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirMemRefTypeGetMemorySpace(self.to_ffi()) })
    }
    pub fn get_strides_and_offset(self) -> (Vec<i64>, i64) {
        let mut strides = Vec::<i64>::new();
        strides.resize(self.ty.get_rank() as _, 0);
        let mut offset = 0;
        unsafe {
            mlirMemRefTypeGetStridesAndOffset(self.to_ffi(), strides.as_mut_ptr(), &mut offset);
        }
        (strides, offset)
    }
    // FIXME: unranked memory space
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TupleType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> TupleType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(ctx: &'ctx Context, elements: &[Type<'ctx>]) -> Self {
        Self::from_ffi(unsafe {
            mlirTupleTypeGet(ctx.to_ffi(), elements.len() as _, elements.as_ptr() as _)
        })
    }
    pub fn get_num_types(self) -> usize {
        (unsafe { mlirTupleTypeGetNumTypes(self.to_ffi()) }) as _
    }
    pub fn get_type(self, pos: usize) -> Type<'ctx> {
        Type::from_ffi(unsafe { mlirTupleTypeGetType(self.to_ffi(), pos as _) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FunctionType<'ctx> {
    pub ty: Type<'ctx>,
}

impl<'ctx> FunctionType<'ctx> {
    pub fn from_ffi(handle: MlirType) -> Self {
        Self {
            ty: Type::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirType {
        self.ty.to_ffi()
    }
    pub fn get(ctx: &'ctx Context, inputs: &[Type<'ctx>], results: &[Type<'ctx>]) -> Self {
        FunctionType::from_ffi(unsafe {
            mlirFunctionTypeGet(
                ctx.to_ffi(),
                inputs.len() as _,
                inputs.as_ptr() as _,
                results.len() as _,
                results.as_ptr() as _,
            )
        })
    }
    pub fn get_num_inputs(self) -> usize {
        (unsafe { mlirFunctionTypeGetNumInputs(self.to_ffi()) }) as _
    }
    pub fn get_num_results(self) -> usize {
        (unsafe { mlirFunctionTypeGetNumResults(self.to_ffi()) }) as _
    }
    pub fn get_input(self, pos: usize) -> Type<'ctx> {
        Type::from_ffi(unsafe { mlirFunctionTypeGetInput(self.to_ffi(), pos as _) })
    }
    pub fn get_result(self, pos: usize) -> Type<'ctx> {
        Type::from_ffi(unsafe { mlirFunctionTypeGetResult(self.to_ffi(), pos as _) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct OpaqueType<'ctx> {
    pub ty: Type<'ctx>,
}
