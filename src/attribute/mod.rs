use crate::context::Context;
use crate::location::Location;
use crate::r#type::Type;
use crate::support::StrRef;
use mlir_capi::BuiltinAttributes::*;
use mlir_capi::IR::MlirAttribute;
use std::convert::{From, Into};
use std::marker::PhantomData;
use std::vec::Vec;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Attr<'ctx> {
    pub handle: MlirAttribute,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> From<MlirAttribute> for Attr<'ctx> {
    fn from(value: MlirAttribute) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirAttribute> for &Attr<'ctx> {
    fn into(self) -> MlirAttribute {
        self.handle
    }
}

impl<'ctx> Attr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.handle
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Attr {
            handle,
            phantom: PhantomData::default(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocationAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AffineMapAttr {
    pub handle: MlirAttribute,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArrayAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> ArrayAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn len(self) -> usize {
        unsafe { mlirArrayAttrGetNumElements(self.attr.handle) as _ }
    }
    pub fn get(self, idx: usize) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirArrayAttrGetElement(self.to_ffi(), idx as _) })
    }
    pub fn from_attr_slice(ctx: &'ctx Context, attrs: &[Attr]) -> Self {
        Self::from_ffi(unsafe {
            mlirArrayAttrGet(ctx.to_ffi(), attrs.len() as _, attrs.as_ptr() as _)
        })
    }
    pub fn from_slice_of<'a, T>(ctx: &'ctx Context, attrs: &'a [T]) -> Self
    where
        &'a T: Into<Attr<'ctx>>,
    {
        let vec: Vec<Attr> = attrs.iter().map(|x| Into::<Attr>::into(x)).collect();
        Self::from_ffi(unsafe { mlirArrayAttrGet(ctx.handle, vec.len() as _, vec.as_ptr() as _) })
    }
    pub fn to_attr_vec(self) -> Vec<Attr<'ctx>> {
        let num_elems = unsafe { mlirArrayAttrGetNumElements(self.to_ffi()) };
        let mut vec = Vec::new();
        vec.reserve_exact(num_elems as _);
        for i in 0..num_elems {
            let handle = unsafe { mlirArrayAttrGetElement(self.to_ffi(), i) };
            vec.push(Attr::from_ffi(handle));
        }
        vec
    }
    pub fn to_vec_of<T: From<Attr<'ctx>>>(self) -> Vec<T> {
        let num_elems = unsafe { mlirArrayAttrGetNumElements(self.to_ffi()) };
        let mut vec = Vec::new();
        vec.reserve_exact(num_elems as _);
        for i in 0..num_elems {
            let handle = unsafe { mlirArrayAttrGetElement(self.to_ffi(), i) };
            let attr = Attr::from_ffi(handle);
            vec.push(From::<Attr>::from(attr));
        }
        vec
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DictionaryAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> DictionaryAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }

    pub fn len(self) -> usize {
        unsafe { mlirDictionaryAttrGetNumElements(self.to_ffi()) as _ }
    }
    pub fn get(self, pos: usize) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirDictionaryAttrGetElement(self.to_ffi(), pos as _).attribute })
    }
    pub fn get_by_name(self, name: &str) -> Attr<'ctx> {
        let name_ref = StrRef::from_str(name);
        Attr::from_ffi(unsafe {
            mlirDictionaryAttrGetElementByName(self.to_ffi(), name_ref.handle)
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FloatAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> FloatAttr<'ctx> {
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn f64_get(ctx: &'ctx Context, val: f64, ty: Type) -> Self {
        Self::from_ffi(unsafe { mlirFloatAttrDoubleGet(ctx.to_ffi(), ty.to_ffi(), val) })
    }
    pub fn f64_get_checked(loc: Location, val: f64, ty: Type) -> Self {
        Self::from_ffi(unsafe { mlirFloatAttrDoubleGetChecked(loc.to_ffi(), ty.to_ffi(), val) })
    }
    pub fn get_value_f64(self) -> f64 {
        unsafe { mlirFloatAttrGetValueDouble(self.to_ffi()) }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IntegerAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> IntegerAttr<'ctx> {
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn get(ty: Type, val: i64) -> Self {
        Self::from_ffi(unsafe { mlirIntegerAttrGet(ty.to_ffi(), val) })
    }
    pub fn get_val_int(self) -> i64 {
        unsafe { mlirIntegerAttrGetValueInt(self.to_ffi()) as _ }
    }
    pub fn get_val_sint(self) -> i64 {
        unsafe { mlirIntegerAttrGetValueSInt(self.to_ffi()) as _ }
    }
    pub fn get_val_uint(self) -> u64 {
        unsafe { mlirIntegerAttrGetValueUInt(self.to_ffi()) as _ }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BoolAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> BoolAttr<'ctx> {
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn get(ctx: &'ctx Context, val: bool) -> Self {
        let i_val: i32 = if val { 1 } else { 0 };
        Self::from_ffi(unsafe { mlirBoolAttrGet(ctx.to_ffi(), i_val) })
    }
    pub fn get_val(self) -> bool {
        let val = unsafe { mlirBoolAttrGetValue(self.to_ffi()) };
        val != 0
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IntegerSetAttr {
    pub handle: MlirAttribute,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct OpaqueAttr {
    pub handle: MlirAttribute,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct StringAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> StringAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn get(ctx: &'ctx Context, str_ref: StrRef) -> Self {
        Self::from_ffi(unsafe { mlirStringAttrGet(ctx.to_ffi(), str_ref.to_ffi()) })
    }
    pub fn typed_get(ty: Type<'ctx>, str_ref: StrRef) -> Self {
        Self::from_ffi(unsafe { mlirStringAttrTypedGet(ty.to_ffi(), str_ref.to_ffi()) })
    }
    pub fn get_value(self) -> StrRef<'ctx> {
        StrRef::from_ffi(unsafe { mlirStringAttrGetValue(self.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SymbolRefAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> SymbolRefAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn get(ctx: &'ctx Context, symbol: StrRef, refs: &[Attr]) -> Self {
        Self::from_ffi(unsafe {
            mlirSymbolRefAttrGet(
                ctx.to_ffi(),
                symbol.to_ffi(),
                refs.len() as _,
                refs.as_ptr() as _,
            )
        })
    }
    pub fn get_root_ref(self) -> StrRef<'ctx> {
        StrRef::from_ffi(unsafe { mlirSymbolRefAttrGetRootReference(self.to_ffi()) })
    }
    pub fn get_leaf_ref(self) -> StrRef<'ctx> {
        StrRef::from_ffi(unsafe { mlirSymbolRefAttrGetLeafReference(self.to_ffi()) })
    }
    pub fn get_num_nested_ref(self) -> usize {
        (unsafe { mlirSymbolRefAttrGetNumNestedReferences(self.to_ffi()) }) as _
    }
    pub fn get_nested_ref(self, pos: usize) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirSymbolRefAttrGetNestedReference(self.to_ffi(), pos as _) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct FlatSymbolRefAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> FlatSymbolRefAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn get(ctx: &'ctx Context, str_ref: StrRef) -> Self {
        Self::from_ffi(unsafe { mlirFlatSymbolRefAttrGet(ctx.to_ffi(), str_ref.to_ffi()) })
    }
    pub fn get_value(self) -> StrRef<'ctx> {
        StrRef::from_ffi(unsafe { mlirFlatSymbolRefAttrGetValue(self.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TypeAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> TypeAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn get(ty: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe { mlirTypeAttrGet(ty.to_ffi()) })
    }
    pub fn get_value(self) -> Type<'ctx> {
        Type::from_ffi(unsafe { mlirTypeAttrGetValue(self.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct UnitAttr<'ctx> {
    pub attr: Attr<'ctx>,
}
impl<'ctx> UnitAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn get(ctx: &'ctx Context) -> Self {
        Self::from_ffi(unsafe { mlirUnitAttrGet(ctx.to_ffi()) })
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ElementsAttr<'ctx> {
    pub attr: Attr<'ctx>,
}
impl<'ctx> ElementsAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn is_valid_index(self, indices: &[usize]) -> bool {
        (unsafe {
            mlirElementsAttrIsValidIndex(
                self.to_ffi(),
                indices.len() as _,
                indices.as_ptr() as *mut _,
            )
        }) != 0
    }
    pub fn get_value(self, indices: &[usize]) -> Attr<'ctx> {
        Attr::from_ffi(unsafe {
            mlirElementsAttrGetValue(
                self.to_ffi(),
                indices.len() as _,
                indices.as_ptr() as *mut _,
            )
        })
    }
    pub fn get_num_elements(self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.to_ffi()) }) as _
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DenseArrrayAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> DenseArrrayAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn get_num_elements(self) -> usize {
        (unsafe { mlirDenseArrayGetNumElements(self.to_ffi()) }) as _
    }
    pub fn get_bool_arr(ctx: &'ctx Context, values: &[std::ffi::c_int]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseBoolArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn get_i8_arr(ctx: &'ctx Context, values: &[i8]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseI8ArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn get_i16_arr(ctx: &'ctx Context, values: &[i16]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseI16ArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn get_i32_arr(ctx: &'ctx Context, values: &[i32]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseI32ArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn get_i64_arr(ctx: &'ctx Context, values: &[i64]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseI64ArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn get_f32_arr(ctx: &'ctx Context, values: &[f32]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseF32ArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn get_f64_arr(ctx: &'ctx Context, values: &[f64]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseF64ArrayGet(ctx.to_ffi(), values.len() as _, values.as_ptr() as _)
        })
    }
    pub fn bool_get_elem(self, pos: usize) -> bool {
        (unsafe { mlirDenseBoolArrayGetElement(self.to_ffi(), pos as _) }) != 0
    }
    pub fn i8_get_elem(self, pos: usize) -> i8 {
        (unsafe { mlirDenseI8ArrayGetElement(self.to_ffi(), pos as _) }) as _
    }
    pub fn i16_get_elem(self, pos: usize) -> i16 {
        (unsafe { mlirDenseI16ArrayGetElement(self.to_ffi(), pos as _) }) as _
    }
    pub fn i32_get_elem(self, pos: usize) -> i32 {
        (unsafe { mlirDenseI32ArrayGetElement(self.to_ffi(), pos as _) }) as _
    }
    pub fn i64_get_elem(self, pos: usize) -> i64 {
        (unsafe { mlirDenseI64ArrayGetElement(self.to_ffi(), pos as _) }) as _
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DenseElementsAttr<'ctx> {
    pub attr: Attr<'ctx>,
}

impl<'ctx> DenseElementsAttr<'ctx> {
    pub fn to_ffi(self) -> MlirAttribute {
        self.attr.to_ffi()
    }
    pub fn from_ffi(handle: MlirAttribute) -> Self {
        Self {
            attr: Attr::from_ffi(handle),
        }
    }
    pub fn is_int_elements(self) -> bool {
        (unsafe { mlirAttributeIsADenseIntElements(self.to_ffi()) }) != 0
    }
    pub fn is_fp_elements(self) -> bool {
        (unsafe { mlirAttributeIsADenseFPElements(self.to_ffi()) }) != 0
    }
    pub fn get(shaped_type: Type<'ctx>, elements: &[Attr<'ctx>]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrGet(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn raw_buffer_get(shaped_type: Type<'ctx>, elements: &[std::ffi::c_void]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrRawBufferGet(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn splat_get(shaped_type: Type<'ctx>, attr: Attr<'ctx>) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrSplatGet(shaped_type.to_ffi(), attr.to_ffi())
        })
    }
    pub fn bool_splat_get(shaped_type: Type<'ctx>, val: bool) -> Self {
        let i32_val = if val { 1 } else { 0 };
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrBoolSplatGet(shaped_type.to_ffi(), i32_val as _)
        })
    }
    pub fn u8_splat_get(shaped_type: Type<'ctx>, val: u8) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrUInt8SplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn i8_splat_get(shaped_type: Type<'ctx>, val: i8) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrInt8SplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn u32_splat_get(shaped_type: Type<'ctx>, val: u32) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrUInt32SplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn i32_splat_get(shaped_type: Type<'ctx>, val: i32) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrInt32SplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn u64_splat_get(shaped_type: Type<'ctx>, val: u64) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrUInt64SplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn i64_splat_get(shaped_type: Type<'ctx>, val: i64) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrInt64SplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn f32_splat_get(shaped_type: Type<'ctx>, val: f32) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrFloatSplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn f64_splat_get(shaped_type: Type<'ctx>, val: f64) -> Self {
        Self::from_ffi(unsafe { mlirDenseElementsAttrDoubleSplatGet(shaped_type.to_ffi(), val) })
    }
    pub fn bool_get(shaped_type: Type<'ctx>, elements: &[std::ffi::c_int]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrBoolGet(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn u8_get(shaped_type: Type<'ctx>, elements: &[u8]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrUInt8Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn i8_get(shaped_type: Type<'ctx>, elements: &[i8]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrInt8Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn u16_get(shaped_type: Type<'ctx>, elements: &[u16]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrUInt16Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn i16_get(shaped_type: Type<'ctx>, elements: &[i16]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrInt16Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn u32_get(shaped_type: Type<'ctx>, elements: &[u32]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrUInt32Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn i32_get(shaped_type: Type<'ctx>, elements: &[i32]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrInt32Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn u64_get(shaped_type: Type<'ctx>, elements: &[u64]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrUInt64Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn i64_get(shaped_type: Type<'ctx>, elements: &[i64]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrInt64Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn bf16_get(shaped_type: Type<'ctx>, elements: &[u16]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrBFloat16Get(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn f32_get(shaped_type: Type<'ctx>, elements: &[f32]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrFloatGet(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn f64_get(shaped_type: Type<'ctx>, elements: &[f64]) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrDoubleGet(
                shaped_type.to_ffi(),
                elements.len() as _,
                elements.as_ptr() as _,
            )
        })
    }
    pub fn str_get(shaped_type: Type<'ctx>, elements: &[&str]) -> Self {
        let strs: Vec<_> = elements.iter().map(|x| StrRef::from_str(x)).collect();
        Self::from_ffi({
            unsafe {
                mlirDenseElementsAttrStringGet(
                    shaped_type.to_ffi(),
                    strs.len() as _,
                    strs.as_ptr() as _,
                )
            }
        })
    }
    pub fn reshape_get(self, shaped_type: Type<'ctx>) -> Self {
        Self::from_ffi(unsafe {
            mlirDenseElementsAttrReshapeGet(self.to_ffi(), shaped_type.to_ffi())
        })
    }
    pub fn is_splat(self) -> bool {
        (unsafe { mlirDenseElementsAttrIsSplat(self.to_ffi()) }) != 0
    }
    pub fn get_splat_value(self) -> Attr<'ctx> {
        Attr::from_ffi(unsafe { mlirDenseElementsAttrGetSplatValue(self.to_ffi()) })
    }
    pub fn get_splat_bool(self) -> bool {
        (unsafe { mlirDenseElementsAttrGetBoolSplatValue(self.to_ffi()) }) != 0
    }
    pub fn get_splat_i8(self) -> i8 {
        unsafe { mlirDenseElementsAttrGetInt8SplatValue(self.to_ffi()) }
    }
    pub fn get_splat_u8(self) -> u8 {
        unsafe { mlirDenseElementsAttrGetUInt8SplatValue(self.to_ffi()) }
    }
    pub fn get_splat_i32(self) -> i32 {
        unsafe { mlirDenseElementsAttrGetInt32SplatValue(self.to_ffi()) }
    }
    pub fn get_splat_u32(self) -> u32 {
        unsafe { mlirDenseElementsAttrGetUInt32SplatValue(self.to_ffi()) }
    }
    pub fn get_splat_i64(self) -> i64 {
        unsafe { mlirDenseElementsAttrGetInt64SplatValue(self.to_ffi()) }
    }
    pub fn get_splat_u64(self) -> u64 {
        unsafe { mlirDenseElementsAttrGetUInt64SplatValue(self.to_ffi()) }
    }
    pub fn get_splat_f32(self) -> f32 {
        unsafe { mlirDenseElementsAttrGetFloatSplatValue(self.to_ffi()) }
    }
    pub fn get_splat_f64(self) -> f64 {
        unsafe { mlirDenseElementsAttrGetDoubleSplatValue(self.to_ffi()) }
    }
    pub fn get_splat_str(self) -> &'ctx str {
        StrRef::from_ffi(unsafe { mlirDenseElementsAttrGetStringSplatValue(self.to_ffi()) })
            .to_str()
    }
    pub fn get_bool(self, pos: usize) -> bool {
        (unsafe { mlirDenseElementsAttrGetBoolValue(self.to_ffi(), pos as _) }) != 0
    }
    pub fn get_i8(self, pos: usize) -> i8 {
        unsafe { mlirDenseElementsAttrGetInt8Value(self.to_ffi(), pos as _) }
    }
    pub fn get_u8(self, pos: usize) -> u8 {
        unsafe { mlirDenseElementsAttrGetUInt8Value(self.to_ffi(), pos as _) }
    }
    pub fn get_i16(self, pos: usize) -> i16 {
        unsafe { mlirDenseElementsAttrGetInt16Value(self.to_ffi(), pos as _) }
    }
    pub fn get_u16(self, pos: usize) -> u16 {
        unsafe { mlirDenseElementsAttrGetUInt16Value(self.to_ffi(), pos as _) }
    }
    pub fn get_i32(self, pos: usize) -> i32 {
        unsafe { mlirDenseElementsAttrGetInt32Value(self.to_ffi(), pos as _) }
    }
    pub fn get_u32(self, pos: usize) -> u32 {
        unsafe { mlirDenseElementsAttrGetUInt32Value(self.to_ffi(), pos as _) }
    }
    pub fn get_i64(self, pos: usize) -> i64 {
        unsafe { mlirDenseElementsAttrGetInt64Value(self.to_ffi(), pos as _) }
    }
    pub fn get_u64(self, pos: usize) -> u64 {
        unsafe { mlirDenseElementsAttrGetUInt64Value(self.to_ffi(), pos as _) }
    }
    pub fn get_f32(self, pos: usize) -> f32 {
        unsafe { mlirDenseElementsAttrGetFloatValue(self.to_ffi(), pos as _) }
    }
    pub fn get_f64(self, pos: usize) -> f64 {
        unsafe { mlirDenseElementsAttrGetDoubleValue(self.to_ffi(), pos as _) }
    }
    pub fn get_str(self, pos: usize) -> &'ctx str {
        StrRef::from_ffi(unsafe { mlirDenseElementsAttrGetStringValue(self.to_ffi(), pos as _) })
            .to_str()
    }
    pub fn get_raw_data(self) -> *const std::ffi::c_void {
        unsafe { mlirDenseElementsAttrGetRawData(self.to_ffi()) }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ResourceBlobAttr {
    pub handle: MlirAttribute,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SparseElementsAttr {
    pub handle: MlirAttribute,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct StridedLayoutAttr {
    pub handle: MlirAttribute,
}
