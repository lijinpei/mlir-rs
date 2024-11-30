use crate::affine_map::*;
use crate::common::*;
use crate::context::*;
use crate::dialect::*;
use crate::integer_set::*;
use crate::location::*;
use crate::r#type::*;
use crate::support::*;
use crate::type_cast::*;

use mlir_capi::BuiltinAttributes;
use mlir_capi::IR;
use mlir_capi::IR::{MlirAttribute, MlirContext, MlirIdentifier, MlirNamedAttribute};

use std::convert::{From, Into};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Attr<'ctx> {
    pub handle: MlirAttribute,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> HandleWithContext<'ctx> for Attr<'ctx> {
    type HandleTy = MlirAttribute;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { IR::FFIVal_::mlirAttributeGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> Into<MlirAttribute> for Attr<'ctx> {
    fn into(self) -> MlirAttribute {
        self.handle
    }
}

impl<'ctx> Attr<'ctx> {
    pub fn parse(ctx: &'ctx Context, attr: &str) -> Self {
        let handle = unsafe { IR::FFIVal_::mlirAttributeParseGet(ctx, StrRef::from(attr)) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
}

pub trait AttrTrait<'ctx>: Into<MlirAttribute> + HandleWithContext<'ctx> + Copy {
    fn get_type(self) -> Type<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirAttributeGetType(self) };
        Type::from_handle_same_context(handle, &self)
    }

    fn is_a_elements_attr(self) -> bool {
        to_rbool(unsafe { BuiltinAttributes::FFIVal_::mlirAttributeIsAElements(self) })
    }

    fn get_typeid(self) -> TypeID {
        unsafe { IR::FFIVal_::mlirAttributeGetTypeID(self) }
    }
    fn get_dialect(self) -> Dialect<'ctx> {
        unsafe { IR::FFIVal_::mlirAttributeGetDialect(self) }
    }
    fn print(self, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirAttributePrint(
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
        let mut printer = PrintToFormatter::new(formatter);
        self.print(&mut printer);
        Ok(())
    }
    fn dump(self) {
        unsafe {
            IR::FFIVoid_::mlirAttributeDump(self);
        }
    }
}

impl<'ctx> AttrTrait<'ctx> for Attr<'ctx> {}

impl<'ctx> NullableRef for Attr<'ctx> {
    fn is_null(self) -> bool {
        self.handle.ptr == std::ptr::null()
    }
    fn create_null() -> Self {
        Attr {
            handle: MlirAttribute {
                ptr: std::ptr::null(),
            },
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> PartialEq for Attr<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        to_rbool(unsafe { IR::FFIVal_::mlirAttributeEqual(*self, *other) })
    }
}
impl<'ctx> Eq for Attr<'ctx> {}

impl<'ctx> Debug for Attr<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> Display for Attr<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

#[cfg(test)]
pub mod attr_test {
    use super::*;

    pub fn create_some_attrs<'ctx>(ctx: &'ctx Context) -> Vec<Attr<'ctx>> {
        // FIXME: float attr of inf/nan
        [
            "affine_map<(d0) -> (d0)>",
            "[10, i32]",
            "array<f32: 1.0, -2.0>",
            "dense<[1.0, -1.0, 2., -2.]>: tensor<4xf16>",
            "0xDEAD : f16",
        ]
        .iter()
        .map(|x| Attr::parse(ctx, x))
        .collect()
    }

    pub fn create_some_named_attrs<'ctx>(ctx: &'ctx Context) -> Vec<NamedAttr<'ctx>> {
        let attrs = create_some_attrs(ctx);
        let mut index = 0;
        attrs
            .into_iter()
            .map(|x| {
                let name = format!("name_{}", index);
                let name = Identifier::get(ctx, &name);
                index += 1;
                NamedAttr {
                    name: name,
                    attribute: x,
                }
            })
            .collect()
    }

    #[test]
    fn test_null() {
        let a = Attr::create_null();
        assert!(a.is_null());
    }

    #[test]
    fn test_parse_get_type_print_eq() {
        let ctx = Context::create();
        ctx.set_allow_unregistered_dialects(true);
        let affine_map = Attr::parse(&ctx, "affine_map<(d0) -> (d0)>");
        assert!(IsA::<AffineMapAttr>::is_a_non_null(affine_map));
        affine_map.get_type();
        println!("{}", affine_map);
        let array = Attr::parse(&ctx, "[]");
        assert!(IsA::<ArrayAttr>::is_a_non_null(array));
        array.get_type();
        println!("{}", array);
        let dense_array = Attr::parse(&ctx, "array<f64: 42., 12.>");
        assert!(IsA::<DenseF64ArrayAttr>::is_a_non_null(dense_array));
        dense_array.get_type();
        println!("{}", dense_array);
        let dense_elements = Attr::parse(&ctx, "dense<[10.0, 11.0]> : tensor<2xf32>");
        assert!(IsA::<DenseFPElementsAttr>::is_a_non_null(dense_elements));
        dense_elements.get_type();
        println!("{}", dense_elements);
        // FIXME: how to parse dense resource
        // FIXME: not dense elements string  attr
        let dense_string = Attr::parse(
            &ctx,
            "dense<[\"example1\", \"example2\"]> : tensor<2x!foo.string>",
        );
        assert!(!dense_string.is_null());
        println!("{}", dense_string);
        let dict_attr = Attr::parse(
            &ctx,
            "{int_attr = 10, \"string attr name\" = \"string attribute\"}",
        );
        assert!(IsA::<DictionaryAttr>::is_a_non_null(dict_attr));
        dict_attr.get_type();
        println!("{}", dict_attr);
        let float_attr = Attr::parse(&ctx, "0x7CFF : f16");
        assert!(IsA::<FloatAttr>::is_a_non_null(float_attr));
        float_attr.get_type();
        println!("{}", float_attr);
        let int_attr = Attr::parse(&ctx, "true");
        assert!(IsA::<IntegerAttr>::is_a_non_null(int_attr));
        int_attr.get_type();
        println!("{}", int_attr);
        let int_set_attr = Attr::parse(&ctx, "affine_set<(d0) : (d0 - 2 >= 0)>");
        assert!(IsA::<IntegerSetAttr>::is_a_non_null(int_set_attr));
        int_set_attr.get_type();
        println!("{}", int_set_attr);
        let opaque_attr = Attr::parse(&ctx, "#foobar_dialect<\"opaque attribute data\">");
        assert!(IsA::<OpaqueAttr>::is_a_non_null(opaque_attr));
        opaque_attr.get_type();
        println!("{}", opaque_attr);
        let sparse_elem_attr =
            Attr::parse(&ctx, "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>");
        assert!(IsA::<SparseElementsAttr>::is_a_non_null(sparse_elem_attr));
        sparse_elem_attr.get_type();
        println!("{}", sparse_elem_attr);
        let str_attr = Attr::parse(&ctx, "\"An important string\"");
        assert!(IsA::<StringAttr>::is_a_non_null(str_attr));
        str_attr.get_type();
        println!("{}", str_attr);
        let flat_sym_ref = Attr::parse(&ctx, "@flat_reference");
        assert!(IsA::<FlatSymbolRefAttr>::is_a_non_null(flat_sym_ref));
        assert!(IsA::<SymbolRefAttr>::is_a_non_null(flat_sym_ref));
        flat_sym_ref.get_type();
        println!("{}", flat_sym_ref);
        let nested_sym_ref = Attr::parse(&ctx, "@parent_reference::@nested_reference");
        assert!(!IsA::<FlatSymbolRefAttr>::is_a_non_null(nested_sym_ref));
        assert!(IsA::<SymbolRefAttr>::is_a_non_null(nested_sym_ref));
        nested_sym_ref.get_type();
        println!("{}", nested_sym_ref);
        let type_attr = Attr::parse(&ctx, "i32");
        assert!(IsA::<TypeAttr>::is_a_non_null(type_attr));
        type_attr.get_type();
        println!("{}", type_attr);
        let unit_attr = Attr::parse(&ctx, "unit");
        assert!(IsA::<UnitAttr>::is_a_non_null(unit_attr));
        unit_attr.get_type();
        println!("{}", unit_attr);
        let strided_attr = Attr::parse(&ctx, "strided<[7, 2]>");
        assert!(IsA::<StridedLayoutAttr>::is_a_non_null(strided_attr));
        strided_attr.get_type();
        println!("{}", strided_attr);
        let loc_attr1 = Attr::parse(&ctx, "loc(callsite(\"foo\" at \"mysource.cc\":10:8))");
        assert!(IsA::<LocationAttr>::is_a_non_null(loc_attr1));
        loc_attr1.get_type();
        println!("{}", loc_attr1);
        let loc_attr2 = Attr::parse(&ctx, "loc(\"mysource.cc\":10:8 to 12:18)");
        assert!(IsA::<LocationAttr>::is_a_non_null(loc_attr2));
        loc_attr2.get_type();
        println!("{}", loc_attr2);
        let loc_attr3 = Attr::parse(
            &ctx,
            "loc(fused<\"CSE\">[\"mysource.cc\":10:8, \"mysource.cc\":22:8])",
        );
        assert!(IsA::<LocationAttr>::is_a_non_null(loc_attr3));
        loc_attr3.get_type();
        println!("{}", loc_attr3);
        let loc_attr4 = Attr::parse(&ctx, "loc(\"CSE\"(\"mysource.cc\":10:8))");
        assert!(IsA::<LocationAttr>::is_a_non_null(loc_attr4));
        loc_attr4.get_type();
        println!("{}", loc_attr4);
        let loc_attr5 = Attr::parse(&ctx, "loc(\"mysource\")");
        assert!(IsA::<LocationAttr>::is_a_non_null(loc_attr5));
        loc_attr5.get_type();
        println!("{}", loc_attr5);
        let loc_attr6 = Attr::parse(&ctx, "loc(unknown)");
        assert!(IsA::<LocationAttr>::is_a_non_null(loc_attr6));
        loc_attr6.get_type();
        println!("{}", loc_attr6);
        let dist_attr = Attr::parse(&ctx, "distinct[0]<42.0 : f32>");
        // FIXME: no dist attr
        assert!(!dist_attr.is_null());
        dist_attr.get_type();
        println!("{}", dist_attr);

        let attrs: &[Attr] = &[
            affine_map.into(),
            array.into(),
            dense_array.into(),
            dense_elements.into(),
            dense_string.into(),
            dict_attr.into(),
            float_attr.into(),
            int_attr.into(),
            int_set_attr.into(),
            opaque_attr.into(),
            sparse_elem_attr.into(),
            str_attr.into(),
            flat_sym_ref.into(),
            nested_sym_ref.into(),
            type_attr.into(),
            unit_attr.into(),
            strided_attr.into(),
            loc_attr1.into(),
            loc_attr2.into(),
            loc_attr3.into(),
            loc_attr4.into(),
            loc_attr5.into(),
            loc_attr6.into(),
        ];
        let num_attrs = attrs.len();
        for i in 0..num_attrs {
            for j in 0..num_attrs {
                assert_eq!(attrs[i] == attrs[j], i == j);
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Identifier<'ctx> {
    pub handle: MlirIdentifier,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> HandleWithContext<'ctx> for Identifier<'ctx> {
    type HandleTy = MlirIdentifier;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { IR::FFIVal_::mlirIdentifierGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> Into<MlirIdentifier> for Identifier<'ctx> {
    fn into(self) -> MlirIdentifier {
        self.handle
    }
}

impl<'ctx> Debug for Identifier<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(formatter, "id(\"{:?}\")", self.str())?;
        Ok(())
    }
}

impl<'ctx> Display for Identifier<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(formatter, "id(\"{}\")", self.str())?;
        Ok(())
    }
}

impl<'ctx> Identifier<'ctx> {
    pub fn get(ctx: &'ctx Context, s: &str) -> Self {
        let s_ref: StrRef = s.into();
        let handle = unsafe { IR::FFIVal_::mlirIdentifierGet(ctx, s_ref) };
        unsafe { Self::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn str(self) -> &'ctx str {
        let str_ref: StrRef = unsafe { IR::FFIVal_::mlirIdentifierStr(self) };
        str_ref.to_str()
    }
}

impl<'ctx> PartialEq for Identifier<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        to_rbool(unsafe { IR::FFIVal_::mlirIdentifierEqual(*self, *other) })
    }
}
impl<'ctx> Eq for Identifier<'ctx> {}

#[cfg(test)]
mod identfier_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let id_01 = Identifier::get(&ctx, "id_01");
        let id_01_01 = Identifier::get(&ctx, "id_01");
        let id_02 = Identifier::get(&ctx, "id_02");

        assert!(id_01.str() == "id_01");
        assert!(id_01_01.str() == "id_01");
        assert!(id_02.str() == "id_02");

        assert_eq!(id_01, id_01_01);
        assert_ne!(id_01, id_02);
        assert_ne!(id_01_01, id_02);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NamedAttr<'ctx> {
    pub name: Identifier<'ctx>,
    pub attribute: Attr<'ctx>,
}

impl<'ctx> Into<MlirNamedAttribute> for NamedAttr<'ctx> {
    fn into(self) -> MlirNamedAttribute {
        MlirNamedAttribute {
            name: self.name.into(),
            attribute: self.attribute.into(),
        }
    }
}

impl<'ctx> Display for NamedAttr<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(
            formatter,
            "named_attr(name: {}, attr: {})",
            self.name, self.attribute
        )?;
        Ok(())
    }
}

impl<'ctx> HandleWithContext<'ctx> for NamedAttr<'ctx> {
    type HandleTy = MlirNamedAttribute;
    fn get_context_handle(&self) -> MlirContext {
        self.attribute.get_context_handle()
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        let attribute = Attr::from_handle_and_phantom(handle.attribute, phantom);
        let name = Identifier::from_handle_and_phantom(handle.name, phantom);
        Self { attribute, name }
    }
}

mlir_impl_macros::define_builtin_attrs!(
    Location,
    AffineMap,
    Array,
    Dictionary,
    Float,
    Integer,
    Bool,
    IntegerSet,
    Opaque,
    String,
    SymbolRef,
    FlatSymbolRef,
    Type,
    Unit,
    DenseBoolArray,
    DenseI8Array,
    DenseI16Array,
    DenseI32Array,
    DenseI64Array,
    DenseF32Array,
    DenseF64Array,
    DenseElements,
    DenseIntElements,
    DenseFPElements,
    DenseResourceElements,
    SparseElements,
    StridedLayout,
);

impl<'ctx> AffineMapAttr<'ctx> {
    pub fn get(affine_map: AffineMap<'ctx>) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirAffineMapAttrGet(affine_map) };
        let attr = Attr::from_handle_same_context(handle, &affine_map);
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value(self) -> AffineMap<'ctx> {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirAffineMapAttrGetValue(self) };
        AffineMap::from_handle_same_context(handle, &self)
    }
}

#[cfg(test)]
pub mod affine_map_attr_test {
    use super::*;
    use crate::affine_map::affine_map_test::*;

    pub fn create_affine_layout_attr<'ctx>(
        ctx: &'ctx Context,
        shape: &[i64],
        k_dynamic: i64,
        col_major: bool,
    ) -> AffineMapAttr<'ctx> {
        let affine_map = create_affine_layout(ctx, shape, k_dynamic, col_major);
        AffineMapAttr::get(affine_map)
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let affine_maps = generate_some_affine_maps(&ctx);
        assert!(affine_maps.len() > 0);

        for map in affine_maps {
            let affine_map_attr = AffineMapAttr::get(map);
            assert!(IsA::<AffineMapAttr>::is_a(affine_map_attr));
            assert_eq!(affine_map_attr.get_value(), map);
        }
    }
}

impl<'ctx> ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, attrs: &[Attr<'ctx>]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirArrayAttrGet(
                ctx,
                attrs.len() as i64,
                attrs.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn len(self) -> usize {
        (unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirArrayAttrGetNumElements(self) }) as _
    }
    pub fn get_element(self, pos: usize) -> Attr<'ctx> {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirArrayAttrGetElement(self, pos as i64) };
        Attr::from_handle_same_context(handle, &self)
    }
}

#[cfg(test)]
pub mod array_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let attrs = attr_test::create_some_attrs(&ctx);
        let array_attr = ArrayAttr::get(&ctx, &attrs);
        assert!(IsA::<ArrayAttr>::is_a_non_null(array_attr));
        let num_elems = array_attr.len();
        assert_eq!(num_elems, attrs.len());
        for i in 0..num_elems {
            assert_eq!(attrs[i], array_attr.get_element(i));
        }
    }
}

impl<'ctx> DictionaryAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, attrs: &[NamedAttr]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDictionaryAttrGet(
                ctx,
                attrs.len() as i64,
                attrs.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<DictionaryAttr<'ctx>>::cast(attr) }
    }
    pub fn len(self) -> usize {
        (unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirDictionaryAttrGetNumElements(self) }) as _
    }
    pub fn get_element(self, pos: usize) -> NamedAttr<'ctx> {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDictionaryAttrGetElement(self, pos as i64) };
        NamedAttr::from_handle_same_context(handle, &self)
    }
    pub fn get_element_by_name(self, name: &str) -> Attr<'ctx> {
        let name_ref = StrRef::from_str(name);
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDictionaryAttrGetElementByName(self, name_ref)
        };
        Attr::from_handle_same_context(handle, &self)
    }
}

#[cfg(test)]
pub mod dict_attr_test {
    use super::attr_test::*;
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let named_attrs = create_some_named_attrs(&ctx);
        let num_elements = named_attrs.len();
        assert!(num_elements != 0);
        let dict_attr = DictionaryAttr::get(&ctx, &named_attrs);
        assert!(dict_attr.is_a_non_null());
        assert_eq!(dict_attr.len(), num_elements);
        for i in 0..num_elements {
            assert_eq!(dict_attr.get_element(i), named_attrs[i]);
            assert_eq!(
                dict_attr.get_element_by_name(named_attrs[i].name.str()),
                named_attrs[i].attribute
            );
        }
    }
}

impl<'ctx> FloatAttr<'ctx> {
    pub fn f64_get(ctx: &'ctx Context, ty: Type<'ctx>, val: f64) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirFloatAttrDoubleGet(ctx, ty, val) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn f64_get_checked(loc: Location, ty: Type<'ctx>, val: f64) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirFloatAttrDoubleGetChecked(loc, ty, val) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value_f64(self) -> f64 {
        unsafe { BuiltinAttributes::FFIVal_::mlirFloatAttrGetValueDouble(self) }
    }
}

#[cfg(test)]
pub mod float_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let loc = Location::unknown_get(&ctx);
        let bf16_ty = FloatType::get(&ctx, FloatKind::BF16);
        let f16_ty = FloatType::get(&ctx, FloatKind::F16);
        let f32_ty = FloatType::get(&ctx, FloatKind::F32);
        let f64_ty = FloatType::get(&ctx, FloatKind::F64);
        let tf32_ty = FloatType::get(&ctx, FloatKind::TF32);
        let float_tys = [bf16_ty, f16_ty, f16_ty, f32_ty, f64_ty, tf32_ty];
        let float_vals: &[f64] = &[
            0.0,
            0.0,
            -0.0,
            1.0,
            -1.0,
            std::f64::INFINITY,
            -std::f64::INFINITY,
            std::f64::NAN,
        ];
        for f_ty in float_tys {
            for f_val in float_vals {
                let f_attr = FloatAttr::f64_get(&ctx, f_ty.into(), *f_val);
                assert!(IsA::<FloatAttr>::is_a_non_null(f_attr));
                assert_eq!(f_attr, FloatAttr::f64_get_checked(loc, f_ty.into(), *f_val));
                let lhs = f_attr.get_value_f64();
                let rhs = *f_val;
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }
}

impl<'ctx> IntegerAttr<'ctx> {
    pub fn get(ty: Type, val: i64) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirIntegerAttrGet(ty, val) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value_int(self) -> i64 {
        unsafe { BuiltinAttributes::FFIVal_::mlirIntegerAttrGetValueInt(self) }
    }
    pub fn get_value_sint(self) -> i64 {
        unsafe { BuiltinAttributes::FFIVal_::mlirIntegerAttrGetValueSInt(self) }
    }
    pub fn get_value_uint(self) -> u64 {
        unsafe { BuiltinAttributes::FFIVal_::mlirIntegerAttrGetValueUInt(self) }
    }
}

#[cfg(test)]
pub mod int_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();

        let i64_ty = IntegerType::get(&ctx, 64);
        let i32_ty = IntegerType::get(&ctx, 32);
        let i16_ty = IntegerType::get(&ctx, 16);
        let i8_ty = IntegerType::get(&ctx, 8);
        let i_tys = [i64_ty, i32_ty, i16_ty, i8_ty];

        let u64_ty = IntegerType::unsigned_get(&ctx, 64);
        let u32_ty = IntegerType::unsigned_get(&ctx, 32);
        let u16_ty = IntegerType::unsigned_get(&ctx, 16);
        let u8_ty = IntegerType::unsigned_get(&ctx, 8);
        let u_tys = [u64_ty, u32_ty, u16_ty, u8_ty];

        let s64_ty = IntegerType::signed_get(&ctx, 64);
        let s32_ty = IntegerType::signed_get(&ctx, 32);
        let s16_ty = IntegerType::signed_get(&ctx, 16);
        let s8_ty = IntegerType::signed_get(&ctx, 8);
        let s_tys = [s64_ty, s32_ty, s16_ty, s8_ty];

        let vals: &[i64] = &[
            0,
            1,
            -1,
            127,
            128,
            129,
            -127,
            -128,
            -129,
            255,
            256,
            257,
            -255,
            -256,
            -257,
            32767,
            32768,
            32769,
            -32767,
            -32768,
            -32769,
            65535,
            65536,
            65537,
            -65535,
            -65536,
            -65537,
            2147483647,
            2147483648,
            2147483649,
            -2147483647,
            -2147483648,
            -3247483649,
            4294967295,
            4294967296,
            4294967297,
            -4294967295,
            -4294967296,
            -4294967297,
            9223372036854775805,
            9223372036854775806,
            9223372036854775807,
            -9223372036854775806,
            -9223372036854775807,
            -9223372036854775808,
        ];

        let sign_ext = |x: i64, width: u32| -> i64 {
            let mut u_val = x as u64;
            if width != 64 {
                u_val = u_val & ((1u64 << width) - 1);
                if u_val & (1u64 << (width - 1)) != 0 {
                    u_val = u_val | ((u64::MAX >> width) << width);
                }
            }
            u_val as i64
        };

        let zero_ext = |x: i64, width: u32| -> u64 {
            let mut u_val = x as u64;
            if width != 64 {
                u_val = u_val & ((1u64 << width) - 1);
            }
            u_val
        };

        for ty in i_tys {
            for val in vals {
                let int_attr = IntegerAttr::get(ty.into(), *val);
                assert!(IsA::<IntegerAttr>::is_a_non_null(int_attr));
                assert_eq!(int_attr.get_value_int(), sign_ext(*val, ty.get_width()));
            }
        }

        for ty in u_tys {
            for val in vals {
                let int_attr = IntegerAttr::get(ty.into(), *val);
                assert!(IsA::<IntegerAttr>::is_a_non_null(int_attr));
                assert_eq!(int_attr.get_value_uint(), zero_ext(*val, ty.get_width()));
            }
        }

        for ty in s_tys {
            for val in vals {
                let int_attr = IntegerAttr::get(ty.into(), *val);
                assert!(IsA::<IntegerAttr>::is_a_non_null(int_attr));
                assert_eq!(int_attr.get_value_sint(), sign_ext(*val, ty.get_width()));
            }
        }
    }
}

impl<'ctx> BoolAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, val: bool) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirBoolAttrGet(ctx, to_cbool(val)) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_val(self) -> bool {
        to_rbool(unsafe { BuiltinAttributes::FFIVal_::mlirBoolAttrGetValue(self) })
    }
}

#[cfg(test)]
pub mod bool_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let true_attr = BoolAttr::get(&ctx, true);
        let another_true_attr = BoolAttr::get(&ctx, true);
        let false_attr = BoolAttr::get(&ctx, false);

        assert!(IsA::<BoolAttr>::is_a_non_null(true_attr));
        assert!(IsA::<BoolAttr>::is_a_non_null(another_true_attr));
        assert!(IsA::<BoolAttr>::is_a_non_null(false_attr));

        assert_eq!(true_attr.get_val(), true);
        assert_eq!(another_true_attr.get_val(), true);
        assert_eq!(false_attr.get_val(), false);

        assert_eq!(true_attr, another_true_attr);
        assert_ne!(false_attr, true_attr);
        assert_ne!(false_attr, another_true_attr);
    }
}

impl<'ctx> IntegerSetAttr<'ctx> {
    pub fn get(int_set: IntegerSet<'ctx>) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirIntegerSetAttrGet(int_set) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value(self) -> IntegerSet<'ctx> {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirIntegerSetAttrGetValue(self) };
        IntegerSet::from_handle_same_context(handle, &self)
    }
}

// FIXME test integer set

impl<'ctx> OpaqueAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, dialect_namespace: &str, data: &[u8], ty: Type<'ctx>) -> Self {
        let dialect_namespace_ref: StrRef = dialect_namespace.into();
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirOpaqueAttrGet(
                ctx,
                dialect_namespace_ref,
                data.len() as i64,
                data.as_ptr() as *const _,
                ty,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_dialect_namespace(self) -> &'ctx str {
        let str_ref: StrRef =
            unsafe { BuiltinAttributes::FFIVal_::mlirOpaqueAttrGetDialectNamespace(self) };
        str_ref.into()
    }
    pub fn get_data(self) -> &'ctx [u8] {
        let str_ref: StrRef = unsafe { BuiltinAttributes::FFIVal_::mlirOpaqueAttrGetData(self) };
        str_ref.into()
    }
}

// FIXME: test opaque attr

impl<'ctx> StringAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, s: &str) -> Self {
        let str_ref = StrRef::from(s);
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirStringAttrGet(ctx, str_ref) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn typed_get(ty: Type<'ctx>, s: &str) -> Self {
        let str_ref = StrRef::from(s);
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirStringAttrTypedGet(ty, str_ref) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value(self) -> &'ctx str {
        let str_ref: StrRef<'ctx> =
            unsafe { BuiltinAttributes::FFIVal_::mlirStringAttrGetValue(self) };
        str_ref.into()
    }
}

#[cfg(test)]
pub mod string_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let i32_ty: Type = IntegerType::get(&ctx, 32).into();

        let strs: &[&str] = &["", "hello"];
        for s in strs {
            let str_attr_01 = StringAttr::get(&ctx, s);
            let str_attr_02 = StringAttr::typed_get(i32_ty, s);
            assert_eq!(str_attr_01.get_value(), *s);
            assert_eq!(str_attr_02.get_value(), *s);
        }
    }
}

impl<'ctx> SymbolRefAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, symbol: &str, refs: &[Attr<'ctx>]) -> Self {
        let symbol_str_ref = StrRef::from(symbol);
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirSymbolRefAttrGet(
                ctx,
                symbol_str_ref,
                refs.len() as i64,
                refs.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<SymbolRefAttr>::cast(attr) }
    }
    pub fn get_root_ref(self) -> &'ctx str {
        let str_ref: StrRef =
            unsafe { BuiltinAttributes::FFIVal_::mlirSymbolRefAttrGetRootReference(self) };
        str_ref.into()
    }
    pub fn get_leaf_ref(self) -> &'ctx str {
        let str_ref: StrRef =
            unsafe { BuiltinAttributes::FFIVal_::mlirSymbolRefAttrGetLeafReference(self) };
        str_ref.into()
    }
    pub fn get_num_nested_ref(self) -> usize {
        (unsafe {
            BuiltinAttributes::FFIVal_::<i64>::mlirSymbolRefAttrGetNumNestedReferences(self)
        }) as _
    }
    pub fn get_nested_ref(self, pos: usize) -> Attr<'ctx> {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirSymbolRefAttrGetNestedReference(self, pos as i64)
        };
        unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) }
    }
}

impl<'ctx> FlatSymbolRefAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, s: &str) -> Self {
        let str_ref = StrRef::from(s);
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirFlatSymbolRefAttrGet(ctx, str_ref) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value(self) -> &'ctx str {
        let str_ref: StrRef =
            unsafe { BuiltinAttributes::FFIVal_::mlirFlatSymbolRefAttrGetValue(self) };
        str_ref.into()
    }
}

#[cfg(test)]
pub mod symbolref_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();

        let strs: &[&str] = &["some", "arbitrary", "words"];
        let mut attrs: Vec<Attr> = Vec::new();
        for s in strs {
            let flat_sym_ref = FlatSymbolRefAttr::get(&ctx, s);
            assert_eq!(flat_sym_ref.get_value(), *s);
            attrs.push(flat_sym_ref.into());
        }
        let root = "root";
        let sym_ref_attr = SymbolRefAttr::get(&ctx, root, &attrs);
        assert_eq!(sym_ref_attr.get_root_ref(), root);
        assert_eq!(sym_ref_attr.get_num_nested_ref(), strs.len());
        let leaf_ref = sym_ref_attr.get_leaf_ref();
        assert_eq!(leaf_ref, strs[strs.len() - 1]);
        for i in 0..strs.len() {
            let nested_attr = sym_ref_attr.get_nested_ref(i);
            let nested_flat_ref = unsafe { IsA::<FlatSymbolRefAttr>::cast(nested_attr) };
            assert_eq!(nested_flat_ref.get_value(), strs[i]);
        }
    }
}

impl<'ctx> TypeAttr<'ctx> {
    pub fn get(ty: Type<'ctx>) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirTypeAttrGet(ty) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_value(self) -> Type<'ctx> {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirTypeAttrGetValue(self) };
        Type::from_handle_same_context(handle, &self)
    }
}

#[cfg(test)]
pub mod type_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();

        let i32_ty = IntegerType::get(&ctx, 32).into();
        let type_attr = TypeAttr::get(i32_ty);
        assert_eq!(type_attr.get_value(), i32_ty);
    }
}

impl<'ctx> UnitAttr<'ctx> {
    pub fn get(ctx: &'ctx Context) -> Self {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirUnitAttrGet(ctx) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
}

#[cfg(test)]
pub mod unit_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let unit_attr_01 = UnitAttr::create_null();
        let unit_attr_02 = UnitAttr::get(&ctx);
        let unit_attr_03 = UnitAttr::get(&ctx);
        assert_ne!(unit_attr_01, unit_attr_02);
        assert_ne!(unit_attr_01, unit_attr_03);
        assert_eq!(unit_attr_02, unit_attr_03);
        assert!(IsA::<UnitAttr>::is_a_non_null(unit_attr_02));
        assert!(IsA::<UnitAttr>::is_a_non_null(unit_attr_03));
    }
}

pub trait ElementsAttr<'ctx>: AttrTrait<'ctx> {
    fn get_value(self, indices: &[u64]) -> Attr<'ctx> {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirElementsAttrGetValue(
                self,
                indices.len() as i64,
                indices.as_ptr() as *mut _,
            )
        };
        unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    fn is_valid_index(self, indices: &[u64]) -> bool {
        to_rbool(unsafe {
            BuiltinAttributes::FFIVal_::mlirElementsAttrIsValidIndex(
                self,
                indices.len() as i64,
                indices.as_ptr() as *mut _,
            )
        })
    }
    fn get_num_elements(self) -> usize {
        (unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirElementsAttrGetNumElements(self) }) as _
    }
}

// FIXME: get typeid
pub trait DenseArrrayAttr<'ctx>: AttrTrait<'ctx> {
    fn get_num_elements(self) -> usize {
        (unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirDenseArrayGetNumElements(self) }) as _
    }
}

impl<'ctx> DenseBoolArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[std::ffi::c_int]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseBoolArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> bool {
        to_rbool(unsafe {
            BuiltinAttributes::FFIVal_::<CBool>::mlirDenseBoolArrayGetElement(self, pos as i64)
        })
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseBoolArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_bool_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseBoolArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let b_arr = &[1i32, 0i32, 2i32, 0i32, 3i32, -1i32, -2i32];
        let b_arr_attr = DenseBoolArrayAttr::get(&ctx, b_arr);
        assert!(!b_arr_attr.is_a_elements_attr());
        let num_elem = b_arr_attr.get_num_elements();
        assert_eq!(num_elem, b_arr.len());
        for i in 0..num_elem {
            assert_eq!(b_arr_attr.get_element(i), 0 != b_arr[i]);
        }
    }
}

impl<'ctx> DenseI8ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[i8]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseI8ArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> i8 {
        unsafe { BuiltinAttributes::FFIVal_::<i8>::mlirDenseI8ArrayGetElement(self, pos as i64) }
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseI8ArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_i8_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseI8ArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let i8_arr = &[1i8, 0i8, 2i8, 0i8, 3i8, -1i8, -2i8];
        let i8_arr_attr = DenseI8ArrayAttr::get(&ctx, i8_arr);
        assert!(!i8_arr_attr.is_a_elements_attr());
        let num_elem = i8_arr_attr.get_num_elements();
        assert_eq!(num_elem, i8_arr.len());
        for i in 0..num_elem {
            assert_eq!(i8_arr_attr.get_element(i), i8_arr[i]);
        }
    }
}

impl<'ctx> DenseI16ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[i16]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseI16ArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> i16 {
        unsafe { BuiltinAttributes::FFIVal_::<i16>::mlirDenseI16ArrayGetElement(self, pos as i64) }
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseI16ArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_i16_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseI16ArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let i16_arr = &[1i16, 0i16, 2i16, 0i16, 3i16, -1i16, -2i16];
        let i16_arr_attr = DenseI16ArrayAttr::get(&ctx, i16_arr);
        assert!(!i16_arr_attr.is_a_elements_attr());
        let num_elem = i16_arr_attr.get_num_elements();
        assert_eq!(num_elem, i16_arr.len());
        for i in 0..num_elem {
            assert_eq!(i16_arr_attr.get_element(i), i16_arr[i]);
        }
    }
}

impl<'ctx> DenseI32ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[i32]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseI32ArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> i32 {
        unsafe { BuiltinAttributes::FFIVal_::<i32>::mlirDenseI32ArrayGetElement(self, pos as i64) }
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseI32ArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_i32_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseI32ArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let i32_arr = &[1i32, 0i32, 2i32, 0i32, 3i32, -1i32, -2i32];
        let i32_arr_attr = DenseI32ArrayAttr::get(&ctx, i32_arr);
        assert!(!i32_arr_attr.is_a_elements_attr());
        let num_elem = i32_arr_attr.get_num_elements();
        assert_eq!(num_elem, i32_arr.len());
        for i in 0..num_elem {
            assert_eq!(i32_arr_attr.get_element(i), i32_arr[i]);
        }
    }
}

impl<'ctx> DenseI64ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[i64]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseI64ArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> i64 {
        unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirDenseI64ArrayGetElement(self, pos as i64) }
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseI64ArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_i64_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseI64ArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let i64_arr = &[1i64, 0i64, 2i64, 0i64, 3i64, -1i64, -2i64];
        let i64_arr_attr = DenseI64ArrayAttr::get(&ctx, i64_arr);
        assert!(!i64_arr_attr.is_a_elements_attr());
        let num_elem = i64_arr_attr.get_num_elements();
        assert_eq!(num_elem, i64_arr.len());
        for i in 0..num_elem {
            assert_eq!(i64_arr_attr.get_element(i), i64_arr[i]);
        }
    }
}

impl<'ctx> DenseF32ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[f32]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseF32ArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> f32 {
        unsafe { BuiltinAttributes::FFIVal_::<f32>::mlirDenseF32ArrayGetElement(self, pos as i64) }
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseF32ArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_f32_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseF32ArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let f32_arr = &[1f32, 0f32, 2f32, 0f32, 3f32, -1f32, -2f32];
        let f32_arr_attr = DenseF32ArrayAttr::get(&ctx, f32_arr);
        assert!(!f32_arr_attr.is_a_elements_attr());
        let num_elem = f32_arr_attr.get_num_elements();
        assert_eq!(num_elem, f32_arr.len());
        for i in 0..num_elem {
            assert_eq!(f32_arr_attr.get_element(i), f32_arr[i]);
        }
    }
}

impl<'ctx> DenseF64ArrayAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, values: &[f64]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseF64ArrayGet(
                ctx,
                values.len() as i64,
                values.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_element(self, pos: usize) -> f64 {
        unsafe { BuiltinAttributes::FFIVal_::<f64>::mlirDenseF64ArrayGetElement(self, pos as i64) }
    }
}
impl<'ctx> DenseArrrayAttr<'ctx> for DenseF64ArrayAttr<'ctx> {}

#[cfg(test)]
pub mod dense_f64_array_attr_test {
    use super::*;

    #[test]
    fn test_create_empty() {
        let ctx = Context::create();
        let attr = DenseF64ArrayAttr::get(&ctx, &[]);
        assert!(!attr.is_null());
    }

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let f64_arr = &[1f64, 0f64, 2f64, 0f64, 3f64, -1f64, -2f64];
        let f64_arr_attr = DenseF64ArrayAttr::get(&ctx, f64_arr);
        assert!(!f64_arr_attr.is_a_elements_attr());
        let num_elem = f64_arr_attr.get_num_elements();
        assert_eq!(num_elem, f64_arr.len());
        for i in 0..num_elem {
            assert_eq!(f64_arr_attr.get_element(i), f64_arr[i]);
        }
    }
}

impl<'ctx> DenseElementsAttr<'ctx> {
    pub fn get(ty: Type<'ctx>, elements: &[Attr<'ctx>]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrGet(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn raw_buffer_get(ty: Type<'ctx>, buffer: &[u8]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrRawBufferGet(
                ty,
                buffer.len() as u64,
                buffer.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn splat_get(ty: Type<'ctx>, element: Attr<'ctx>) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrSplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn bool_splat_get(ty: Type<'ctx>, element: CBool) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrBoolSplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i8_splat_get(ty: Type<'ctx>, element: i8) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt8SplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u8_splat_get(ty: Type<'ctx>, element: u8) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt8SplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i32_splat_get(ty: Type<'ctx>, element: i32) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt32SplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u32_splat_get(ty: Type<'ctx>, element: u32) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt32SplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i64_splat_get(ty: Type<'ctx>, element: i64) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt64SplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u64_splat_get(ty: Type<'ctx>, element: u64) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt64SplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn f32_splat_get(ty: Type<'ctx>, element: f32) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrFloatSplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn f64_splat_get(ty: Type<'ctx>, element: f64) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrDoubleSplatGet(ty, element) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn bool_get(ty: Type<'ctx>, elements: &[CBool]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrBoolGet(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u8_get(ty: Type<'ctx>, elements: &[u8]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt8Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i8_get(ty: Type<'ctx>, elements: &[i8]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt8Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u16_get(ty: Type<'ctx>, elements: &[u16]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt16Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i16_get(ty: Type<'ctx>, elements: &[i16]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt16Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u32_get(ty: Type<'ctx>, elements: &[u32]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt32Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i32_get(ty: Type<'ctx>, elements: &[i32]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt32Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn u64_get(ty: Type<'ctx>, elements: &[u64]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrUInt64Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn i64_get(ty: Type<'ctx>, elements: &[i64]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrInt64Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn f32_get(ty: Type<'ctx>, elements: &[f32]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrFloatGet(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn f64_get(ty: Type<'ctx>, elements: &[f64]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrDoubleGet(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn bf16_get(ty: Type<'ctx>, elements: &[i16]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrBFloat16Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn f16_get(ty: Type<'ctx>, elements: &[i16]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrFloat16Get(
                ty,
                elements.len() as i64,
                elements.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn str_get(ty: Type<'ctx>, elements: &[&str]) -> Self {
        let str_ref_elements: Vec<_> = elements.iter().map(|x| StrRef::from(*x)).collect();
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirDenseElementsAttrStringGet(
                ty,
                elements.len() as i64,
                str_ref_elements.as_ptr() as *mut _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn reshape_get(self, ty: Type<'ctx>) -> Self {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrReshapeGet(self, ty) };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn is_splat(self) -> bool {
        to_rbool(unsafe { BuiltinAttributes::FFIVal_::<CBool>::mlirDenseElementsAttrIsSplat(self) })
    }
    pub fn get_splat_value(self) -> Attr<'ctx> {
        let handle =
            unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrGetSplatValue(self) };
        unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn get_bool_splat_value(self) -> bool {
        let val = unsafe {
            BuiltinAttributes::FFIVal_::<i32>::mlirDenseElementsAttrGetBoolSplatValue(self)
        };
        if val != 0 {
            true
        } else {
            false
        }
    }
    pub fn get_i8_splat_value(self) -> i8 {
        unsafe { BuiltinAttributes::FFIVal_::<i8>::mlirDenseElementsAttrGetInt8SplatValue(self) }
    }
    pub fn get_u8_splat_value(self) -> u8 {
        unsafe { BuiltinAttributes::FFIVal_::<u8>::mlirDenseElementsAttrGetUInt8SplatValue(self) }
    }
    //pub fn get_i16_splat_value(self) -> i16 {
    //    (unsafe {
    //        BuiltinAttributes::FFIVal_::<i16>::mlirDenseElementsAttrGetInt16SplatValue(self)
    //    }).into()
    //}
    //pub fn get_u16_splat_value(self) -> u16 {
    //    (unsafe {
    //        BuiltinAttributes::FFIVal_::<u16>::mlirDenseElementsAttrGetUInt16SplatValue(self)
    //    }).into()
    //}
    pub fn get_i32_splat_value(self) -> i32 {
        unsafe { BuiltinAttributes::FFIVal_::<i32>::mlirDenseElementsAttrGetInt32SplatValue(self) }
    }
    pub fn get_u32_splat_value(self) -> u32 {
        unsafe { BuiltinAttributes::FFIVal_::<u32>::mlirDenseElementsAttrGetUInt32SplatValue(self) }
    }
    pub fn get_i64_splat_value(self) -> i64 {
        unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirDenseElementsAttrGetInt64SplatValue(self) }
    }
    pub fn get_u64_splat_value(self) -> u64 {
        unsafe { BuiltinAttributes::FFIVal_::<u64>::mlirDenseElementsAttrGetUInt64SplatValue(self) }
    }
    pub fn get_f32_splat_value(self) -> f32 {
        unsafe { BuiltinAttributes::FFIVal_::<f32>::mlirDenseElementsAttrGetFloatSplatValue(self) }
    }
    pub fn get_f64_splat_value(self) -> f64 {
        unsafe { BuiltinAttributes::FFIVal_::<f64>::mlirDenseElementsAttrGetDoubleSplatValue(self) }
    }
    pub fn get_str_splat_value(self) -> &'ctx str {
        (unsafe {
            BuiltinAttributes::FFIVal_::<StrRef>::mlirDenseElementsAttrGetStringSplatValue(self)
        })
        .into()
    }
    pub fn get_bool_value(self, pos: usize) -> bool {
        to_rbool(unsafe {
            BuiltinAttributes::FFIVal_::<CBool>::mlirDenseElementsAttrGetBoolValue(self, pos as i64)
        })
    }
    pub fn get_i8_value(self, pos: usize) -> i8 {
        unsafe {
            BuiltinAttributes::FFIVal_::<i8>::mlirDenseElementsAttrGetInt8Value(self, pos as i64)
        }
    }
    pub fn get_u8_value(self, pos: usize) -> u8 {
        unsafe {
            BuiltinAttributes::FFIVal_::<u8>::mlirDenseElementsAttrGetUInt8Value(self, pos as i64)
        }
    }
    pub fn get_i32_value(self, pos: usize) -> i32 {
        unsafe {
            BuiltinAttributes::FFIVal_::<i32>::mlirDenseElementsAttrGetInt32Value(self, pos as i64)
        }
    }
    pub fn get_u32_value(self, pos: usize) -> u32 {
        unsafe {
            BuiltinAttributes::FFIVal_::<u32>::mlirDenseElementsAttrGetUInt32Value(self, pos as i64)
        }
    }
    pub fn get_i64_value(self, pos: usize) -> i64 {
        unsafe {
            BuiltinAttributes::FFIVal_::<i64>::mlirDenseElementsAttrGetInt64Value(self, pos as i64)
        }
    }
    pub fn get_u64_value(self, pos: usize) -> u64 {
        unsafe {
            BuiltinAttributes::FFIVal_::<u64>::mlirDenseElementsAttrGetUInt64Value(self, pos as i64)
        }
    }
    pub fn get_f32_value(self, pos: usize) -> f32 {
        unsafe {
            BuiltinAttributes::FFIVal_::<f32>::mlirDenseElementsAttrGetFloatValue(self, pos as i64)
        }
    }
    pub fn get_f64_value(self, pos: usize) -> f64 {
        unsafe {
            BuiltinAttributes::FFIVal_::<f64>::mlirDenseElementsAttrGetDoubleValue(self, pos as i64)
        }
    }
    pub fn get_str_value(self, pos: usize) -> &'ctx str {
        (unsafe {
            BuiltinAttributes::FFIVal_::<StrRef>::mlirDenseElementsAttrGetStringValue(
                self, pos as i64,
            )
        })
        .into()
    }
    pub fn get_raw_data(self) -> *const std::ffi::c_void {
        unsafe { BuiltinAttributes::FFIVal_::mlirDenseElementsAttrGetRawData(self) }
    }
}
impl<'ctx> ElementsAttr<'ctx> for DenseElementsAttr<'ctx> {}

impl<'ctx> SparseElementsAttr<'ctx> {
    pub fn get(ty: Type<'ctx>, dense_indices: Attr<'ctx>, dense_values: Attr<'ctx>) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirSparseElementsAttribute(ty, dense_indices, dense_values)
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_indices(self) -> Attr<'ctx> {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirSparseElementsAttrGetIndices(self) };
        unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) }
    }
    pub fn get_values(self) -> Attr<'ctx> {
        let handle = unsafe { BuiltinAttributes::FFIVal_::mlirSparseElementsAttrGetValues(self) };
        unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) }
    }
}

impl<'ctx> StridedLayoutAttr<'ctx> {
    pub fn get(ctx: &'ctx Context, offset: i64, strides: &[i64]) -> Self {
        let handle = unsafe {
            BuiltinAttributes::FFIVal_::mlirStridedLayoutAttrGet(
                ctx,
                offset,
                strides.len() as i64,
                strides.as_ptr() as *const _,
            )
        };
        let attr = unsafe { Attr::from_handle_and_phantom(handle, PhantomData::default()) };
        unsafe { IsA::<Self>::cast(attr) }
    }
    pub fn get_offset(self) -> i64 {
        unsafe { BuiltinAttributes::FFIVal_::mlirStridedLayoutAttrGetOffset(self) }
    }
    pub fn get_num_strides(self) -> usize {
        (unsafe { BuiltinAttributes::FFIVal_::<i64>::mlirStridedLayoutAttrGetNumStrides(self) })
            as _
    }
    pub fn get_stride(self, pos: usize) -> i64 {
        unsafe { BuiltinAttributes::FFIVal_::mlirStridedLayoutAttrGetStride(self, pos as i64) }
    }
}

#[cfg(test)]
pub mod strided_layout_attr_test {
    use super::*;

    #[test]
    fn test_create() {
        let ctx = Context::create();
        let strides: &[i64] = &[1, 2, 3, 5, 7, 11];
        let strided_layout_attr = StridedLayoutAttr::get(&ctx, 13, strides);
        assert_eq!(strided_layout_attr.get_offset(), 13);
        assert_eq!(strided_layout_attr.get_num_strides(), strides.len());
        for i in 0..strides.len() {
            assert_eq!(strided_layout_attr.get_stride(i), strides[i]);
        }
    }
}
