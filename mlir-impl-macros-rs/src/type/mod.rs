use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parenthesized, parse2, Token};
use syn::{Ident, Result};

#[derive(Clone)]
enum TypeWithParent {
    DefaultParent { name: Ident },
    ExplicitParent { name: Ident, parent: Ident },
}

impl TypeWithParent {
    pub fn get_span(&self) -> Span {
        match self {
            Self::DefaultParent { name } => name.span(),
            Self::ExplicitParent { name, parent: _ } => name.span(),
        }
    }
    pub fn get_name(&self) -> String {
        match self {
            Self::DefaultParent { name } => name.to_string(),
            Self::ExplicitParent { name, parent: _ } => name.to_string(),
        }
    }
    pub fn get_parent_type_name(&self) -> String {
        match self {
            Self::DefaultParent { .. } => "Type".to_owned(),
            Self::ExplicitParent { parent, .. } => format!("{}Type", parent),
        }
    }
}

impl Parse for TypeWithParent {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(syn::Ident) {
            let name = input.parse::<syn::Ident>()?;
            //if !input.is_empty() {
            //    return Err(Error::new(input.span(), format!("expected one identifier only: {}", input)));
            //}
            return Ok(TypeWithParent::DefaultParent { name: name });
        }
        let content;
        parenthesized!(content in input);
        //if !content.is_empty() {
        //    return Err(Error::new(
        //        input.span(),
        //        "expected nothing beyond parenthesis",
        //    ));
        //}
        let name = content.parse::<syn::Ident>()?;
        content.parse::<Token![,]>()?;
        let parent = content.parse::<syn::Ident>()?;
        //if !content.is_empty() {
        //    return Err(Error::new(
        //        content.span(),
        //        "expected just two names within parentheses",
        //    ));
        //}
        Ok(TypeWithParent::ExplicitParent { name, parent })
    }
}

struct TypesVec {
    types: Vec<TypeWithParent>,
}

impl Parse for TypesVec {
    fn parse(input: ParseStream) -> Result<Self> {
        let types = Punctuated::<TypeWithParent, Token![,]>::parse_terminated(input)?;
        Ok(TypesVec {
            types: types.iter().map(|x| x.clone()).collect(),
        })
    }
}

pub fn define_builtin_types(input: TokenStream) -> TokenStream {
    let types = parse2::<TypesVec>(input).unwrap().types;
    let mut structs_stream = TokenStream::new();
    let mut enum_names = Vec::new();
    for ty in types {
        let type_name = ty.get_name();
        let span = ty.get_span();
        let type_name_str = format!("{}Type", type_name);
        let type_name_ident = Ident::new(&type_name_str, span);
        let par_name_str = ty.get_parent_type_name();
        let par_name_ident = Ident::new(&par_name_str, span);
        let is_a_func_str = format!("mlirTypeIsA{}", type_name);
        let is_a_func_ident = Ident::new(&is_a_func_str, span);
        let not_a_type_message = format!("not a {} type", type_name);
        let from_parent_ty_tokens = match ty {
            TypeWithParent::DefaultParent { .. } => quote! {ty},
            TypeWithParent::ExplicitParent { .. } => {
                quote! { #par_name_ident::from_type(ty).unwrap()}
            }
        };
        let from_parent_ty_unchecked_tokens = match ty {
            TypeWithParent::DefaultParent { .. } => quote! {ty},
            TypeWithParent::ExplicitParent { .. } => {
                quote! {unsafe { #par_name_ident::from_type_unchecked(ty) }}
            }
        };
        structs_stream.extend(quote! {
        #[repr(C)]
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct #type_name_ident <'ctx> {
            pub ty: #par_name_ident<'ctx>,
        }
        impl<'ctx> #type_name_ident<'ctx> {
            pub fn is_a(ty: Type<'ctx>) -> bool {
                to_rbool(unsafe { BuiltinTypes::FFIVal_::#is_a_func_ident(ty) })
            }
            pub fn from_type(ty: Type<'ctx>) -> Option<Self> {
                if Self::is_a(ty) {
                    Some(Self { ty: #from_parent_ty_tokens })
                } else {
                    None
                }
            }
            pub unsafe fn from_type_unchecked(ty: Type<'ctx>) -> Self {
                //!debug_assert(Self::is_a(ty));
                Self {
                    ty: #from_parent_ty_unchecked_tokens
                }
            }
        }
        impl<'ctx> Into<#par_name_ident<'ctx>> for #type_name_ident<'ctx> {
            fn into(self) -> #par_name_ident<'ctx> {
                self.ty
            }
        }
        impl<'ctx> Into<MlirType> for #type_name_ident<'ctx> {
            fn into(self) -> MlirType {
                self.ty.into()
            }
        }
        impl<'ctx> TryFrom<Type<'ctx>> for #type_name_ident<'ctx> {
            type Error = &'static str;

            fn try_from(ty: Type<'ctx>) -> Result<Self, Self::Error> {
                Self::from_type(ty).ok_or(#not_a_type_message)
            }
        }
                });
        enum_names.push(type_name_ident);
    }
    let mut res = TokenStream::new();
    res.extend(structs_stream);
    res.extend(quote! {
        enum BuilintTypeKind {
            #(#enum_names),*
        }
    });
    //    eprintln!("TOKENS: {}", res);
    res
}

struct FloatKinds {
    kinds: Vec<Ident>,
}

impl Parse for FloatKinds {
    fn parse(input: ParseStream) -> Result<Self> {
        let kinds = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        Ok(Self {
            kinds: kinds.iter().cloned().collect(),
        })
    }
}

pub fn define_float_kind(input: TokenStream) -> TokenStream {
    let float_kinds = parse2::<FloatKinds>(input).unwrap().kinds;
    let span = float_kinds[0].clone().span();
    let mut capi_get_func_idents = Vec::new();
    let mut capi_isa_func_idents = Vec::new();
    let mut capi_get_typeid_func_idents = Vec::new();
    for fp in float_kinds.clone() {
        let fp_name = fp.to_string();
        let (short_name, long_name) = {
            if fp_name.starts_with("F4")
                || (fp_name.starts_with("F6") && !fp_name.starts_with("F64"))
                || fp_name.starts_with("F8")
            {
                let name = format!("Float{}", fp_name.strip_prefix("F").unwrap());
                (name.clone(), name.clone())
            } else if fp_name.starts_with("TF") {
                (fp_name.clone(), format!("Float{}", fp_name))
            } else if fp_name.starts_with("BF") {
                (
                    fp_name.clone(),
                    format!("BFloat{}", fp_name.strip_prefix("BF").unwrap()),
                )
            } else {
                (
                    fp_name.clone(),
                    format!("Float{}", fp_name.strip_prefix("F").unwrap()),
                )
            }
        };
        capi_get_func_idents.push(Ident::new(&format!("mlir{}TypeGet", short_name), span));
        capi_isa_func_idents.push(Ident::new(&format!("mlirTypeIsA{}", short_name), span));
        capi_get_typeid_func_idents
            .push(Ident::new(&format!("mlir{}TypeGetTypeID", long_name), span));
    }
    quote! {
            #[derive(EnumIter, Copy, Clone)]
            pub enum FloatKind {
                #(#float_kinds),*
            }
    impl<'ctx> FloatType<'ctx> {
        pub fn get(ctx: &'ctx Context, kind: FloatKind) -> Self {
            match kind {
                #(FloatKind::#float_kinds=> {let ty = unsafe {
                    BuiltinTypes::FFIVal_::#capi_get_func_idents(ctx)
                };
                unsafe {Self::from_type_unchecked(ty)}}),*
            }
        }
        pub fn is_a_fp(self, kind: FloatKind) -> bool {
            match kind {
                #(FloatKind::#float_kinds=> to_rbool(unsafe {
                    BuiltinTypes::FFIVal_::#capi_isa_func_idents(self)
                })),*
            }
        }
        pub fn get_typeid(kind: FloatKind) -> TypeID {
            match kind {
                #(FloatKind::#float_kinds=> unsafe {
                    BuiltinTypes::FFIVal_::#capi_get_typeid_func_idents()
                }),*
            }
        }
        }}
}
