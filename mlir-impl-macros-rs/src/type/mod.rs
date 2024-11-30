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
    //let mut type_impl_stream = TokenStream::new();
    let mut enum_names = Vec::new();
    for ty in types {
        let name_str = format!("{}Type", ty.get_name());
        let type_name = Ident::new(&name_str, ty.get_span());
        let par_name = ty.get_parent_type_name();
        let par_type_name = Ident::new(&par_name, ty.get_span());
        structs_stream.extend(quote! {
        #[repr(C)]
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct #type_name <'ctx> {
            pub ty: #par_type_name<'ctx>,
        }
                });
        enum_names.push(type_name);
    }
    let mut res = TokenStream::new();
    res.extend(structs_stream);
    res.extend(quote! {
        enum BuilintTypeKind {
            #(#enum_names),*
        }
    });
    res
}
