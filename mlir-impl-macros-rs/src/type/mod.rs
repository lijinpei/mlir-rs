use crate::type_attr::*;
use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{parenthesized, Token};
use syn::{parse_quote, Ident, Result};

pub mod floats;

#[derive(Clone)]
enum TypeParent {
    DefaultParent,
    ExplicitParent { parent: Ident },
}

#[derive(Clone)]
struct BuiltinTypeSyntax {
    pub name: Ident,
    pub parent: TypeParent,
}

impl ElementSyntax for BuiltinTypeSyntax {
    fn get_name(&self) -> Ident {
        self.name.clone()
    }
}

impl Parse for BuiltinTypeSyntax {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(syn::Ident) {
            let name = input.parse::<syn::Ident>()?;
            let parent = TypeParent::DefaultParent;
            return Ok(BuiltinTypeSyntax { name, parent });
        }
        let content;
        parenthesized!(content in input);
        let name = content.parse::<syn::Ident>()?;
        content.parse::<Token![,]>()?;
        let parent = TypeParent::ExplicitParent {
            parent: content.parse::<syn::Ident>()?,
        };
        Ok(BuiltinTypeSyntax { name, parent })
    }
}

struct BuiltinTypesInfo;
impl BuilintsInfo for BuiltinTypesInfo {
    type ElemSyn = BuiltinTypeSyntax;
    fn get_kind_trait(&self, bi: &Self::ElemSyn) -> Ident {
        bi.to_ident("TypeTrait")
    }
    fn get_struct_name(&self, bi: &BuiltinTypeSyntax) -> Ident {
        bi.to_ident(&format!("{}Type", bi.name))
    }
    fn get_parent(&self, bi: &BuiltinTypeSyntax) -> Ident {
        match bi.parent {
            TypeParent::DefaultParent => bi.to_ident("Type"),
            TypeParent::ExplicitParent { ref parent } => bi.to_ident(&format!("{}Type", parent)),
        }
    }
    fn get_capi_handle_type(&self, bi: &BuiltinTypeSyntax) -> Ident {
        bi.to_ident("MlirType")
    }
    fn get_is_a_capi_func(&self, bi: &BuiltinTypeSyntax) -> syn::Path {
        let func_name = bi.to_ident(&format!("mlirTypeIsA{}", bi.name));
        parse_quote! {
            ::mlir_capi::BuiltinTypes::FFIVal_::#func_name
        }
    }
    fn get_ancestors(&self, bi: &BuiltinTypeSyntax) -> Vec<syn::Ident> {
        let mut ancestors = vec![];
        if let TypeParent::ExplicitParent { parent: _ } = bi.parent {
            ancestors.push(bi.to_ident("Type"));
        }
        ancestors
    }
}

pub fn define_builtin_types(input: TokenStream) -> TokenStream {
    let tys_info = BuiltinTypesInfo {};
    define_builtin_type_attr(input, &tys_info)
}
