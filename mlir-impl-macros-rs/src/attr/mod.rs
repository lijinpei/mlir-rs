use crate::type_attr::*;
use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{parenthesized, Token};
use syn::{parse_quote, Ident, Result};

#[derive(Clone)]
enum AttrParent {
    DefaultParent,
    ExplicitParent { parent: Ident },
}

#[derive(Clone)]
struct BuiltinAttrSyntax {
    pub name: Ident,
    pub parent: AttrParent,
}

impl ElementSyntax for BuiltinAttrSyntax {
    fn get_name(&self) -> Ident {
        self.name.clone()
    }
}

impl Parse for BuiltinAttrSyntax {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(syn::Ident) {
            let name = input.parse::<syn::Ident>()?;
            let parent = AttrParent::DefaultParent;
            return Ok(BuiltinAttrSyntax { name, parent });
        }
        let content;
        parenthesized!(content in input);
        let name = content.parse::<syn::Ident>()?;
        content.parse::<Token![,]>()?;
        let parent = AttrParent::ExplicitParent {
            parent: content.parse::<syn::Ident>()?,
        };
        Ok(BuiltinAttrSyntax { name, parent })
    }
}

struct BuiltinAttrsInfo;

impl BuilintsInfo for BuiltinAttrsInfo {
    type ElemSyn = BuiltinAttrSyntax;
    fn get_kind_trait(&self, bi: &Self::ElemSyn) -> Ident {
        bi.to_ident("AttrTrait")
    }
    fn get_struct_name(&self, bi: &BuiltinAttrSyntax) -> Ident {
        bi.to_ident(&format!("{}Attr", bi.get_name()))
    }
    fn get_parent(&self, bi: &BuiltinAttrSyntax) -> Ident {
        match bi.parent {
            AttrParent::DefaultParent => bi.to_ident("Attr"),
            AttrParent::ExplicitParent { ref parent } => bi.to_ident(&format!("{}Attr", parent)),
        }
    }
    fn get_capi_handle_type(&self, bi: &Self::ElemSyn) -> Ident {
        bi.to_ident("MlirAttribute")
    }
    fn get_is_a_capi_func(&self, bi: &BuiltinAttrSyntax) -> syn::Path {
        let func_name = bi.to_ident(&format!("mlirAttributeIsA{}", bi.name));
        parse_quote! {
            ::mlir_capi::BuiltinAttributes::FFIVal_::#func_name
        }
    }
    fn get_ancestors(&self, bi: &BuiltinAttrSyntax) -> Vec<syn::Ident> {
        let mut ancestors = vec![];
        if let AttrParent::ExplicitParent { parent: _ } = bi.parent {
            ancestors.push(bi.to_ident("Attr"));
        }
        ancestors
    }
}

pub fn define_builtin_attrs(input: TokenStream) -> TokenStream {
    let tys_info = BuiltinAttrsInfo {};
    define_builtin_type_attr(input, &tys_info)
}
