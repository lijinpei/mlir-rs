use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parse2, Token};
use syn::{Ident, Result};

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
                #(FloatKind::#float_kinds=> {let handle = unsafe {
                    BuiltinTypes::FFIVal_::#capi_get_func_idents(ctx)
                };
                let ty = unsafe { Type::from_handle_and_phantom(handle, PhantomData::default()) };
                unsafe { IsA::<Self>::cast(ty)}}),*
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
