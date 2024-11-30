use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parse2, Token};
use syn::{Ident, Result};

pub trait ElementSyntax {
    fn get_name(&self) -> Ident;
    fn get_span(&self) -> Span {
        self.get_name().span()
    }
    fn to_ident(&self, s: &str) -> Ident {
        Ident::new(s, self.get_span())
    }
}

struct ElementListSyntax<T: Parse> {
    elements: Vec<T>,
}

impl<T: Parse + Clone> Parse for ElementListSyntax<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let elements_syntax = Punctuated::<T, Token![,]>::parse_terminated(input)?;
        let elements = elements_syntax.iter().map(|x| (*x).clone()).collect();
        Ok(ElementListSyntax { elements })
    }
}

pub trait BuilintsInfo {
    type ElemSyn: ElementSyntax + Parse + Clone;
    fn parse_input(&self, input: TokenStream) -> Vec<Self::ElemSyn> {
        parse2::<ElementListSyntax<Self::ElemSyn>>(input)
            .unwrap()
            .elements
    }
    fn get_kind_trait(&self, bi: &Self::ElemSyn) -> Ident;
    fn get_struct_name(&self, bi: &Self::ElemSyn) -> Ident;
    fn get_parent(&self, bi: &Self::ElemSyn) -> Ident;
    fn get_capi_handle_type(&self, bi: &Self::ElemSyn) -> Ident;
    fn get_is_a_capi_func(&self, bi: &Self::ElemSyn) -> syn::Path;
    fn get_not_of_type_message(&self, bi: &Self::ElemSyn) -> String {
        format!("not a {} type", bi.get_name())
    }
    fn get_ancestors(&self, bi: &Self::ElemSyn) -> Vec<syn::Ident>;
}

pub fn define_builtin_type_attr<T>(input: TokenStream, builtin_info: &T) -> TokenStream
where
    T: BuilintsInfo,
{
    let builtins = builtin_info.parse_input(input);
    let mut structs_stream = TokenStream::new();
    for bi in builtins {
        let kind_trait = builtin_info.get_kind_trait(&bi);
        let ty = builtin_info.get_struct_name(&bi);
        let parent = builtin_info.get_parent(&bi);
        let capi_handle = builtin_info.get_capi_handle_type(&bi);
        let is_a_capi_func = builtin_info.get_is_a_capi_func(&bi);
        let not_a_ty_message = builtin_info.get_not_of_type_message(&bi);
        structs_stream.extend(quote! {
            #[repr(C)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub struct #ty <'ctx> {
                pub handle : #parent<'ctx>,
            }

            impl<'ctx> Display for #ty<'ctx> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
                    Into::<#parent<'ctx>>::into(*self).print_to_formatter(formatter)
                }
            }

            impl<'ctx> Into<#capi_handle> for #ty<'ctx> {
                fn into(self) -> #capi_handle {
                    let parent = Into::<#parent::<'ctx>>::into(self);
                    Into::<#capi_handle>::into(parent)
                }
            }

            impl<'ctx> #kind_trait<'ctx> for #ty<'ctx> {}
            impl<'ctx> HandleWithContext<'ctx> for #ty<'ctx> {
                type HandleTy = #parent<'ctx>;
                fn get_context_handle(&self) -> MlirContext {
                    self.handle.get_context_handle()
                }
                unsafe fn from_handle_and_phantom(handle: Self::HandleTy, phantom: PhantomData<&'ctx Context>) -> Self {
                    Self {
                        handle
                    }
                }
            }
            impl<'ctx> NullableRef for #ty<'ctx> {
                fn is_null(self) -> bool {
                    self.handle.is_null()
                }
                fn create_null() -> Self {
                    unsafe {
                        IsA::<Self>::cast(#parent::<'ctx>::create_null())
                    }
                }
            }

            impl<'ctx> From<#ty<'ctx>> for #parent<'ctx> {
                fn from(value : #ty<'ctx>) -> Self {
                    value.handle
                }
            }

            impl <'ctx> TryFrom<#parent<'ctx>> for #ty<'ctx> {
                type Error = crate::type_cast::DownCastError;
                fn try_from(parent: #parent<'ctx>) -> Result<Self, crate::type_cast::DownCastError> {
                    if parent.is_null() || IsA::<#ty>::is_a(parent) {
                        Ok(#ty { handle: parent} )
                    } else {
                        Err(crate::type_cast::DownCastError{message: #not_a_ty_message.to_string()})
                    }
                }
            }

            impl<'ctx> PartialEq<#parent<'ctx>> for #ty<'ctx> {
                fn eq(&self, other: &#parent<'ctx>) -> bool {
                    self.handle == *other
                }
            }

            impl<'ctx> PartialEq<#ty<'ctx>> for #parent<'ctx> {
                fn eq(&self, other: &#ty<'ctx>) -> bool {
                    *self == other.handle
                }
            }

            impl<'ctx> IsA<#ty<'ctx>> for #parent<'ctx> {
                fn is_a_impl(self) -> bool {
                    crate::common::to_rbool(unsafe {#is_a_capi_func(self)} )
                }
                unsafe fn cast(self) -> #ty<'ctx> {
                    #ty {
                        handle: self,
                    }
                }
            }
        });
        for ancestor in builtin_info.get_ancestors(&bi) {
            structs_stream.extend(quote! {

                impl<'ctx> From<#ty<'ctx>> for #ancestor<'ctx> {
                    fn from(value : #ty<'ctx>) -> Self {
                        From::<#parent>::from(From::from(value))
                    }
                }

                impl <'ctx> TryFrom<#ancestor<'ctx>> for #ty<'ctx> {
                    type Error = crate::type_cast::DownCastError;
                    fn try_from(ancestor: #ancestor<'ctx>) -> Result<Self, crate::type_cast::DownCastError> {
                        if ancestor.is_null() || IsA::<#ty>::is_a(ancestor) {
                            let handle = unsafe { IsA::<#parent<'ctx>>::cast(ancestor) };
                            Ok(#ty { handle } )
                        } else {
                            Err(crate::type_cast::DownCastError{message: #not_a_ty_message.to_string()})
                        }
                    }
                }

                impl<'ctx> PartialEq<#ancestor<'ctx>> for #ty<'ctx> {
                    fn eq(&self, other: &#ancestor<'ctx>) -> bool {
                        self.handle == *other
                    }
                }

                impl<'ctx> PartialEq<#ty<'ctx>> for #ancestor<'ctx> {
                    fn eq(&self, other: &#ty<'ctx>) -> bool {
                        *self == other.handle
                    }
                }

                impl<'ctx> IsA<#ty<'ctx>> for #ancestor<'ctx> {
                    fn is_a_impl(self) -> bool {
                        crate::common::to_rbool(unsafe {#is_a_capi_func(self)})
                    }
                    unsafe fn cast(self) -> #ty<'ctx> {
                        let handle = unsafe { IsA::<#parent<'ctx>>::cast(self) };
                        #ty {
                            handle
                        }
                    }
                }
            });
        }
    }
    let mut res = TokenStream::new();
    res.extend(structs_stream);
    // eprintln!("TOKENS: {}", res);
    res
}
