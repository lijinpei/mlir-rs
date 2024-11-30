use proc_macro;

mod attr;
mod r#type;
mod type_attr;

#[proc_macro]
pub fn define_builtin_types(types: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let types = proc_macro2::TokenStream::from(types);
    let res = r#type::define_builtin_types(types);
    proc_macro::TokenStream::from(res)
}

#[proc_macro]
pub fn define_float_kind(types: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let types = proc_macro2::TokenStream::from(types);
    let res = r#type::floats::define_float_kind(types);
    proc_macro::TokenStream::from(res)
}

#[proc_macro]
pub fn define_builtin_attrs(types: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let attrs = proc_macro2::TokenStream::from(types);
    let res = attr::define_builtin_attrs(attrs);
    proc_macro::TokenStream::from(res)
}
