use proc_macro;

mod r#type;

#[proc_macro]
pub fn define_builtin_types(types: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let types = proc_macro2::TokenStream::from(types);
    let res = r#type::define_builtin_types(types);
    proc_macro::TokenStream::from(res)
}
