#![feature(ptr_as_ref_unchecked)]

pub mod affine_expr;
pub mod affine_map;
pub mod asm_state;
pub mod attribute;
pub mod block;
pub mod builder;
pub mod common;
pub mod context;
pub mod dialect;
pub mod integer_set;
pub mod location;
pub mod module;
pub mod op_printing_flags;
pub mod operation;
pub mod operation_state;
pub mod region;
pub mod support;
pub mod symbol_table;
pub mod r#type;
pub mod value;

pub mod type_cast;
