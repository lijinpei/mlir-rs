use crate::attribute::*;
use crate::context::*;
use crate::operation::*;
use crate::support::*;

use mlir_capi::IR;
use mlir_capi::IR::*;

use std::marker::PhantomData;

#[repr(C)]
pub struct SymbolTable<'op, 'ctx> {
    pub handle: MlirSymbolTable,
    _phantom: PhantomData<&'op Operation<'ctx>>,
}

impl<'op, 'ctx> HandleWithContext<'ctx> for SymbolTable<'op, 'ctx> {
    type HandleTy = MlirSymbolTable;
    fn get_context_handle(&self) -> MlirContext {
        // FIXME
        unimplemented!()
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        _phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self {
            handle: handle,
            _phantom: PhantomData::default(),
        }
    }
}

impl<'op, 'ctx> Into<MlirSymbolTable> for &SymbolTable<'op, 'ctx> {
    fn into(self) -> MlirSymbolTable {
        self.handle
    }
}

impl<'op, 'ctx> SymbolTable<'op, 'ctx> {
    pub fn get_symbol_attr_name() -> String {
        let str_ref: StrRef = unsafe { IR::FFIVal_::mlirSymbolTableGetSymbolAttributeName() };
        str_ref.to_str().into()
    }
    pub fn get_visibility_attr_name() -> String {
        let str_ref: StrRef = unsafe { IR::FFIVal_::mlirSymbolTableGetVisibilityAttributeName() };
        str_ref.to_str().into()
    }

    pub fn create(op: &'op Operation<'ctx>) -> Self {
        let handle = unsafe { IR::FFIVal_::mlirSymbolTableCreate(op) };
        Self {
            handle,
            _phantom: PhantomData::default(),
        }
    }

    pub fn lookup(&self, name: &str) -> OperationRef<'ctx> {
        let name_ref: StrRef = name.into();
        let handle = unsafe { IR::FFIVal_::mlirSymbolTableLookup(self, name_ref) };
        unsafe { OperationRef::from_handle_and_phantom(handle, PhantomData::default()) }
    }

    pub fn insert(&self, op: &Operation<'ctx>) -> Attr<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirSymbolTableInsert(self, op) };
        Attr::from_handle_same_context(handle, self)
    }

    pub fn erase(&self, op: &Operation<'ctx>) {
        unsafe { IR::FFIVoid_::mlirSymbolTableErase(self, op) }
    }
    pub fn replace_all_symbol_uses(
        old: &str,
        new: &str,
        op: &'op Operation<'ctx>,
    ) -> LogicalResult {
        let old_ref: StrRef = old.into();
        let new_ref: StrRef = new.into();
        unsafe { IR::FFIVal_::mlirSymbolTableReplaceAllSymbolUses(old_ref, new_ref, op) }
    }

    // FIXME: is null

    // FIXME: mlirSymbolTableWalkSymbolTables
}

impl<'op, 'ctx> Drop for SymbolTable<'op, 'ctx> {
    fn drop(&mut self) {
        unsafe {
            IR::FFIVoid_::mlirSymbolTableDestroy(&*self);
        }
    }
}
