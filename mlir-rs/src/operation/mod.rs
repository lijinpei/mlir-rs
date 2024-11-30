use crate::asm_state::*;
use crate::attribute::*;
use crate::block::*;
use crate::common::*;
use crate::context::*;
use crate::location::*;
use crate::op_printing_flags::*;
use crate::operation_state::*;
use crate::region::*;
use crate::support::*;
use crate::value::*;

use mlir_capi;
use mlir_capi::IR;
use mlir_capi::IR::*;
use std::convert::Into;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

#[repr(C)]
pub struct Operation<'ctx> {
    pub handle: MlirOperation,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirOperation> for &Operation<'ctx> {
    fn into(self) -> MlirOperation {
        self.handle
    }
}

impl<'ctx> HandleWithContext<'ctx> for Operation<'ctx> {
    type HandleTy = MlirOperation;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { IR::FFIVal_::mlirOperationGetContext(self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> PartialEq for Operation<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        to_rbool(unsafe { mlir_capi::IR::FFIVal_::mlirOperationEqual(self, other) })
    }
}

impl<'ctx> Operation<'ctx> {
    pub fn create(state: &OperationState<'ctx>) -> Self {
        let mut op_state: MlirOperationState = state.into();
        let handle = unsafe { IR::FFIVal_::mlirOperationCreate((&mut op_state) as *mut _) };
        Self {
            handle: handle,
            phantom: PhantomData::default(),
        }
    }

    pub fn create_parse(ctx: &'ctx Context, source_str: &str, source_name: &str) -> Self {
        // TODO: it's better that mlir can parse input without nul terminator.
        let source_str = source_str.as_bytes();
        let mut src_vec = Vec::new();
        src_vec.reserve_exact(source_str.len() + 1);
        src_vec.extend_from_slice(source_str);
        src_vec.push(0);
        let source_str_ref = StrRef::from(&src_vec[..source_str.len()]);
        let source_name_ref: StrRef = source_name.into();
        let handle =
            unsafe { IR::FFIVal_::mlirOperationCreateParse(ctx, source_str_ref, source_name_ref) };
        Self {
            handle: handle,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Clone for Operation<'ctx> {
    fn clone(&self) -> Self {
        let handle = unsafe { IR::FFIVal_::mlirOperationClone(self) };
        Self {
            handle: handle,
            phantom: self.phantom,
        }
    }
}

impl<'ctx> Drop for Operation<'ctx> {
    fn drop(&mut self) {
        unsafe {
            FFIVoid_::mlirOperationDestroy(&*self);
        }
    }
}

#[cfg(test)]
mod operation_test {
    use super::*;
    use crate::dialect::*;
    use crate::r#type::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        let loc = Location::unknown_get(&ctx);
        let arith_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Arith);
        arith_handle.load_dialect(&ctx);
        let func_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Func);
        func_handle.load_dialect(&ctx);
        let i32_ty: Type = IntegerType::get(&ctx, 32).into();
        let constant_val = NamedAttr {
            name: Identifier::get(&ctx, "value"),
            attribute: IntegerAttr::get(i32_ty, 10).into(),
        };
        let c10_op = {
            let mut op_state = OperationState::get("arith.constant", loc);
            op_state.add_results(&[i32_ty]);
            op_state.add_attributes(&[constant_val]);
            Operation::create(&op_state)
        };
        let add_op = {
            let mut op_state = OperationState::get("arith.addi", loc);
            op_state.add_operands(&[c10_op.get_result(0), c10_op.get_result(0)]);
            op_state.enable_type_inference();
            Operation::create(&op_state)
        };
        assert!(!c10_op.is_null());
        assert!(c10_op.get_context() == ctx);
        assert!(!add_op.is_null());
        assert!(add_op.get_context() == ctx);
        assert!(c10_op.get_location() == loc);
        assert!(add_op.get_location() == loc);
        assert_eq!(c10_op.get_name(), Identifier::get(&ctx, "arith.constant"));
        assert_eq!(add_op.get_name(), Identifier::get(&ctx, "arith.addi"));
        // FIXME: test add_owned_regions add_successors
    }

    #[test]
    fn create_parse() {
        let ctx = Context::create();
        ctx.set_allow_unregistered_dialects(true);
        let _op1 = Operation::create_parse(
            &ctx,
            "%result:2 = \"foo_div\"() : () -> (f32, i32)",
            "unknown_source_01",
        );
        let func_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Func);
        func_handle.load_dialect(&ctx);
        let func_op = Operation::create_parse(
            &ctx,
            "func.func @count(%x: i64) -> (i64, i64)
  attributes {fruit = \"banana\"} {
  return %x, %x: i64, i64
}",
            "unknown_source_02",
        );
        assert_eq!(func_op.get_name(), Identifier::get(&ctx, "func.func"));
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OperationRef<'ctx> {
    pub handle: MlirOperation,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> Into<MlirOperation> for OperationRef<'ctx> {
    fn into(self) -> MlirOperation {
        self.handle
    }
}

impl<'ctx> PartialEq for OperationRef<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        to_rbool(unsafe { mlir_capi::IR::FFIVal_::mlirOperationEqual(*self, *other) })
    }
}

impl<'ctx> HandleWithContext<'ctx> for OperationRef<'ctx> {
    type HandleTy = MlirOperation;
    fn get_context_handle(&self) -> MlirContext {
        unsafe { IR::FFIVal_::mlirOperationGetContext(*self) }
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> OperationRef<'ctx> {
    pub fn remove_from_parent(self) -> Operation<'ctx> {
        unsafe { IR::FFIVoid_::mlirOperationRemoveFromParent(self) };
        Operation {
            handle: self.handle,
            phantom: self.phantom,
        }
    }
}

impl<'ctx> std::ops::Deref for OperationRef<'ctx> {
    type Target = Operation<'ctx>;

    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(self) }
    }
}

impl<'ctx> Operation<'ctx> {
    pub fn is_null(&self) -> bool {
        let handle: MlirOperation = self.into();
        handle.ptr == std::ptr::null_mut()
    }
    pub fn get_context(&self) -> ContextRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetContext(self) };
        ContextRef::from_handle_same_context(handle, self)
    }
    pub fn get_location(&self) -> Location<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirOperationGetLocation(self) };
        Location::from_handle_same_context(handle, self)
    }
    pub fn get_typeid(&self) -> TypeID {
        unsafe { IR::FFIVal_::mlirOperationGetTypeID(self) }
    }
    pub fn get_name(&self) -> Identifier<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetName(self) };
        Identifier::from_handle_same_context(handle, self)
    }
    pub fn get_block(&self) -> BlockRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetBlock(self) };
        unsafe { BlockRef::wrap(handle, PhantomData::default()) }
    }
    pub fn get_first_region(&self) -> RegionRef<'ctx> {
        let handle = unsafe { mlir_capi::IR::FFIVal_::mlirOperationGetFirstRegion(self) };
        unsafe { RegionRef::wrap(handle, self.phantom) }
    }
    pub fn get_parent_op(&self) -> OperationRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetParentOperation(self) };
        OperationRef::from_handle_same_context(handle, self)
    }
    pub fn get_num_regions(&self) -> usize {
        (unsafe { IR::FFIVal_::<i64>::mlirOperationGetNumRegions(self) }) as _
    }
    pub fn get_region(&self, pos: usize) -> RegionRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetRegion(self, pos as i64) };
        unsafe { RegionRef::wrap(handle, self.phantom) }
    }
    pub fn get_next_in_block(&self) -> OperationRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetNextInBlock(self) };
        OperationRef::from_handle_same_context(handle, self)
    }
    pub fn get_num_operands(&self) -> usize {
        (unsafe { IR::FFIVal_::<i64>::mlirOperationGetNumOperands(self) }) as _
    }
    pub fn get_operand(&self, pos: usize) -> Value<'ctx> {
        unsafe { IR::FFIVal_::mlirOperationGetOperand(self, pos as i64) }
    }
    pub fn set_operand(&self, pos: usize, value: Value<'ctx>) {
        unsafe {
            IR::FFIVoid_::mlirOperationSetOperand(self, pos as i64, value);
        }
    }
    pub fn set_operands(&self, operands: &[Value<'ctx>]) {
        unsafe {
            IR::FFIVoid_::mlirOperationSetOperands(
                self,
                operands.len() as i64,
                operands.as_ptr() as *const _,
            )
        }
    }
    pub fn get_num_results(&self) -> usize {
        (unsafe { IR::FFIVal_::<i64>::mlirOperationGetNumResults(self) }) as _
    }
    pub fn get_result(&self, pos: usize) -> Value<'ctx> {
        unsafe { IR::FFIVal_::mlirOperationGetResult(self, pos as i64) }
    }
    pub fn get_num_successors(&self) -> usize {
        (unsafe { IR::FFIVal_::<i64>::mlirOperationGetNumSuccessors(self) }) as _
    }
    pub fn get_successor(&self, pos: usize) -> BlockRef<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetSuccessor(self, pos as i64) };
        unsafe { BlockRef::wrap(handle, self.phantom) }
    }
    pub fn set_succeccor(&self, pos: usize, block: BlockRef<'ctx>) {
        unsafe { IR::FFIVoid_::mlirOperationSetSuccessor(self, pos as i64, block) };
    }
    pub fn has_inherent_attr_by_name(&self, name: &str) -> bool {
        let name_ref: StrRef = name.into();
        to_rbool(unsafe { IR::FFIVal_::mlirOperationHasInherentAttributeByName(self, name_ref) })
    }
    pub fn remove_discardable_attr_by_name(&self, name: &str) -> bool {
        let name_ref: StrRef = name.into();
        to_rbool(unsafe {
            IR::FFIVal_::mlirOperationRemoveDiscardableAttributeByName(self, name_ref)
        })
    }
    pub fn get_num_attrs(&self) -> usize {
        (unsafe { IR::FFIVal_::<i64>::mlirOperationGetNumAttributes(self) }) as _
    }
    pub fn get_attr(&self, pos: usize) -> NamedAttr<'ctx> {
        let handle = unsafe { IR::FFIVal_::mlirOperationGetAttribute(self, pos as i64) };
        NamedAttr::from_handle_same_context(handle, self)
    }
    pub fn get_attr_by_name(&self, name: &str) -> Attr<'ctx> {
        let name_ref: StrRef = name.into();
        let handle = unsafe { IR::FFIVal_::mlirOperationGetAttributeByName(self, name_ref) };
        Attr::from_handle_same_context(handle, self)
    }
    pub fn set_attr_by_name(&self, name: &str, attr: Attr<'ctx>) {
        let name_ref: StrRef = name.into();
        unsafe { IR::FFIVoid_::mlirOperationSetAttributeByName(self, name_ref, attr) }
    }
    pub fn remove_attr_by_name(&self, name: &str) -> bool {
        let name_ref: StrRef = name.into();
        to_rbool(unsafe { IR::FFIVal_::mlirOperationRemoveAttributeByName(self, name_ref) })
    }
    pub fn print(&self, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirOperationPrint(
                self,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
    }
    pub fn print_with_flags(&self, flags: &OpPrintingFlags, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirOperationPrintWithFlags(
                self,
                flags,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
    }
    pub fn print_with_state(&self, state: &AsmState, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirOperationPrintWithState(
                self,
                state,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            );
        }
    }
    pub fn write_bytecode(&self, callback: &mut dyn PrintCallback) {
        unsafe {
            IR::FFIVoid_::mlirOperationWriteBytecode(
                self,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            )
        }
    }
    pub fn write_bytecode_with_config(
        &self,
        config: &ByteCodeWriterConfig,
        callback: &mut dyn PrintCallback,
    ) -> LogicalResult {
        unsafe {
            IR::FFIVal_::mlirOperationWriteBytecodeWithConfig(
                self,
                config,
                print_helper as *mut _,
                &callback as *const &mut dyn PrintCallback as *mut _,
            )
        }
    }
    pub fn dump(&self) {
        unsafe {
            IR::FFIVoid_::mlirOperationDump(self);
        }
    }
    pub fn verify(&self) -> bool {
        to_rbool(unsafe { IR::FFIVal_::mlirOperationVerify(self) })
    }
    pub fn move_after(&self, other: &Operation<'ctx>) {
        unsafe { IR::FFIVoid_::mlirOperationMoveAfter(self, other) }
    }
    pub fn move_before(&self, other: &Operation<'ctx>) {
        unsafe { IR::FFIVoid_::mlirOperationMoveBefore(self, other) }
    }
    // FIXME: mlirOperationWalk
    pub fn print_to_formatter(
        &self,
        formatter: &mut std::fmt::Formatter,
    ) -> Result<(), std::fmt::Error> {
        let mut printer = PrintToFormatter::new(formatter);
        self.print(&mut printer);
        Ok(())
    }
}

impl<'ctx> Debug for Operation<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

impl<'ctx> Display for Operation<'ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        self.print_to_formatter(formatter)
    }
}

#[repr(C)]
pub struct ByteCodeWriterConfig {
    handle: MlirBytecodeWriterConfig,
}

impl Into<MlirBytecodeWriterConfig> for &ByteCodeWriterConfig {
    fn into(self) -> MlirBytecodeWriterConfig {
        self.handle
    }
}

impl Drop for ByteCodeWriterConfig {
    fn drop(&mut self) {
        unsafe { IR::FFIVoid_::mlirBytecodeWriterConfigDestroy(&*self) }
    }
}

impl ByteCodeWriterConfig {
    pub fn create() -> Self {
        let handle = unsafe { IR::FFIVal_::mlirBytecodeWriterConfigCreate() };
        Self { handle }
    }
    pub fn set_desired_emit_version(&self, version: i64) {
        unsafe { IR::FFIVoid_::mlirBytecodeWriterConfigDesiredEmitVersion(self, version) }
    }
}
