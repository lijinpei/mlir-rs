use crate::common;
use crate::common::{to_cbool, to_rbool, to_string_ref};
use crate::dialect::Dialect;
use crate::dialect::DialectRegistry;
use mlir_capi::IR::MlirContext;
use mlir_capi::IR::*;
use std::cmp::{Eq, PartialEq};
use std::convert::Into;
use std::marker::PhantomData;

#[repr(C)]
pub struct Context {
    pub handle: MlirContext,
}

impl Context {
    pub fn create() -> Self {
        let handle = unsafe { FFIVal_::mlirContextCreate() };
        Context { handle }
    }

    pub fn create_with_threading(threading_enabled: bool) -> Self {
        let te = to_cbool(threading_enabled);
        let handle = unsafe { FFIVal_::mlirContextCreateWithThreading(te) };
        Context { handle }
    }

    pub fn create_with_registry(registry: &DialectRegistry, threading_enabled: bool) -> Self {
        let te = to_cbool(threading_enabled);
        let handle = unsafe { FFIVal_::mlirContextCreateWithRegistry(registry, te) };
        Context { handle }
    }

    pub fn is_null(&self) -> bool {
        common::is_null(self.handle.ptr)
    }

    pub fn set_allow_unregistered_dialects(&self, allow: bool) {
        let c_allow = to_cbool(allow);
        unsafe {
            FFIVoid_::mlirContextSetAllowUnregisteredDialects(self, c_allow);
        }
    }

    pub fn get_allow_unregistered_dialects(&self) -> bool {
        let res = unsafe { FFIVal_::mlirContextGetAllowUnregisteredDialects(self) };
        to_rbool(res)
    }

    pub fn get_num_registered_dialects(&self) -> usize {
        (unsafe { FFIVal_::<i64>::mlirContextGetNumRegisteredDialects(self) }) as _
    }

    pub fn append_dialect_registry(&self, registry: &DialectRegistry) {
        unsafe {
            FFIVoid_::mlirContextAppendDialectRegistry(self, registry);
        }
    }

    pub fn get_num_loaded_dialects(&self) -> usize {
        (unsafe { FFIVal_::<i64>::mlirContextGetNumLoadedDialects(self) }) as _
    }

    pub fn get_or_load_dialect(&self, name: &str) -> Dialect {
        unsafe { FFIVal_::mlirContextGetOrLoadDialect(self, to_string_ref(name)) }
    }

    pub fn enable_threading(&self, enable: bool) {
        let c_enable = to_cbool(enable);
        unsafe {
            FFIVoid_::mlirContextEnableMultithreading(self, c_enable);
        }
    }

    pub fn is_threading_enabled(&self) -> bool {
        // FIXME: need MLIR-C-Extra
        let c_res = unsafe { mlir_capi_extra::mlirContextIsMultithreadingEnabled(self.handle) };
        to_rbool(c_res)
    }

    pub fn load_all_available_dialects(&self) {
        unsafe {
            FFIVoid_::mlirContextLoadAllAvailableDialects(self);
        }
    }

    pub fn is_registered_operation(&self, name: &str) -> bool {
        let res = unsafe { FFIVal_::mlirContextIsRegisteredOperation(self, to_string_ref(name)) };
        to_rbool(res)
    }
    // FIXME: support mlirContextSetThreadPool
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            FFIVoid_::mlirContextDestroy(self as &_);
        }
    }
}

impl PartialEq<Context> for Context {
    fn eq(&self, other: &Context) -> bool {
        let res = unsafe { FFIVal_::mlirContextEqual(self, other) };
        to_rbool(res)
    }
}
impl<'ctx> PartialEq<ContextRef<'ctx>> for Context {
    fn eq(&self, other: &ContextRef<'ctx>) -> bool {
        let res = unsafe { FFIVal_::mlirContextEqual(self, *other) };
        to_rbool(res)
    }
}
impl Eq for Context {}

impl Into<MlirContext> for &Context {
    fn into(self) -> MlirContext {
        self.handle
    }
}

pub trait HandleWithContext<'ctx>: Sized {
    type HandleTy: Copy;
    fn get_context_handle(&self) -> MlirContext;
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self;
    fn get_phantom(&self) -> PhantomData<&'ctx Context> {
        PhantomData::default()
    }
    fn from_handle_same_context<T: HandleWithContext<'ctx>>(
        handle: Self::HandleTy,
        other: &T,
    ) -> Self {
        unsafe { Self::from_handle_and_phantom(handle, other.get_phantom()) }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContextRef<'ctx> {
    pub handle: MlirContext,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> std::ops::Deref for ContextRef<'ctx> {
    type Target = Context;
    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(self) }
    }
}

impl<'ctx> HandleWithContext<'ctx> for ContextRef<'ctx> {
    type HandleTy = MlirContext;
    fn get_context_handle(&self) -> MlirContext {
        self.handle
    }
    unsafe fn from_handle_and_phantom(
        handle: Self::HandleTy,
        phantom: PhantomData<&'ctx Context>,
    ) -> Self {
        Self { handle, phantom }
    }
}

impl<'ctx> Into<MlirContext> for ContextRef<'ctx> {
    fn into(self) -> MlirContext {
        self.handle
    }
}

impl<'ctx> PartialEq<Context> for ContextRef<'ctx> {
    fn eq(&self, other: &Context) -> bool {
        let res = unsafe { FFIVal_::mlirContextEqual(*self, other) };
        to_rbool(res)
    }
}

impl<'ctx> PartialEq<ContextRef<'ctx>> for ContextRef<'ctx> {
    fn eq(&self, other: &ContextRef<'ctx>) -> bool {
        let res = unsafe { FFIVal_::mlirContextEqual(*self, *other) };
        to_rbool(res)
    }
}
impl<'ctx> Eq for ContextRef<'ctx> {}

#[cfg(test)]
pub mod context_test {
    use super::*;
    use crate::dialect::*;

    #[test]
    fn create() {
        let ctx = Context::create();
        assert!(!ctx.is_null());
        let ctx_ref = &ctx;
        assert!(*ctx_ref == ctx);
    }

    #[test]
    fn create_with_threading() {
        let ctx1 = Context::create_with_threading(true);
        assert!(!ctx1.is_null());
        let ctx2 = Context::create_with_threading(false);
        assert!(!ctx2.is_null());
        assert!(ctx1 != ctx2);
        assert!(ctx2 != ctx1);

        assert!(ctx1.is_threading_enabled());
        ctx1.enable_threading(false);
        assert!(!ctx1.is_threading_enabled());
        ctx1.enable_threading(true);
        assert!(ctx1.is_threading_enabled());

        assert!(!ctx2.is_threading_enabled());
        ctx2.enable_threading(true);
        assert!(ctx2.is_threading_enabled());
        ctx2.enable_threading(false);
        assert!(!ctx2.is_threading_enabled());
    }

    #[test]
    fn create_with_registry() {
        let amdgpu_handle = get_handle_for_upstream_dialect(UpstreamDialectName::AMDGPU);
        let arith_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Arith);
        let vector_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Vector);
        let reg1 = DialectRegistry::create();
        amdgpu_handle.insert_dialect(&reg1);
        arith_handle.insert_dialect(&reg1);
        let reg2 = DialectRegistry::create();
        vector_handle.insert_dialect(&reg2);
        let ctx1 = Context::create_with_registry(&reg1, true);
        let ctx2 = Context::create_with_registry(&reg1, false);
        assert!(ctx1.is_threading_enabled());
        assert!(!ctx2.is_threading_enabled());
        assert!(3 <= ctx1.get_num_registered_dialects());
        assert!(2 <= ctx2.get_num_registered_dialects());
        assert_eq!(1, ctx1.get_num_loaded_dialects());
        assert_eq!(1, ctx2.get_num_loaded_dialects());
    }

    #[test]
    fn allow_unregistred_dialect() {
        let ctx = Context::create();
        ctx.get_allow_unregistered_dialects();
        ctx.set_allow_unregistered_dialects(true);
        assert!(ctx.get_allow_unregistered_dialects());
        assert_eq!(1, ctx.get_num_loaded_dialects());
        ctx.set_allow_unregistered_dialects(false);
        assert!(!ctx.get_allow_unregistered_dialects());
        assert_eq!(1, ctx.get_num_loaded_dialects());
    }

    #[test]
    fn num_registered_dialects() {
        let async_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Async);
        let ctx = Context::create();
        assert!(1 == ctx.get_num_registered_dialects());
        async_handle.register_dialect(&ctx);
        assert!(1 < ctx.get_num_registered_dialects());
    }

    #[test]
    fn get_or_load_dialect() {
        let amdgpu_handle = get_handle_for_upstream_dialect(UpstreamDialectName::AMDGPU);
        let arith_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Arith);
        let reg = DialectRegistry::create();
        amdgpu_handle.insert_dialect(&reg);
        arith_handle.insert_dialect(&reg);
        let ctx = Context::create();
        let amdgpu_dialect = ctx.get_or_load_dialect("amdgpu");
        assert!(amdgpu_dialect.is_null());
        let arith_dialect = ctx.get_or_load_dialect("arith");
        assert!(arith_dialect.is_null());
        ctx.append_dialect_registry(&reg);
        let amdgpu_dialect = ctx.get_or_load_dialect("amdgpu");
        assert!(!amdgpu_dialect.is_null());
        let arith_dialect = ctx.get_or_load_dialect("arith");
        assert!(!arith_dialect.is_null());
    }

    #[test]
    fn is_registered_operation() {
        let ctx = Context::create();
        assert!(!ctx.is_registered_operation("vector.load"));
        assert!(!ctx.is_registered_operation("vector.foobar"));
        let vector_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Vector);
        vector_handle.load_dialect(&ctx);
        ctx.load_all_available_dialects();
        assert!(ctx.is_registered_operation("vector.load"));
        assert!(!ctx.is_registered_operation("vector.foobar"));
    }

    #[test]
    fn load_all_available_dialects() {
        let ctx = Context::create();
        assert!(!ctx.is_registered_operation("func.func"));
        assert!(!ctx.is_registered_operation("func.foobar"));
        let func_handle = get_handle_for_upstream_dialect(UpstreamDialectName::Func);
        func_handle.register_dialect(&ctx);
        assert!(!ctx.is_registered_operation("func.func"));
        assert!(!ctx.is_registered_operation("func.foobar"));
        ctx.load_all_available_dialects();
        assert!(ctx.is_registered_operation("func.func"));
        assert!(!ctx.is_registered_operation("func.foobar"));
    }

    use crate::dialect::dialect_test::*;
    pub fn create_context_with_all_upstream_dialects() -> Context {
        let ctx = Context::create();
        for info in get_all_dialect_info() {
            get_handle_for_upstream_dialect(info.0).load_dialect(&ctx);
        }
        ctx
    }
}
