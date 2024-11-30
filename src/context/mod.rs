use mlir_capi::IR::MlirContext;
use std::cell::Cell;
use std::option::Option;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Context {
    pub handle: MlirContext,
}

thread_local! {
    pub static CTX: Cell<Option<Context>> = Cell::new(Option::None);
}

impl Context {
    pub fn init() {
        assert!(CTX.get().is_none());
        let handle = unsafe { mlir_capi::IR::mlirContextCreate() };
        CTX.set(Some(Context { handle: handle }));
    }
    pub fn fini() {
        if let Some(ctx) = CTX.get() {
            CTX.set(None);
            unsafe {
                mlir_capi::IR::mlirContextDestroy(ctx.handle);
            }
        }
    }
    pub fn get() -> Context {
        CTX.get().unwrap()
    }
    pub fn set(ctx: Context) -> Option<Context> {
        CTX.replace(Some(ctx))
    }
    pub fn unset() -> Option<Context> {
        CTX.replace(None)
    }
    pub fn to_ffi(self) -> MlirContext {
        self.handle
    }
    pub fn from_ffi(handle: MlirContext) -> Self {
        Self { handle }
    }
}

impl From<MlirContext> for Context {
    fn from(value: MlirContext) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirContext> for &Context {
    fn into(self) -> MlirContext {
        self.handle
    }
}
