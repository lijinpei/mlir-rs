use crate::support::*;
use mlir_capi::Support::*;

type CBool = u8;

pub fn to_cbool(b: bool) -> CBool {
    if b {
        1
    } else {
        0
    }
}

pub fn to_rbool(cb: CBool) -> bool {
    if cb == 0 {
        false
    } else {
        true
    }
}

pub fn to_string_ref(s: &str) -> MlirStringRef {
    MlirStringRef {
        data: s.as_ptr() as _,
        length: s.len() as _,
    }
}

pub fn is_null<T>(ptr: *const T) -> bool {
    ptr == std::ptr::null()
}

pub(crate) extern "C" fn print_helper(
    s: mlir_capi::Support::MlirStringRef,
    ptr: *mut std::ffi::c_void,
) {
    let ptr_to_callback = ptr as *mut &mut dyn PrintCallback;
    let callback: &mut dyn PrintCallback = unsafe { *ptr_to_callback };
    let str_ref = StrRef::from_ffi(s);
    callback.print(str_ref.to_str());
}
