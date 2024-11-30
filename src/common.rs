use mlir_capi::Support::MlirStringRef;

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
