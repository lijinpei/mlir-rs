pub trait NullableRef: Copy {
    fn is_null(self) -> bool;
    fn create_null() -> Self;
}

pub trait NullableOwned {
    fn is_null(&self) -> bool;
    fn create_null() -> Self;
}

pub trait IsA<T: NullableRef>: NullableRef {
    fn is_a_impl(self) -> bool;
    unsafe fn cast(self) -> T;
    fn is_a_non_null(self) -> bool {
        if self.is_null() {
            false
        } else {
            self.is_a_impl()
        }
    }
    fn is_a(self) -> bool {
        if self.is_null() {
            true
        } else {
            self.is_a_impl()
        }
    }
    fn dyn_cast(self) -> T {
        debug_assert!(!self.is_null());
        if self.is_a() {
            unsafe { self.cast() }
        } else {
            T::create_null()
        }
    }
    fn dyn_cast_or_null(self) -> T {
        if self.is_null() {
            T::create_null()
        } else {
            self.dyn_cast()
        }
    }
}

impl<T: NullableRef> IsA<T> for T {
    fn is_a_impl(self) -> bool {
        true
    }
    unsafe fn cast(self) -> T {
        self
    }
}

pub struct DownCastError {
    pub message: String,
}
