use crate::common;
use crate::common::*;
use crate::context::*;
use crate::support::*;
use mlir_capi;
use mlir_capi::Dialect_::*;
use mlir_capi::IR::*;
use std::cmp::{Eq, PartialEq};
use std::convert::{From, Into};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Dialect<'ctx> {
    pub handle: MlirDialect,
    phantom: PhantomData<&'ctx Context>,
}

impl<'ctx> PartialEq<Dialect<'ctx>> for Dialect<'ctx> {
    fn eq(&self, other: &Dialect<'ctx>) -> bool {
        to_rbool(unsafe { mlir_capi::IR::FFIVal_::mlirDialectEqual(self, other) })
    }
}
impl<'ctx> Eq for Dialect<'ctx> {}

impl<'ctx> From<MlirDialect> for Dialect<'ctx> {
    fn from(value: MlirDialect) -> Self {
        Self {
            handle: value,
            phantom: PhantomData::default(),
        }
    }
}

impl<'ctx> Into<MlirDialect> for &Dialect<'ctx> {
    fn into(self) -> MlirDialect {
        self.handle
    }
}

impl<'ctx> Dialect<'ctx> {
    pub fn get_namespace(self) -> &'ctx str {
        let str_ref: StrRef<'ctx> =
            unsafe { mlir_capi::IR::FFIVal_::<_>::mlirDialectGetNamespace(&self) };
        str_ref.into()
    }
    pub fn is_null(self) -> bool {
        common::is_null(self.handle.ptr)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DialectHandle {
    pub handle: MlirDialectHandle,
}

impl From<MlirDialectHandle> for DialectHandle {
    fn from(value: MlirDialectHandle) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirDialectHandle> for &DialectHandle {
    fn into(self) -> MlirDialectHandle {
        self.handle
    }
}

impl DialectHandle {
    pub fn insert_dialect(self, reg: &DialectRegistry) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirDialectHandleInsertDialect(&self, reg);
        }
    }
    pub fn register_dialect(self, ctx: &Context) {
        unsafe {
            mlir_capi::IR::FFIVoid_::mlirDialectHandleRegisterDialect(&self, ctx);
        }
    }
    pub fn load_dialect<'ctx>(self, ctx: &'ctx Context) -> Dialect<'ctx> {
        unsafe { mlir_capi::IR::FFIVal_::mlirDialectHandleLoadDialect(&self, ctx) }
    }
}

#[repr(C)]
pub struct DialectRegistry {
    pub handle: MlirDialectRegistry,
}

impl From<MlirDialectRegistry> for DialectRegistry {
    fn from(value: MlirDialectRegistry) -> Self {
        Self { handle: value }
    }
}

impl Into<MlirDialectRegistry> for &DialectRegistry {
    fn into(self) -> MlirDialectRegistry {
        self.handle
    }
}

impl DialectRegistry {
    pub fn create() -> Self {
        unsafe { mlir_capi::IR::FFIVal_::mlirDialectRegistryCreate() }
    }
    pub fn is_null(self) -> bool {
        common::is_null(self.handle.ptr)
    }
}

impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe {
            FFIVoid_::mlirDialectRegistryDestroy(self as &_);
        }
    }
}

#[cfg(test)]
mod dialect_registry_test {
    use super::*;
    #[test]
    fn create() {
        let reg = DialectRegistry::create();
        assert!(!reg.is_null());
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum UpstreamDialectName {
    AMDGPU,
    Arith,
    Async,
    ControlFlow,
    Func,
    GPU,
    IRDL,
    Linalg,
    LLVM,
    Math,
    MemRef,
    MLProgram,
    NVGPU,
    NVVM,
    OpenMP,
    PDL,
    Quant,
    ROCDL,
    SCF,
    Shape,
    SparseTensor,
    SPIRV,
    Tensor,
    Transform,
    Vector,
}

pub fn get_handle_for_upstream_dialect(dialect: UpstreamDialectName) -> DialectHandle {
    let handle = unsafe {
        match dialect {
            UpstreamDialectName::AMDGPU => AMDGPU::mlirGetDialectHandle__amdgpu__(),
            UpstreamDialectName::Arith => Arith::mlirGetDialectHandle__arith__(),
            UpstreamDialectName::Async => Async::mlirGetDialectHandle__async__(),
            UpstreamDialectName::ControlFlow => ControlFlow::mlirGetDialectHandle__cf__(),
            UpstreamDialectName::Func => Func::mlirGetDialectHandle__func__(),
            UpstreamDialectName::GPU => GPU::mlirGetDialectHandle__gpu__(),
            UpstreamDialectName::IRDL => IRDL::mlirGetDialectHandle__irdl__(),
            UpstreamDialectName::Linalg => Linalg::mlirGetDialectHandle__linalg__(),
            UpstreamDialectName::LLVM => LLVM::mlirGetDialectHandle__llvm__(),
            UpstreamDialectName::Math => Math::mlirGetDialectHandle__math__(),
            UpstreamDialectName::MemRef => MemRef::mlirGetDialectHandle__memref__(),
            UpstreamDialectName::MLProgram => MLProgram::mlirGetDialectHandle__ml_program__(),
            UpstreamDialectName::NVGPU => NVGPU::mlirGetDialectHandle__nvgpu__(),
            UpstreamDialectName::NVVM => NVVM::mlirGetDialectHandle__nvvm__(),
            UpstreamDialectName::OpenMP => OpenMP::mlirGetDialectHandle__omp__(),
            UpstreamDialectName::PDL => PDL::mlirGetDialectHandle__pdl__(),
            UpstreamDialectName::Quant => Quant::mlirGetDialectHandle__quant__(),
            UpstreamDialectName::ROCDL => ROCDL::mlirGetDialectHandle__rocdl__(),
            UpstreamDialectName::SCF => SCF::mlirGetDialectHandle__scf__(),
            UpstreamDialectName::Shape => Shape::mlirGetDialectHandle__shape__(),
            UpstreamDialectName::SparseTensor => {
                SparseTensor::mlirGetDialectHandle__sparse_tensor__()
            }
            UpstreamDialectName::SPIRV => SPIRV::mlirGetDialectHandle__spirv__(),
            UpstreamDialectName::Tensor => Tensor::mlirGetDialectHandle__tensor__(),
            UpstreamDialectName::Transform => Transform::mlirGetDialectHandle__transform__(),
            UpstreamDialectName::Vector => Vector::mlirGetDialectHandle__vector__(),
        }
    };
    DialectHandle { handle }
}

#[cfg(test)]
pub mod dialect_test {
    use super::*;
    pub fn get_all_dialect_info() -> Vec<(UpstreamDialectName, &'static str)> {
        [
            (UpstreamDialectName::AMDGPU, "amdgpu"),
            (UpstreamDialectName::Arith, "arith"),
            (UpstreamDialectName::Async, "async"),
            (UpstreamDialectName::ControlFlow, "cf"),
            (UpstreamDialectName::Func, "func"),
            (UpstreamDialectName::GPU, "gpu"),
            (UpstreamDialectName::IRDL, "irdl"),
            (UpstreamDialectName::Linalg, "linalg"),
            (UpstreamDialectName::LLVM, "llvm"),
            (UpstreamDialectName::Math, "math"),
            (UpstreamDialectName::MemRef, "memref"),
            (UpstreamDialectName::MLProgram, "ml_program"),
            (UpstreamDialectName::NVGPU, "nvgpu"),
            (UpstreamDialectName::NVVM, "nvvm"),
            (UpstreamDialectName::OpenMP, "omp"),
            (UpstreamDialectName::PDL, "pdl"),
            (UpstreamDialectName::SCF, "scf"),
            (UpstreamDialectName::Shape, "shape"),
            (UpstreamDialectName::SparseTensor, "sparse_tensor"),
            (UpstreamDialectName::SPIRV, "spirv"),
            (UpstreamDialectName::Tensor, "tensor"),
            (UpstreamDialectName::Transform, "transform"),
            (UpstreamDialectName::Vector, "vector"),
        ]
        .to_vec()
    }
    #[test]
    fn dialect_load() {
        let infos = get_all_dialect_info();
        let ctx = Context::create();
        let mut dialects = Vec::new();
        for info in infos {
            let dialect_handle = get_handle_for_upstream_dialect(info.0);
            dialect_handle.load_dialect(&ctx);
            let dialect = ctx.get_or_load_dialect(info.1);
            assert!(!dialect.is_null());
            assert_eq!(dialect.get_namespace(), info.1);
            dialects.push(dialect);
        }
        let num_dialects = dialects.len();
        for i in 0..num_dialects {
            let d_i = dialects[i];
            for j in 0..num_dialects {
                let d_j = dialects[j];
                assert_eq!(i == j, d_i == d_j);
            }
        }
    }
}
