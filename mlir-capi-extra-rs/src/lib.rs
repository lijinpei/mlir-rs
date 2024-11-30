use mlir_capi::IR::*;

#[link(name = "MLIR-C-Extra")]
extern "C" {

    pub fn mlirContextIsMultithreadingEnabled(ctx: MlirContext) -> u8;
    pub fn mlirTypeIsIntegerType(r#type: MlirType) -> u8;

}
