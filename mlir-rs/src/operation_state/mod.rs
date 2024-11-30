use crate::attribute::*;
use crate::block::*;
use crate::common::*;
use crate::location::*;
use crate::r#type::*;
use crate::region::*;
use crate::support::*;
use crate::value::*;

use mlir_capi::IR::MlirOperationState;

pub struct OperationState<'ctx> {
    pub name: String,
    pub location: Location<'ctx>,
    pub results: Vec<Type<'ctx>>,
    pub operands: Vec<Value<'ctx>>,
    pub regions: Vec<Region<'ctx>>,
    pub successors: Vec<Block<'ctx>>,
    pub attributes: Vec<NamedAttr<'ctx>>,
    pub enable_result_type_inference: bool,
}

impl<'ctx> Into<MlirOperationState> for &OperationState<'ctx> {
    fn into(self) -> MlirOperationState {
        MlirOperationState {
            name: Into::<StrRef>::into(self.name.as_str()).into(),
            location: self.location.into(),
            nResults: self.results.len() as _,
            results: self.results.as_ptr() as *mut _,
            nOperands: self.operands.len() as _,
            operands: self.operands.as_ptr() as *mut _,
            nRegions: self.regions.len() as _,
            regions: self.regions.as_ptr() as *mut _,
            nSuccessors: self.successors.len() as _,
            successors: self.successors.as_ptr() as *mut _,
            nAttributes: self.attributes.len() as _,
            attributes: self.attributes.as_ptr() as *mut _,
            enableResultTypeInference: to_cbool(self.enable_result_type_inference),
        }
    }
}
