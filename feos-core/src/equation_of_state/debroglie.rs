use num_dual::*;
use std::fmt;
use ndarray::Array1;

pub trait DeBroglieWavelengthDual<D: DualNum<f64>> {
    fn evaluate(&self, temperature: D) -> Array1<D>;
}

pub trait DeBroglieWavelength:
    DeBroglieWavelengthDual<f64>
    + DeBroglieWavelengthDual<Dual64>
    + DeBroglieWavelengthDual<Dual<DualVec64<3>, f64>>
    + DeBroglieWavelengthDual<HyperDual64>
    + DeBroglieWavelengthDual<Dual2_64>
    + DeBroglieWavelengthDual<Dual3_64>
    + DeBroglieWavelengthDual<HyperDual<Dual64, f64>>
    + DeBroglieWavelengthDual<HyperDual<DualVec64<2>, f64>>
    + DeBroglieWavelengthDual<HyperDual<DualVec64<3>, f64>>
    + DeBroglieWavelengthDual<Dual3<Dual64, f64>>
    + DeBroglieWavelengthDual<Dual3<DualVec64<2>, f64>>
    + DeBroglieWavelengthDual<Dual3<DualVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T> DeBroglieWavelength for T where
    T: DeBroglieWavelengthDual<f64>
        + DeBroglieWavelengthDual<Dual64>
        + DeBroglieWavelengthDual<Dual<DualVec64<3>, f64>>
        + DeBroglieWavelengthDual<HyperDual64>
        + DeBroglieWavelengthDual<Dual2_64>
        + DeBroglieWavelengthDual<Dual3_64>
        + DeBroglieWavelengthDual<HyperDual<Dual64, f64>>
        + DeBroglieWavelengthDual<HyperDual<DualVec64<2>, f64>>
        + DeBroglieWavelengthDual<HyperDual<DualVec64<3>, f64>>
        + DeBroglieWavelengthDual<Dual3<Dual64, f64>>
        + DeBroglieWavelengthDual<Dual3<DualVec64<2>, f64>>
        + DeBroglieWavelengthDual<Dual3<DualVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}


