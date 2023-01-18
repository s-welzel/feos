use std::marker;

use self::residual::{HelmholtzEnergy, HelmholtzEnergyDual};
use crate::{EosUnit, StateHD};
use ideal_gas::{DefaultIdealGas, IdealGas};
use ndarray::Array1;
use num_dual::DualNum;
use quantity::{
    si::{SIArray1, SIUnit, MOL},
    QuantityArray1,
};
use residual::Residual;

pub mod ideal_gas;
pub mod residual;

/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight {
    fn molar_weight(&self) -> SIArray1;
}

pub struct EquationOfState<I: IdealGas, R: Residual> {
    ideal_gas: I,
    residual: R,
}

impl<R: Residual> EquationOfState<DefaultIdealGas, R> {
    pub fn new_default_ideal_gas(residual: R) -> Self {
        Self {
            ideal_gas: DefaultIdealGas(residual.components()),
            residual,
        }
    }
}

impl<I: IdealGas, R: Residual> EquationOfState<I, R> {
    pub fn new(residual: R, ideal_gas: I) -> Self {
        Self {
            ideal_gas,
            residual,
        }
    }

    pub fn residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.residual.helmholtz_energy(state)
    }
}

impl<I: IdealGas, R: Residual + MolarWeight> EquationOfState<I, R> {
    fn molar_weight(&self) -> Array1<f64> {
        self.residual.molar_weight().to_reduced(MOL).unwrap()
    }
}
