
use crate::{EosError, EosResult, EosUnit, StateHD};
use ndarray::{Array, Array1};
use num_dual::DualNum;
use quantity::si::{SIArray1, SINumber, SIUnit, MOL};

pub mod debroglie;
pub mod helmholtz_energy;
pub mod ideal_gas;
pub mod residual;
pub use helmholtz_energy::{HelmholtzEnergy, HelmholtzEnergyDual};
pub use ideal_gas::{DefaultIdealGas, IdealGas};
pub use residual::Residual;

pub use self::debroglie::{DeBroglieWavelength, DeBroglieWavelengthDual};
/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight {
    fn molar_weight(&self) -> SIArray1;
}

#[derive(Debug, Clone)]
pub struct EquationOfState<I: IdealGas, R: Residual> {
    pub ideal_gas: I,
    pub residual: R,
    components: usize,
}

impl<R: Residual> EquationOfState<DefaultIdealGas, R> {
    pub fn new_default_ideal_gas(residual: R) -> Self {
        let components = residual.components();
        Self::new(DefaultIdealGas::new(components), residual)
    }
}

impl<I: IdealGas, R: Residual> EquationOfState<I, R> {
    pub fn new(ideal_gas: I, residual: R) -> Self {
        assert_eq!(residual.components(), ideal_gas.components());
        let components = residual.components();
        Self {
            ideal_gas,
            residual,
            components,
        }
    }

    pub fn components(&self) -> usize {
        self.components
    }

    pub fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(
            self.ideal_gas.subset(component_list),
            self.residual.subset(component_list),
        )
    }

    /// Check if the provided optional mole number is consistent with the
    /// equation of state.
    ///
    /// In general, the number of elements in `moles` needs to match the number
    /// of components of the equation of state. For a pure component, however,
    /// no moles need to be provided. In that case, it is set to the constant
    /// reference value.
    pub fn validate_moles(&self, moles: Option<&SIArray1>) -> EosResult<SIArray1> {
        let l = moles.map_or(1, |m| m.len());
        if self.components() == l {
            match moles {
                Some(m) => Ok(m.to_owned()),
                None => Ok(Array::ones(1) * SIUnit::reference_moles()),
            }
        } else {
            Err(EosError::IncompatibleComponents(self.components(), l))
        }
    }

    /// Calculate the maximum density.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    pub fn max_density(&self, moles: Option<&SIArray1>) -> EosResult<SINumber> {
        let mr = self
            .residual
            .validate_moles(moles)?
            .to_reduced(SIUnit::reference_moles())?;
        Ok(self.residual.compute_max_density(&mr) * SIUnit::reference_density())
    }

    pub fn evaluate_residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.residual.helmholtz_energy(state)
    }

    pub fn evaluate_ideal_gas<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn DeBroglieWavelength: DeBroglieWavelengthDual<D>,
    {
        self.ideal_gas.helmholtz_energy(state)
    }
}

impl<I: IdealGas, R: Residual + MolarWeight> EquationOfState<I, R> {
    pub fn molar_weight(&self) -> Array1<f64> {
        self.residual.molar_weight().to_reduced(MOL).unwrap()
    }
}
