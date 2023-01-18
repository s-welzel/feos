use crate::StateHD;
use crate::{EosError, EosResult, EosUnit};
use ndarray::prelude::*;
use num_dual::*;
use num_traits::{One, Zero};
use quantity::*;
use std::fmt;

/// Individual Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [HelmholtzEnergy]
/// so that the implementor can be used as a Helmholtz energy
/// contribution in the equation of state.
pub trait HelmholtzEnergyDual<D: DualNum<f64>> {
    /// The Helmholtz energy contribution $\beta A$ of a given state in reduced units.
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D;
}

/// Object safe version of the [HelmholtzEnergyDual] trait.
///
/// The trait is implemented automatically for every struct that implements
/// the supertraits.
pub trait HelmholtzEnergy:
    HelmholtzEnergyDual<f64>
    + HelmholtzEnergyDual<Dual64>
    + HelmholtzEnergyDual<Dual<DualVec64<3>, f64>>
    + HelmholtzEnergyDual<HyperDual64>
    + HelmholtzEnergyDual<Dual2_64>
    + HelmholtzEnergyDual<Dual3_64>
    + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
    + HelmholtzEnergyDual<HyperDual<DualVec64<2>, f64>>
    + HelmholtzEnergyDual<HyperDual<DualVec64<3>, f64>>
    + HelmholtzEnergyDual<Dual3<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<DualVec64<2>, f64>>
    + HelmholtzEnergyDual<Dual3<DualVec64<3>, f64>>
    + fmt::Display
    + Send
    + Sync
{
}

impl<T> HelmholtzEnergy for T where
    T: HelmholtzEnergyDual<f64>
        + HelmholtzEnergyDual<Dual64>
        + HelmholtzEnergyDual<Dual<DualVec64<3>, f64>>
        + HelmholtzEnergyDual<HyperDual64>
        + HelmholtzEnergyDual<Dual2_64>
        + HelmholtzEnergyDual<Dual3_64>
        + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
        + HelmholtzEnergyDual<HyperDual<DualVec64<2>, f64>>
        + HelmholtzEnergyDual<HyperDual<DualVec64<3>, f64>>
        + HelmholtzEnergyDual<Dual3<Dual64, f64>>
        + HelmholtzEnergyDual<Dual3<DualVec64<2>, f64>>
        + HelmholtzEnergyDual<Dual3<DualVec64<3>, f64>>
        + fmt::Display
        + Send
        + Sync
{
}

/// A general equation of state.
pub trait Residual: Send + Sync {
    /// Return the number of components of the equation of state.
    fn components(&self) -> usize;

    /// Return an equation of state consisting of the components
    /// contained in component_list.
    fn subset(&self, component_list: &[usize]) -> Self;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64;

    /// Return a slice of the individual contributions (excluding the ideal gas)
    /// of the equation of state.
    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>];

    /// Evaluate the residual reduced Helmholtz energy $\beta A^\mathrm{res}$.
    fn helmholtz_energy<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.contributions()
            .iter()
            .map(|c| c.helmholtz_energy(state))
            .sum()
    }

    /// Evaluate the reduced Helmholtz energy of each individual contribution
    /// and return them together with a string representation of the contribution.
    fn evaluate_residual_contributions<D: DualNum<f64>>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)>
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.contributions()
            .iter()
            .map(|c| (c.to_string(), c.helmholtz_energy(state)))
            .collect()
    }

    /// Check if the provided optional mole number is consistent with the
    /// equation of state.
    ///
    /// In general, the number of elements in `moles` needs to match the number
    /// of components of the equation of state. For a pure component, however,
    /// no moles need to be provided. In that case, it is set to the constant
    /// reference value.
    fn validate_moles<U: EosUnit>(
        &self,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityArray1<U>> {
        let l = moles.map_or(1, |m| m.len());
        if self.components() == l {
            match moles {
                Some(m) => Ok(m.to_owned()),
                None => Ok(Array::ones(1) * U::reference_moles()),
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
    fn max_density<U: EosUnit>(
        &self,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self
            .validate_moles(moles)?
            .to_reduced(U::reference_moles())?;
        Ok(self.compute_max_density(&mr) * U::reference_density())
    }

    /// Calculate the second virial coefficient $B(T)$
    fn second_virial_coefficient<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let mut rho = HyperDual64::zero();
        rho.eps1[0] = 1.0;
        rho.eps2[0] = 1.0;
        let t = HyperDual64::from(temperature.to_reduced(U::reference_temperature())?);
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.helmholtz_energy(&s).eps1eps2[(0, 0)] * 0.5 / U::reference_density())
    }

    /// Calculate the third virial coefficient $C(T)$
    fn third_virial_coefficient<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let rho = Dual3_64::zero().derive();
        let t = Dual3_64::from(temperature.to_reduced(U::reference_temperature())?);
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.helmholtz_energy(&s).v3 / 3.0 / U::reference_density().powi(2))
    }

    /// Calculate the temperature derivative of the second virial coefficient $B'(T)$
    fn second_virial_coefficient_temperature_derivative<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let mut rho = HyperDual::zero();
        rho.eps1[0] = Dual64::one();
        rho.eps2[0] = Dual64::one();
        let t = HyperDual::from_re(
            Dual64::from(temperature.to_reduced(U::reference_temperature())?).derive(),
        );
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.helmholtz_energy(&s).eps1eps2[(0, 0)].eps[0] * 0.5
            / (U::reference_density() * U::reference_temperature()))
    }

    /// Calculate the temperature derivative of the third virial coefficient $C'(T)$
    fn third_virial_coefficient_temperature_derivative<U: EosUnit>(
        &self,
        temperature: QuantityScalar<U>,
        moles: Option<&QuantityArray1<U>>,
    ) -> EosResult<QuantityScalar<U>> {
        let mr = self.validate_moles(moles)?;
        let x = mr.to_reduced(mr.sum())?;
        let rho = Dual3::zero().derive();
        let t = Dual3::from_re(
            Dual64::from(temperature.to_reduced(U::reference_temperature())?).derive(),
        );
        let s = StateHD::new_virial(t, rho, x);
        Ok(self.helmholtz_energy(&s).v3.eps[0]
            / 3.0
            / (U::reference_density().powi(2) * U::reference_temperature()))
    }
}
