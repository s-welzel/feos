use crate::StateHD;
use ndarray::Array1;
use num_dual::DualNum;
use std::fmt;

use super::debroglie::{DeBroglieWavelength, DeBroglieWavelengthDual};

/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGas: Sync + Send + fmt::Display {
    /// Return the number of components
    fn components(&self) -> usize;

    /// Return an equation of state consisting of the components
    /// contained in component_list.
    fn subset(&self, component_list: &[usize]) -> Self;

    fn de_broglie_wavelength(&self) -> &Box<dyn DeBroglieWavelength>;

    /// Evaluate the ideal gas contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn helmholtz_energy<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn DeBroglieWavelength: DeBroglieWavelengthDual<D>,
    {
        let lambda = self.de_broglie_wavelength().evaluate(state.temperature);
        ((lambda
            + state.partial_density.mapv(|x| {
                if x.re() == 0.0 {
                    D::from(0.0)
                } else {
                    x.ln() - 1.0
                }
            }))
            * &state.moles)
            .sum()
    }
}

#[derive(Debug)]
pub struct DefaultDeBroglie(pub usize);

impl<D: DualNum<f64>> DeBroglieWavelengthDual<D> for DefaultDeBroglie {
    fn evaluate(&self, _: D) -> Array1<D> {
        Array1::zeros(self.0)
    }
}

impl fmt::Display for DefaultDeBroglie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeBroglie set to one.")
    }
}

pub struct DefaultIdealGas {
    components: usize,
    de_broglie: Box<dyn DeBroglieWavelength>,
}

impl DefaultIdealGas {
    pub fn new(components: usize) -> Self {
        Self {
            components,
            de_broglie: Box::new(DefaultDeBroglie(components)),
        }
    }
}

impl IdealGas for DefaultIdealGas {
    fn components(&self) -> usize {
        self.components
    }
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::new(component_list.len())
    }
    fn de_broglie_wavelength(&self) -> &Box<dyn DeBroglieWavelength> {
        &self.de_broglie
    }
}

impl fmt::Display for DefaultIdealGas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (default) with constant deBroglie wavelength.")
    }
}
