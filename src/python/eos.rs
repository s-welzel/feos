use crate::eos::{ResidualModel, IdealGasModel};
#[cfg(feature = "estimator")]
use crate::estimator::*;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::PyGcPcSaftEosParameters;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
#[cfg(feature = "estimator")]
use crate::impl_estimator;
#[cfg(all(feature = "estimator", feature = "pcsaft"))]
use crate::impl_estimator_entropy_scaling;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::PyPcSaftParameters;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{DQVariants, PcSaft, PcSaftOptions};
#[cfg(feature = "pets")]
use crate::pets::python::PyPetsParameters;
#[cfg(feature = "pets")]
use crate::pets::{Pets, PetsOptions};
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::python::PySaftVRQMieParameters;
#[cfg(feature = "saftvrqmie")]
use crate::saftvrqmie::{FeynmanHibbsOrder, SaftVRQMie, SaftVRQMieOptions};
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::PyUVParameters;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::{Perturbation, UVTheory, UVTheoryOptions, VirialOrder};

use feos_core::cubic::PengRobinson;
use feos_core::equation_of_state::{EquationOfState, Residual, DefaultIdealGas, MolarWeight};
use feos_core::joback::Joback;
use feos_core::python::cubic::PyPengRobinsonParameters;
use feos_core::python::joback::PyJobackRecord;
use feos_core::python::user_defined::{PyResidual, PyIdealGas};
use feos_core::*;
use numpy::convert::ToPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::python::{PySINumber, PySIArray1, PySIArray2};
use quantity::si::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "Residual")]
pub struct PyResidualModel(RefCell<Option<ResidualModel>>);

#[pymethods]
impl PyResidualModel {
    /// PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PcSaftParameters
    ///     The parameters of the PC-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// dq_variant : DQVariants, optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to 'DQVariants.DQ35'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "pcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant=DQVariants::DQ35),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant)"
    )]
    pub fn pcsaft(
        parameters: PyPcSaftParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: DQVariants,
    ) -> Self {
        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let pcsaft = PcSaft::with_options(
            parameters.0,
            options,
        );
        Self(RefCell::new(Some(ResidualModel::PcSaft(pcsaft))))
    }
}

/// Collection of equations of state.
#[pyclass(name = "EquationOfState")]
#[derive(Clone)]
pub struct PyEquationOfState(pub Arc<EquationOfState<IdealGasModel, ResidualModel>>);

#[pymethods]
impl PyEquationOfState {

    #[new]
    pub fn new(residual: &PyResidualModel) -> Self {
        let res = residual.0.take().unwrap();
        let components = res.components();
        Self(Arc::new(EquationOfState::new(IdealGasModel::DefaultIdealGas(DefaultIdealGas::new(components)), res)))
    }

    /// PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PcSaftParameters
    ///     The parameters of the PC-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    /// dq_variant : DQVariants, optional
    ///     Combination rule used in the dipole/quadrupole term. Defaults to 'DQVariants.DQ35'
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "pcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant=DQVariants::DQ35),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10, dq_variant)"
    )]
    pub fn pcsaft(
        parameters: PyPcSaftParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
        dq_variant: DQVariants,
    ) -> Self {
        use feos_core::equation_of_state::DefaultIdealGas;

        let options = PcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
            dq_variant,
        };
        let pcsaft = PcSaft::with_options(
            parameters.0,
            options,
        );
        let components = pcsaft.components();
        Self(Arc::new(EquationOfState::new(IdealGasModel::DefaultIdealGas(DefaultIdealGas::new(components)), ResidualModel::PcSaft(pcsaft))))
    }

    /// (heterosegmented) group contribution PC-SAFT equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : GcPcSaftEosParameters
    ///     The parameters of the PC-SAFT equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The gc-PC-SAFT equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "gc_pcsaft")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = "(parameters, max_eta=0.5, max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"
    )]
    pub fn gc_pcsaft(
        parameters: PyGcPcSaftEosParameters,
        max_eta: f64,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> Self {
        let options = GcPcSaftOptions {
            max_eta,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        Self(Arc::new(ResidualModel::GcPcSaft(GcPcSaft::with_options(
            parameters.0,
            options,
        ))))
    }

    /// Peng-Robinson equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PengRobinsonParameters
    ///     The parameters of the PR equation of state to use.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PR equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    pub fn peng_robinson(parameters: PyPengRobinsonParameters) -> Self {
        let residual = PengRobinson::new(
            parameters.0,
        );
        let components = residual.components();
        Self(Arc::new(EquationOfState::new(IdealGasModel::DefaultIdealGas(DefaultIdealGas::new(components)), ResidualModel::PengRobinson(residual))))
    }

    /// Equation of state from a Python class.
    ///
    /// Parameters
    /// ----------
    /// residual : Class
    ///     A python class implementing the necessary methods
    ///     to be used as a residual model.
    /// ideal_gas : Class, optional
    ///     A python class implementing the necessary methods
    ///     to be used as a ideal gas model.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    #[staticmethod]
    fn python(residual: Py<PyAny>, ideal_gas: Option<Py<PyAny>>) -> PyResult<Self> {
        let residual = ResidualModel::Python(PyResidual::new(residual)?);
        let components = residual.components();
        let ideal_gas = if let Some(ig) = ideal_gas {
            IdealGasModel::Python(PyIdealGas::new(ig)?)
        } else {
            IdealGasModel::DefaultIdealGas(DefaultIdealGas::new(components))
        };
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }

    /// PeTS equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : PetsParameters
    ///     The parameters of the PeTS equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The PeTS equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "pets")]
    #[staticmethod]
    #[pyo3(signature = (parameters, max_eta=0.5), text_signature = "(parameters, max_eta=0.5)")]
    fn pets(parameters: PyPetsParameters, max_eta: f64) -> Self {
        let options = PetsOptions { max_eta };
        Self(Arc::new(ResidualModel::Pets(Pets::with_options(
            parameters.0,
            options,
        ))))
    }

    /// UV-Theory equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : UVParameters
    ///     The parameters of the UV-theory equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// perturbation : Perturbation, optional
    ///     Division type of the Mie potential. Defaults to WCA division.
    /// virial_order : VirialOrder, optional
    ///     Highest order of virial coefficient to consider.
    ///     Defaults to second order (original uv-theory).
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The UV-Theory equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "uvtheory")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, perturbation=Perturbation::WeeksChandlerAndersen, virial_order=VirialOrder::Second),    
        text_signature = "(parameters, max_eta=0.5, perturbation, virial_order)"
    )]
    fn uvtheory(
        parameters: PyUVParameters,
        max_eta: f64,
        perturbation: Perturbation,
        virial_order: VirialOrder,
    ) -> PyResult<Self> {
        let options = UVTheoryOptions {
            max_eta,
            perturbation,
            virial_order,
        };
        Ok(Self(Arc::new(ResidualModel::UVTheory(
            UVTheory::with_options(parameters.0, options)?,
        ))))
    }

    /// SAFT-VRQ Mie equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : SaftVRQMieParameters
    ///     The parameters of the SAFT-VRQ Mie equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// fh_order : FeynmanHibbsOrder, optional
    ///     Which Feyman-Hibbs correction order to use. Defaults to FeynmanHibbsOrder.FH1.
    ///     Currently, only the first order is implemented.
    /// inc_nonadd_term : bool, optional
    ///     Include non-additive correction to the hard-sphere reference. Defaults to True.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The SAFT-VRQ Mie equation of state that can be used to compute thermodynamic
    ///     states.
    #[cfg(feature = "saftvrqmie")]
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, fh_order=FeynmanHibbsOrder::FH1, inc_nonadd_term=true),
        text_signature = "(parameters, max_eta=0.5, fh_order, inc_nonadd_term=True)"
    )]
    fn saftvrqmie(
        parameters: PySaftVRQMieParameters,
        max_eta: f64,
        fh_order: FeynmanHibbsOrder,
        inc_nonadd_term: bool,
    ) -> Self {
        let options = SaftVRQMieOptions {
            max_eta,
            fh_order,
            inc_nonadd_term,
        };
        Self(Arc::new(ResidualModel::SaftVRQMie(SaftVRQMie::with_options(
            parameters.0,
            options,
        ))))
    }

    // fn joback(&self, parameters: Vec<PyJobackRecord>) -> Self {
    //     let records = Arc::new(parameters.iter().map(|p| p.0.clone()).collect());
    //     let ideal_gas = IdealGasModel::Joback(Joback::new(records));
    //     Self(Arc::new(EquationOfState::new(ideal_gas, self.0.residual)))
    // }
}

impl_equation_of_state!(PyEquationOfState);
impl_virial_coefficients!(PyEquationOfState);
impl_state!(IdealGasModel, ResidualModel, PyEquationOfState);
impl_state_molarweight!();
// #[cfg(feature = "pcsaft")]
// impl_state_entropy_scaling!(ResidualModel, PyResidualModel);
impl_phase_equilibrium!(IdealGasModel, ResidualModel, PyEquationOfState);

#[cfg(feature = "estimator")]
impl_estimator!(ResidualModel, PyResidualModel);
#[cfg(all(feature = "estimator", feature = "pcsaft"))]
impl_estimator_entropy_scaling!(ResidualModel, PyResidualModel);

#[pymodule]
pub fn eos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Contributions>()?;
    m.add_class::<Verbosity>()?;

    m.add_class::<PyEquationOfState>()?;
    m.add_class::<PyResidualModel>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyStateVec>()?;
    m.add_class::<PyPhaseDiagram>()?;
    m.add_class::<PyPhaseEquilibrium>()?;

    #[cfg(feature = "estimator")]
    m.add_wrapped(wrap_pymodule!(estimator_eos))?;

    Ok(())
}

#[cfg(feature = "estimator")]
#[pymodule]
pub fn estimator_eos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataSet>()?;
    m.add_class::<PyEstimator>()?;
    m.add_class::<PyLoss>()?;
    m.add_class::<Phase>()
}
