use super::{PhaseEquilibrium, SolverOptions};
use crate::equation_of_state::{EquationOfState, IdealGas, Residual};
use crate::errors::EosResult;
use crate::state::{State, StateVec};
#[cfg(feature = "rayon")]
use crate::EosUnit;
#[cfg(feature = "rayon")]
use ndarray::{Array1, ArrayView1, Axis};
#[cfg(feature = "rayon")]
use quantity::si::SIUnit;
use quantity::si::{SIArray1, SINumber};
#[cfg(feature = "rayon")]
use rayon::{prelude::*, ThreadPool};
use std::sync::Arc;

/// Pure component and binary mixture phase diagrams.
pub struct PhaseDiagram<I: IdealGas, R: Residual, const N: usize> {
    pub states: Vec<PhaseEquilibrium<I, R, N>>,
}

impl<I: IdealGas, R: Residual, const N: usize> Clone for PhaseDiagram<I, R, N> {
    fn clone(&self) -> Self {
        Self {
            states: self.states.clone(),
        }
    }
}

impl<I: IdealGas, R: Residual, const N: usize> PhaseDiagram<I, R, N> {
    /// Create a phase diagram from a list of phase equilibria.
    pub fn new(states: Vec<PhaseEquilibrium<I, R, N>>) -> Self {
        Self { states }
    }
}

impl<I: IdealGas, R: Residual> PhaseDiagram<I, R, 2> {
    /// Calculate a phase diagram for a pure component.
    pub fn pure(
        eos: &Arc<EquationOfState<I, R>>,
        min_temperature: SINumber,
        npoints: usize,
        critical_temperature: Option<SINumber>,
        options: SolverOptions,
    ) -> EosResult<Self> {
        let mut states = Vec::with_capacity(npoints);

        let sc = State::critical_point(eos, None, critical_temperature, SolverOptions::default())?;

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = SIArray1::linspace(min_temperature, max_temperature, npoints - 1)?;

        let mut vle = None;
        for ti in &temperatures {
            vle = PhaseEquilibrium::pure(eos, ti, vle.as_ref(), options).ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));

        Ok(PhaseDiagram::new(states))
    }

    /// Return the vapor states of the diagram.
    pub fn vapor(&self) -> StateVec<'_, I, R> {
        self.states.iter().map(|s| s.vapor()).collect()
    }

    /// Return the liquid states of the diagram.
    pub fn liquid(&self) -> StateVec<'_, I, R> {
        self.states.iter().map(|s| s.liquid()).collect()
    }
}

#[cfg(feature = "rayon")]
impl<I: IdealGas, R: Residual> PhaseDiagram<E, 2> {
    fn solve_temperatures(
        eos: &Arc<EquationOfState<I, R>>,
        temperatures: ArrayView1<f64>,
        options: SolverOptions,
    ) -> EosResult<Vec<PhaseEquilibrium<I, R, 2>>> {
        let mut states = Vec::with_capacity(temperatures.len());
        let mut vle = None;
        for ti in temperatures {
            vle = PhaseEquilibrium::pure(
                eos,
                *ti * SIUnit::reference_temperature(),
                vle.as_ref(),
                options,
            )
            .ok();
            if let Some(vle) = vle.as_ref() {
                states.push(vle.clone());
            }
        }
        Ok(states)
    }

    pub fn par_pure(
        eos: &Arc<EquationOfState<I, R>>,
        min_temperature: SINumber,
        npoints: usize,
        chunksize: usize,
        thread_pool: ThreadPool,
        critical_temperature: Option<SINumber>,
        options: SolverOptions,
    ) -> EosResult<Self> {
        let sc = State::critical_point(eos, None, critical_temperature, SolverOptions::default())?;

        let max_temperature = min_temperature
            + (sc.temperature - min_temperature) * ((npoints - 2) as f64 / (npoints - 1) as f64);
        let temperatures = Array1::linspace(
            min_temperature.to_reduced(SIUnit::reference_temperature())?,
            max_temperature.to_reduced(SIUnit::reference_temperature())?,
            npoints - 1,
        );

        let mut states: Vec<PhaseEquilibrium<I, R, 2>> = thread_pool.install(|| {
            temperatures
                .axis_chunks_iter(Axis(0), chunksize)
                .into_par_iter()
                .filter_map(|t| Self::solve_temperatures(eos, t, options).ok())
                .flatten()
                .collect()
        });

        states.push(PhaseEquilibrium::from_states(sc.clone(), sc));
        Ok(PhaseDiagram::new(states))
    }
}
