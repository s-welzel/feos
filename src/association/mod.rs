//! Generic implementation of the SAFT association contribution
//! that can be used across models.
use crate::hard_sphere::HardSphereProperties;
use feos_core::{EosError, HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::linalg::{norm, LU};
use num_dual::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::SubAssign;
use std::sync::Arc;

#[cfg(feature = "dft")]
mod dft;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub use python::PyAssociationRecord;

/// Pure component association parameters.
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub struct AssociationRecord {
    /// Association volume parameter
    pub kappa_ab: f64,
    /// Association energy parameter in units of Kelvin
    pub epsilon_k_ab: f64,
    /// \# of association sites of type A
    #[serde(skip_serializing_if = "Option::is_none")]
    pub na: Option<f64>,
    /// \# of association sites of type B
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nb: Option<f64>,
}

impl AssociationRecord {
    pub fn new(kappa_ab: f64, epsilon_k_ab: f64, na: Option<f64>, nb: Option<f64>) -> Self {
        Self {
            kappa_ab,
            epsilon_k_ab,
            na,
            nb,
        }
    }
}

impl fmt::Display for AssociationRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssociationRecord(kappa_ab={}", self.kappa_ab)?;
        write!(f, ", epsilon_k_ab={}", self.epsilon_k_ab)?;
        write!(f, ", na={}", self.na.unwrap_or(1.0))?;
        write!(f, ", nb={})", self.nb.unwrap_or(1.0))
    }
}

/// Parameter set required for the SAFT association Helmoltz energy
/// contribution and functional.
#[derive(Clone)]
pub struct AssociationParameters {
    component_index: Array1<usize>,
    pub assoc_comp: Array1<usize>,
    pub kappa_ab: Array1<f64>,
    pub epsilon_k_ab: Array1<f64>,
    pub sigma3_kappa_aibj: Array2<f64>,
    pub epsilon_k_aibj: Array2<f64>,
    pub na: Array1<f64>,
    pub nb: Array1<f64>,
}

impl AssociationParameters {
    pub fn new(
        records: &[Option<AssociationRecord>],
        sigma: &Array1<f64>,
        component_index: Option<&Array1<usize>>,
    ) -> Self {
        let mut assoc_comp = Vec::new();
        let mut sigma_assoc = Vec::new();
        let mut kappa_ab = Vec::new();
        let mut epsilon_k_ab = Vec::new();
        let mut na = Vec::new();
        let mut nb = Vec::new();

        for (i, record) in records.iter().enumerate() {
            if let Some(record) = record.as_ref() {
                if record.kappa_ab > 0.0 && record.epsilon_k_ab > 0.0 {
                    assoc_comp.push(i);
                    sigma_assoc.push(sigma[i]);
                    kappa_ab.push(record.kappa_ab);
                    epsilon_k_ab.push(record.epsilon_k_ab);
                    na.push(record.na.unwrap_or(1.0));
                    nb.push(record.nb.unwrap_or(1.0));
                }
            }
        }

        let sigma3_kappa_aibj = Array2::from_shape_fn([kappa_ab.len(); 2], |(i, j)| {
            (sigma_assoc[i] * sigma_assoc[j]).powf(1.5) * (kappa_ab[i] * kappa_ab[j]).sqrt()
        });
        let epsilon_k_aibj = Array2::from_shape_fn([epsilon_k_ab.len(); 2], |(i, j)| {
            0.5 * (epsilon_k_ab[i] + epsilon_k_ab[j])
        });

        Self {
            component_index: component_index
                .cloned()
                .unwrap_or_else(|| Array1::from_shape_fn(records.len(), |i| i)),
            assoc_comp: Array1::from_vec(assoc_comp),
            kappa_ab: Array1::from_vec(kappa_ab),
            epsilon_k_ab: Array1::from_vec(epsilon_k_ab),
            sigma3_kappa_aibj,
            epsilon_k_aibj,
            na: Array1::from_vec(na),
            nb: Array1::from_vec(nb),
        }
    }
}

/// Implementation of the SAFT association Helmholtz energy
/// contribution and functional.
pub struct Association<P> {
    parameters: Arc<P>,
    association_parameters: AssociationParameters,
    max_iter: usize,
    tol: f64,
    force_cross_association: bool,
}

impl<P: HardSphereProperties> Association<P> {
    pub fn new(
        parameters: &Arc<P>,
        association_parameters: &AssociationParameters,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            parameters: parameters.clone(),
            association_parameters: association_parameters.clone(),
            max_iter,
            tol,
            force_cross_association: false,
        }
    }

    pub fn new_cross_association(
        parameters: &Arc<P>,
        association_parameters: &AssociationParameters,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        let mut res = Self::new(parameters, association_parameters, max_iter, tol);
        res.force_cross_association = true;
        res
    }

    fn association_strength<D: DualNum<f64>>(
        &self,
        temperature: D,
        diameter: &Array1<D>,
        n2: D,
        n3i: D,
        xi: D,
    ) -> Array2<D> {
        // Calculate association strength
        let ac = &self.association_parameters.assoc_comp;
        Array2::from_shape_fn([ac.len(); 2], |(i, j)| {
            let k = diameter[ac[i]] * diameter[ac[j]] / (diameter[ac[i]] + diameter[ac[j]])
                * (n2 * n3i);
            n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
                * self.association_parameters.sigma3_kappa_aibj[(i, j)]
                * (temperature.recip() * self.association_parameters.epsilon_k_aibj[(i, j)])
                    .exp_m1()
        })
    }
}

impl<D: DualNum<f64> + ScalarOperand, P: HardSphereProperties> HelmholtzEnergyDual<D>
    for Association<P>
{
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p: &P = &self.parameters;

        // temperature dependent segment diameter
        let diameter = p.hs_diameter(state.temperature);

        // auxiliary variables
        let [zeta2, n3] = p.zeta(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        if self.association_parameters.assoc_comp.len() > 1 || self.force_cross_association {
            // extract densities of associating segments
            let rho_assoc = self
                .association_parameters
                .assoc_comp
                .mapv(|a| state.partial_density[self.association_parameters.component_index[a]]);

            // Helmholtz energy
            self.helmholtz_energy_density_cross_association(
                state.temperature,
                &rho_assoc,
                &diameter,
                n2,
                n3i,
                D::one(),
                self.max_iter,
                self.tol,
                None,
            )
            .unwrap_or_else(|_| D::from(std::f64::NAN))
                * state.volume
        } else {
            // association strength
            let c = self.association_parameters.component_index
                [self.association_parameters.assoc_comp[0]];
            let deltarho =
                self.association_strength(state.temperature, &diameter, n2, n3i, D::one())[(0, 0)]
                    * state.partial_density[c];

            let na = self.association_parameters.na[0];
            let nb = self.association_parameters.nb[0];
            if nb > 0.0 {
                // no cross association, two association sites
                let xa = Self::assoc_site_frac_ab(deltarho, na, nb);
                let xb = (xa - 1.0) * (na / nb) + 1.0;

                state.moles[c] * ((xa.ln() - xa * 0.5 + 0.5) * na + (xb.ln() - xb * 0.5 + 0.5) * nb)
            } else {
                // no cross association, one association site
                let xa = Self::assoc_site_frac_a(deltarho, na);

                state.moles[c] * (xa.ln() - xa * 0.5 + 0.5) * na
            }
        }
    }
}

impl<P> fmt::Display for Association<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Association")
    }
}

impl<P: HardSphereProperties> Association<P> {
    pub fn assoc_site_frac_ab<D: DualNum<f64>>(deltarho: D, na: f64, nb: f64) -> D {
        (((deltarho * (na - nb) + 1.0).powi(2) + deltarho * nb * 4.0).sqrt()
            + (deltarho * (nb - na) + 1.0))
            .recip()
            * 2.0
    }

    pub fn assoc_site_frac_a<D: DualNum<f64>>(deltarho: D, na: f64) -> D {
        ((deltarho * 4.0 * na + 1.0).sqrt() + 1.0).recip() * 2.0
    }

    #[allow(clippy::too_many_arguments)]
    fn helmholtz_energy_density_cross_association<
        S: Data<Elem = D>,
        D: DualNum<f64> + ScalarOperand,
    >(
        &self,
        temperature: D,
        density: &ArrayBase<S, Ix1>,
        diameter: &Array1<D>,
        n2: D,
        n3i: D,
        xi: D,
        max_iter: usize,
        tol: f64,
        x0: Option<&mut Array1<f64>>,
    ) -> Result<D, EosError> {
        // check if density is close to 0
        if density.sum().re() < f64::EPSILON {
            if let Some(x0) = x0 {
                x0.fill(1.0);
            }
            return Ok(D::zero());
        }

        let assoc_comp = &self.association_parameters.assoc_comp;
        let nassoc = assoc_comp.len();

        // association strength
        let delta = self.association_strength(temperature, diameter, n2, n3i, xi);

        // extract parameters of associating components
        let na = &self.association_parameters.na;
        let nb = &self.association_parameters.nb;

        // cross-association according to Michelsen2006
        // initialize monomer fraction
        let mut x = match &x0 {
            Some(x0) => (*x0).clone(),
            None => Array::from_elem(2 * nassoc, 0.2),
        };

        for k in 0..max_iter {
            if Self::newton_step_cross_association::<_, f64>(
                nassoc,
                &mut x,
                &delta.map(D::re),
                na,
                nb,
                &density.map(D::re),
                tol,
            )? {
                break;
            }
            if k == max_iter - 1 {
                return Err(EosError::NotConverged("Cross association".into()));
            }
        }

        // calculate derivatives
        let mut x_dual = x.mapv(D::from);
        for _ in 0..D::NDERIV {
            Self::newton_step_cross_association(nassoc, &mut x_dual, &delta, na, nb, density, tol)?;
        }

        // save monomer fraction
        if let Some(x0) = x0 {
            *x0 = x;
        }

        // Helmholtz energy density
        let xa = x_dual.slice(s![..nassoc]);
        let xb = x_dual.slice(s![nassoc..]);
        let f = |x: D| x.ln() - x * 0.5 + 0.5;
        Ok((density * (xa.mapv(f) * na + xb.mapv(f) * nb)).sum())
    }

    fn newton_step_cross_association<S: Data<Elem = D>, D: DualNum<f64> + ScalarOperand>(
        nassoc: usize,
        x: &mut Array1<D>,
        delta: &Array2<D>,
        na: &Array1<f64>,
        nb: &Array1<f64>,
        rho: &ArrayBase<S, Ix1>,
        tol: f64,
    ) -> Result<bool, EosError> {
        // gradient
        let mut g = x.map(D::recip);
        // Hessian
        let mut h: Array2<D> = Array::zeros((2 * nassoc, 2 * nassoc));

        // split x array
        let (xa, xb) = x.view().split_at(Axis(0), nassoc);

        // calculate gradients and approximate Hessian
        for i in 0..nassoc {
            let d = &delta.index_axis(Axis(0), i) * rho;

            let dnx = (&xb * nb * &d).sum() + 1.0;
            g[i] -= dnx;
            for j in 0..nassoc {
                h[(i, nassoc + j)] = -d[j] * nb[j];
                h[(nassoc + i, j)] = -d[j] * na[j];
            }
            h[(i, i)] = -dnx / xa[i];

            let dnx = (&xa * na * &d).sum() + 1.0;
            g[nassoc + i] -= dnx;
            h[(nassoc + i, nassoc + i)] = -dnx / xb[i];
        }

        // Newton step
        x.sub_assign(&LU::new(h)?.solve(&g));

        // check convergence
        Ok(norm(&g.map(D::re)) < tol)
    }
}

#[cfg(test)]
#[cfg(feature = "pcsaft")]
mod tests_pcsaft {
    use super::*;
    use crate::pcsaft::parameters::utils::water_parameters;
    use crate::pcsaft::PcSaftParameters;
    use approx::assert_relative_eq;
    use feos_core::parameter::Parameter;

    #[test]
    fn helmholtz_energy() {
        let params = Arc::new(water_parameters());
        let assoc = Association::new(&params, &params.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross() {
        let params = Arc::new(water_parameters());
        let assoc = Association::new_cross_association(&params, &params.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross_3b() {
        let mut params = water_parameters();
        let mut record = params.pure_records.pop().unwrap();
        let mut association_record = record.model_record.association_record.unwrap();
        association_record.na = Some(2.0);
        record.model_record.association_record = Some(association_record);
        let params = Arc::new(PcSaftParameters::new_pure(record));
        let assoc = Association::new(&params, &params.association, 50, 1e-10);
        let cross_assoc =
            Association::new_cross_association(&params, &params.association, 50, 1e-10);
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_assoc = assoc.helmholtz_energy(&s) / n;
        let a_cross_assoc = cross_assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_assoc, a_cross_assoc, epsilon = 1e-10);
    }
}

#[cfg(test)]
#[cfg(feature = "gc_pcsaft")]
mod tests_gc_pcsaft {
    use super::*;
    use crate::gc_pcsaft::eos::parameter::test::*;
    use approx::assert_relative_eq;
    use feos_core::EosUnit;
    use ndarray::arr1;
    use num_dual::Dual64;
    use quantity::si::{METER, MOL, PASCAL};

    #[test]
    fn test_assoc_propanol() {
        let params = Arc::new(propanol());
        let contrib = Association::new(&params, &params.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_cross_assoc_propanol() {
        let params = Arc::new(propanol());
        let contrib = Association::new_cross_association(&params, &params.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (1.5 * MOL).to_reduced(EosUnit::reference_moles()).unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            arr1(&[Dual64::from_re(moles)]),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, -3.6819598891967344 * PASCAL, max_relative = 1e-10);
    }

    #[test]
    fn test_cross_assoc_ethanol_propanol() {
        let params = Arc::new(ethanol_propanol(false));
        let contrib = Association::new(&params, &params.association, 50, 1e-10);
        let temperature = 300.0;
        let volume = METER
            .powi(3)
            .to_reduced(EosUnit::reference_volume())
            .unwrap();
        let moles = (arr1(&[1.5, 2.5]) * MOL)
            .to_reduced(EosUnit::reference_moles())
            .unwrap();
        let state = StateHD::new(
            Dual64::from_re(temperature),
            Dual64::from_re(volume).derive(),
            moles.mapv(Dual64::from_re),
        );
        let pressure =
            -contrib.helmholtz_energy(&state).eps[0] * temperature * EosUnit::reference_pressure();
        assert_relative_eq!(pressure, -26.105606376765632 * PASCAL, max_relative = 1e-10);
    }
}
