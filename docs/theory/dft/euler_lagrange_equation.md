# Euler-Lagrange equation
The fundamental expression in classical density functional theory is the relation between the grand potential functional $\Omega$ and the intrinsic Helmholtz energy functional $F$.

$$\Omega(T,\mu,[\rho(r)])=F(T,[\rho(r)])-\sum_i\int\rho_i(r)\left(\mu_i-V_i^\mathrm{ext}(r)\right)\mathrm{d}r$$

What makes this expression so appealing is that the intrinsic Helmholtz energy functional only depends on the temperature $T$ and the density profiles $\rho_i(r)$ of the system and not on the external potentials $V_i^\mathrm{ext}(r)$.

For a given temperature $T$, chemical potentials $\mu_i$ and external potentials $V_i^\mathrm{ext}(r)$ the grand potential reaches a minimum at equilibrium. Mathematically this condition can be written as

$$\left.\frac{\delta\Omega}{\delta\rho_i(r)}\right|_{T,\mu}=F_{\rho_i}(r)-\mu_i+V_i^{\mathrm{ext}}(r)=0$$ (eqn:euler_lagrange_mu)

where $F_{\rho_i}(r)=\left.\frac{\delta F}{\delta\rho_i(r)}\right|_T$ is short for the functional derivative of the intrinsic Helmholtz energy. In this context, eq. (1) is commonly referred to as the Euler-Lagrange equation, an implicit nonlinear integral equation which needs to be solved for the equilibrium density profiles of the system.

For a homogeneous (bulk) system, $V_i^\mathrm{ext}=0$ and we get

$$F_{\rho_i}^\mathrm{b}-\mu_i=0$$ (eqn:euler_lagrange_bulk)

The functional derivative of the Helmholtz energy of a bulk system $F_{\rho_i}^\mathrm{b}$ is a function of the temperature $T$ and bulk densities $\rho^\mathrm{b}$. At this point, it can be advantageous to relate the grand potential of an inhomogeneous system to the densities of a bulk system that is in equilibrium with the inhomogeneous system. This approach has several advantages:
- The thermal de Broglie wavelength $\Lambda$ cancels out.
- If the chemical potential of the system is not known, all variables are the same quantity (densities).
- The bulk system is described by a Helmholtz energy which is explicit in the density, so there are no internal iterations required.

Using eq. {eq}`eqn:euler_lagrange_bulk` in eq. {eq}`eqn:euler_lagrange_mu` leads to the Euler-Lagrange equation

$$\left.\frac{\delta\Omega}{\delta\rho_i(r)}\right|_{T,\rho^\mathrm{b}}=F_{\rho_i}(r)-F_{\rho_i}^\mathrm{b}+V_i^{\mathrm{ext}}(r)=0$$ (eqn:euler_lagrange_rho)

## Spherical molecules
In the simplest case, the molecules under consideration can be described as spherical. Then the Helmholtz energy can be split into an ideal and a residual part:

$$\beta F=\sum_i\int\rho_i(r)\left(\ln\left(\rho_i(r)\Lambda_i^3\right)-1\right)\mathrm{d}r+\beta F^\mathrm{res}$$

with the thermal de Broglie wavelength $\Lambda_i$. The functional derivatives for an inhomogeneous and a bulk system follow as

$$\beta F_{\rho_i}(r)=\ln\left(\rho_i(r)\Lambda_i^3\right)+\beta F_{\rho_i}^\mathrm{res}$$

$$\beta F_{\rho_i}^\mathrm{b}=\ln\left(\rho_i^\mathrm{b}\Lambda_i^3\right)+\beta F_{\rho_i}^\mathrm{b,res}$$

Using these expressions in eq. {eq}`eqn:euler_lagrange_rho` results in

$$\left.\frac{\delta\beta\Omega}{\delta\rho_i(r)}\right|_{T,\rho^\mathrm{b}}=\ln\left(\frac{\rho_i(r)}{\rho_i^\mathrm{b}}\right)+\beta\left(F_{\rho_i}^\mathrm{res}(r)-F_{\rho_i}^\mathrm{b,res}+V_i^{\mathrm{ext}}(r)\right)=0$$

The Euler-Lagrange equation can be recast as

$$\rho_i(r)=\rho_i^\mathrm{b}e^{\beta\left(F_{\rho_i}^\mathrm{b,res}-F_{\rho_i}^\mathrm{res}(r)-V_i^\mathrm{ext}(r)\right)}$$

which is convenient because it leads directly to a recurrence relation known as Picard iteration.

## Homosegmented chains
For chain molecules that do not resolve individual segments (essentially the PC-SAFT Helmholtz energy functional) a chain contribution is introduced as

$$\beta F^\mathrm{chain}=-\sum_i\int\rho_i(r)\left(m_i-1\right)\ln\left(\frac{y_{ii}\lambda_i(r)}{\rho_i(r)}\right)\mathrm{d}r$$

Where $m_i$ is the number of segments (i.e., the PC-SAFT chain length parameter), $y_{ii}$ is the cavity correlation function at contact in the reference fluid, and $\lambda_i$ is a weighted density.
The presence of $\rho_i(r)$ in the logarithm poses numerical problems. Therefore, it is convenient to rearrange the expression as

$$\begin{align}
\beta F^\mathrm{chain}=&\sum_i\int\rho_i(r)\left(m_i-1\right)\left(\ln\left(\rho_i(r)\Lambda_i^3\right)-1\right)\mathrm{d}r\\
&\underbrace{-\sum_i\int\rho_i(r)\left(m_i-1\right)\left(\ln\left(y_{ii}\lambda_i(r)\Lambda_i^3\right)-1\right)\mathrm{d}r}_{\beta\hat{F}^\mathrm{chain}}
\end{align}$$

Then the total Helmholtz energy

$$\beta F=\sum_i\int\rho_i(r)\left(\ln\left(\rho_i(r)\Lambda_i^3\right)-1\right)\mathrm{d}r+\beta F^\mathrm{chain}+\beta F^\mathrm{res}$$

can be rearranged to

$$\beta F=\sum_i\int\rho_i(r)m_i\left(\ln\left(\rho_i(r)\Lambda_i^3\right)-1\right)\mathrm{d}r+\underbrace{\beta\hat{F}^\mathrm{chain}+\beta F^\mathrm{res}}_{\beta\hat{F}^\mathrm{res}}$$

The functional derivatives are then similar to the spherical case

$$\beta F_{\rho_i}(r)=m_i\ln\left(\rho_i(r)\Lambda_i^3\right)+\beta\hat{F}_{\rho_i}^\mathrm{res}(r)$$

$$\beta F_{\rho_i}^\mathrm{b}=m_i\ln\left(\rho_i^\mathrm{b}\Lambda_i^3\right)+\beta\hat{F}_{\rho_i}^\mathrm{b,res}$$

and lead to a slightly modified Euler-Lagrange equation

$$\rho_i(r)=\rho_i^\mathrm{b}e^{\frac{\beta}{m_i}\left(\hat F_{\rho_i}^\mathrm{b,res}-\hat F_{\rho_i}^\mathrm{res}(r)-V_i^\mathrm{ext}(r)\right)}$$

## Heterosegmented chains
The expressions are more complex for models in which density profiles of individual segments are considered. A derivation is given in the appendix of [Rehner et al. (2022)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.034110). The resulting Euler-Lagrange equation is given as

$$\rho_\alpha(r)=\Lambda_i^{-3}e^{\beta\left(\mu_i-\hat F_{\rho_\alpha}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

with the bond integrals $I_{\alpha\alpha'}(r)$ that are calculated recursively from

$$I_{\alpha\alpha'}(r)=\int e^{-\beta\left(\hat{F}_{\rho_{\alpha'}}(r')+V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r)\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\mathrm{d}r$$

The index $\alpha$ is used for every segment on component $i$, $\alpha'$ refers to all segments bonded to segment $\alpha$ and $\alpha''$ to all segments bonded to $\alpha'$. 
For bulk systems the expressions simplify to

$$\rho_\alpha^\mathrm{b}=\Lambda_i^{-3}e^{\beta\left(\mu_i-\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}\right)}$$

which shows that, by construction, the density of every segment in a molecule is identical in a bulk system. The index $\gamma$ refers to all segments on moecule $i$. The expressions can be combined in a similar way as for the molecular DFT:

$$\rho_\alpha(r)=\rho_\alpha^\mathrm{b}e^{\beta\left(\sum_\gamma\hat F_{\rho_\gamma}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

At this point it can be numerically useful to redistribute the bulk contributions back into the bond integrals

$$\rho_\alpha(r)=\rho_\alpha^\mathrm{b}e^{\beta\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

$$I_{\alpha\alpha'}(r)=\int e^{\beta\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r)\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\mathrm{d}r$$

## Combined expression
To avoid having multiple implementations of the central part of the DFT code, the different descriptions of molecules can be combined in a single version of the Euler-Lagrange equation:

$$\rho_\alpha(r)=\rho_\alpha^\mathrm{b}e^{\frac{\beta}{m_\alpha}\left(\hat F_{\rho_\alpha}^\mathrm{b,res}-\hat F_{\rho_\alpha}^\mathrm{res}(r)-V_\alpha^\mathrm{ext}(r)\right)}\prod_{\alpha'}I_{\alpha\alpha'}(r)$$

$$I_{\alpha\alpha'}(r)=\int e^{\frac{\beta}{m_{\alpha'}}\left(\hat F_{\rho_{\alpha'}}^\mathrm{b,res}-\hat F_{\rho_{\alpha'}}^\mathrm{res}(r')-V_{\alpha'}^\mathrm{ext}(r')\right)}\left(\prod_{\alpha''\neq\alpha}I_{\alpha'\alpha''}(r')\right)\omega_\mathrm{chain}^{\alpha\alpha'}(r-r')\mathrm{d}r'$$

If molecules consist of single (possibly non-spherical) segments, the Euler-Lagrange equation simplifies to that of the homosegmented chains shown above. For heterosegmented chains, the correct expression is obtained by setting $m_\alpha=1$.
