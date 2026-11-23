# Refined Damping Estimation for Machining Structures

## Abstract

Experimental Modal Analysis (EMA) has played a crucial role since the late 1940s.
Today, many engineers consider the field to be a closed case,
with mature commercial software providing standard solutions across various industries.
However, a close inspection of the underlying algorithms reveals
that significant room for improvement remains in many relevant cases,
particularly in systems with nonproportional damping. 

General structures in machining exhibit nonproportional damping,
rendering the standard assumption of proportional damping untenable.
Accurately modeling these characteristics is vital for modern manufacturing applications,
such as chatter suppression utilizing tuned mass dampers
or stability analysis in multi-cutter systems. 

Currently, mainstream software for EMA is proprietary,
which limits accessibility and obscures specific algorithmic implementations.
To benefit the engineering and research community,
we introduce Indar, an open-source Python package for EMA.
Among other advanced capabilities,
Indar implements robust algorithms tailored specifically
for identifying nonproportional damping systems. 

## Introduction

The finish quality of a workpiece is affected by multiple factors,
such as the dynamic response of the machining structure
to the forces involved during manufacturing.
Therefore, it is important to model the structure and operating forces
to mitigate the effect of vibrations,
and to avoid unstable conditions that could give rise to chatter [Muñoa 2016]. 

One of the main frameworks for the modeling of vibrating structures
is Experimental Modal Analysis (EMA),
which relies on the assumption that the dynamics are linear.
A standard mechanical model, based on viscous dissipation, is 

$$M\ddot{x} + C\dot{x} + Kx = f \tag{1}$$

where $x \in \mathbb{R}^n$ is the displacement from equilibrium,
$n$ are the Degrees of Freedom (DoF),
and $M$, $C$, and $K$ are the mass, damping, and stiffness matrices,
which are symmetric and positive definite [Ewins 2000].

Mature software for EMA includes Simcenter™ Testlab™ and BK Connect®,
and many consider them the ultimate solutions.
However, despite the good performance of these programs,
there are still relevant situations where there is room for improvement
over the current algorithmic approaches.
On the other hand, commercial software limits accessibility, hides the implementation of the algorithms, and
does not allow modification of the algorithms.

For these reasons, 
and as part of a more ambitious revision of EMA,
we developed the open-source Python package *indar* [Indar 2026];
in Basque, indar means force.
This package offers capabilities such as methods for leakage-free FRF estimation,
alternative algorithms for pole estimation of challenging configurations, or
methods for retrieval of the mechanical system from partial information;
see [Ponce-Vanegas 2026] for a more comprehensive description.

In this paper we focus on one of the strengths of *indar*,
that is, in the estimation of systems with nonproportional damping.
Here, a brief theoretical review is presented
together with a simple but telling example
that shows the relevance of nonproportional damping in machining.
However, due to the very challenging nature of these systems,
*indar* does not offer a universal solution yet,
so continued research is still needed to cover a broader variety of systems.
To the best of our knowledge,
the only program that truly tries to fit nonproportional damping
is the Structural Dynamics Toolbox (SDT) [Balmes 2019].

A typical assumption in EMA workflows is
that there exists a coordinate basis that decouples
the mechanical system (1) into $n$ single DoF oscillators;
these systems are known as proportionally damped systems.
However, these systems constitute a negligible fraction
of all possible mechanical systems,
and this assumption could severely misrepresent a vibrating structure,
which in turn could imply wrong conclusions about its stability.

Fortunately, if two modes with natural (complex) frequencies 
$\lambda_j \neq \lambda_i$,
where $\lambda = \omega(\zeta + i\sqrt{1-\zeta^2})$,
satisfy the relationship 

$$\frac{|\omega_i - \omega_j|}{\omega_i} \ll \zeta_i \tag{2}$$

where the symbol $\ll$ reads "much smaller than",
then the error introduced by ignoring the coupling between both modes
is negligible [Hasselman 1976].
Notice, however, that this relationship is violated
when the modes are close to each other, or the damping is sufficiently large.
An interesting analysis and physical reinterpretation of these systems
was proposed in [Baldelomar 2026].

In any case, the conditions under which significant coupling between modes occurs
are not rare in practice.
This motivated, for example,
the study and modelling of machine tools with nonproportional damping
in [Neugebauer 2010]. 

In [Reith 2017], during the analysis of a multi-cutter turning structure,
the authors noticed a significant coupling between modes
due to the large damping introduced by the fixturing.
After comparing the stability lobes generated by models
with coupled and decoupled modes,
the authors concluded that nonproportional damping
better reflected the properties of the stability lobes. 

Other systems prone to exhibit significant coupling between modes
are Tuned Mass Dampers (TMD) because of the proximity between the modes.
Recently, slender structures modelling tall buildings with TMDs
located along the structure were studied in [Bellos 2025],
showing a noticeable difference when the couplings are ignored.
In the context of machining,
TMDs are used for chatter suppression [Muñoa 2015].
For boring operations, TMDs have been designed [Hintze 2023]
using peak-picking methods for system identification,
which are inappropriate for systems with nonproportional damping
since they cannot properly capture the couplings.

### AI disclaimer

The main research ideas and content of this paper were devised by the authors.
AI tools were extensively used for theoretical derivation
and software development through GitHub Copilot in VS Code.
The repository, including the agentic structure used for this work,
is available at https://github.com/poncev/2027_HSM_damping_estimation.
The chat sessions are considered raw research data
and are available upon request before 1 March 2027, when they will be deleted.


## Theoretical preliminaries

The standard approach in EMA assumes proportional damping,
which is equivalent to the possibility of choosing real mode shapes.
This assumption drastically simplifies the analysis
and is embedded in most commercial software.
In general, the mode shapes $\{\mathbf{\varphi}_l\}$ are complex, and
they must satisfy the compatibility condition
$$\text{Im}\Big(\sum_l\varphi_{il}\varphi_{jl}\Big) = 0 \quad \text{for all } i, j; \tag{3}$$
see [Balmes 1997].
If the mode shapes are real, then
this condition is trivially satisfied.

The compatibility condition (3) admits a physical interpretation.
If the compatibility condition were violated by the mode shapes,
then the IRF for displacement (also known as receptance)
would not vanish at time zero, which
implies that a structure at rest would suffer a finite displacement instantaneously (infinite acceleration) when excited.
Most EMA software does not enforce this constraint
when fitting complex mode shapes.

To fit the mode shapes,
one must first estimate the poles $\lambda_i$ and amplitudes $A_{ij}^l$ using the IRF $\mathbf{h} = (h_{ij})$, which satisfies
$$h_{ij}(t) = \sum_l A_{ij}^l e^{\lambda_l t}.$$
Ideally, the amplitudes are $A_{ij}^l = \varphi_{il}\varphi_{jl}$, but
this is never the case experimentally.
To fit the mode shapes (and assuming acceleration is measured),
we use a loss function based on the Hilbert--Schmidt norm:
$$R_{ij}^k = \sum_l (A_{ij}^l - \varphi_{il}\varphi_{jl})\lambda_l^2 s_l^k, \tag{4a}$$
$$\mathcal{L}(\Phi) = \sum_{i,j}\sum_k (1-k/N)\left|\text{Im}[R_{ij}^k]\right|^2, \tag{4b}$$
where $s_l = e^{\lambda_l \Delta t}$ are the discrete-time poles,
and the summation runs over output nodes $i$, input nodes $j$, and time samples $k$.
The mode shapes must satisfy the constraints:
$$\text{Im}\left(\sum_l\varphi_{il}\varphi_{jl}\right) = 0, \quad
\text{Im}\left(\sum_l \lambda_l\varphi_{il}\varphi_{jl}\right) > 0, \quad
-\text{Im}\left(\sum_l \lambda_l^2\varphi_{il}\varphi_{jl}\right) > 0, \tag{5}$$
for all $i$, $j$, where the inequalities are understood as matrix inequalities.
The first constraint is the compatibility condition (3),
while the remaining constraints enforce positive definiteness of the reconstructed damping and stiffness matrices.
Mode shapes are normalized using the reduced normalization convention:
$\lambda^2\Phi^T M \Phi - \Phi^T K \Phi = 2i\lambda$.

Once the amplitudes $A_{ij}^l$ are estimated,
the real mode shapes are estimated by minimization of (4) using dual annealing,
a global solver that refines candidate minima using gradient-based algorithms.
After fitting the real modes,
complex mode shapes are locally searched using a Trust-Region constrained method.
The theoretical details behind the optimization
can be found in [Ponce-Vanegas 2026, Ponce-Vanegas 2026b].

## Stability Analysis

For turning operations, chatter stability is determined by analyzing
the characteristic equation of a delayed differential equation.
When the regenerative cutting force acts at a single point (coordinate $x_1$),
the stability boundary can be found analytically.
We express the problem in normalized spindle angle coordinates $\theta = \Omega t$,
where $\Omega$ is the spindle speed and the delay becomes $\tau = 1$ revolution.

The key function for stability analysis is:

$$H(\beta, \Omega) = G_{11}(\beta, \Omega) \left(e^{-i\beta} - 1\right) \tag{6}$$

where $\beta$ is the characteristic exponent,
$G_{11}$ is the $(1,1)$ entry of the mechanical receptance matrix $A^{-1}$,
and $A = -\Omega^2\beta^2 M + i\Omega\beta C + K$.
The stability boundary corresponds to pure imaginary exponents ($\beta = i\omega$) where:

$$\text{Im}[H(i\omega, \Omega)] = 0, \quad h = \frac{1}{K_n \text{Re}[H(i\omega, \Omega)]} \tag{7}$$

where $h$ is the chip thickness and $K_n$ is the cutting coefficient.

This approach works efficiently for single-point forces
because the matrix determinant lemma reduces the problem to scalar computations.
For more complex geometries where forces act at multiple nodes,
analytical solutions break down,
making accurate modal identification through tools like Indar even more critical.
We focus here on the simpler single-point case
to clearly illustrate the effects of nonproportional damping on stability predictions.

For details about the numerical implementation of the construction of the stability lobes,
the reader is referred to the paper's repository.

## Example: Turning with Tuned Mass Damper

To illustrate the practical implications of nonproportional damping and
how the assumption of proportionality affects the conclusions,
we analyzed a 2-DoF system representing a turning tool
equipped with a Tuned Mass Damper (TMD),
following a similar setup as in [Muñoa 2015, Yang 2010].
The system parameters are summarized in Table 1.

**Table 1.** System parameters for turning tool with TMD

| Parameter | Value | Description | Reference |
|-----------|-------|-------------|----------|
| $m_1$ | 2.0 kg | Tool modal mass | [Yang 2010] |
| $f_{n1}$ | 350 Hz | Tool natural frequency (without TMD) | [Yang 2010] |
| $\zeta_1$ | 0.025 | Tool damping ratio | [Yang 2010] |
| $\mu$ | 0.05 | TMD mass ratio ($m_2/m_1$) | [Den Hartog 1956] |
| $K_n$ | 1.2 GPa | Cutting coefficient (Inconel 718) | [Pawade 2008] |

The TMD was designed following Den Hartog optimal tuning criteria [Den Hartog 1956]
with a mass ratio $\mu = 0.05$.
This configuration intentionally creates closely spaced modes to suppress vibrations.
The resulting system exhibits nonproportional damping.

For simplicity, we assume the modal parameters (poles and residue amplitudes)
are exactly known.
The objective is to recover the mode shapes using the loss function (4)
and test whether the proportional damping assumption introduces significant errors.
The system matrices $M$, $C$, and $K$ are reconstructed using two different approaches:

1. **RealModes**: Assumes proportional damping and fits real mode shapes to the data.
This approach mimics the standard algorithms used in commercial EMA software.

2. **ComplexModes**: Properly handles nonproportional damping
by preserving the complex nature of the mode shapes.
This method is implemented in Indar and respects the compatibility condition (3).

Both reconstructed systems were used to compute stability lobe diagrams
using the approach described in the previous section.

Figure 1 compares the frequency response functions (FRF)
reconstructed from both methods against the true system response.
The most striking difference appears in the damper FRF (coordinate 2),
where forcing real mode shapes produces visible discrepancies from the true response.
The tool tip FRF (coordinate 1) also shows deviations, though less pronounced.

Figure 2 presents the stability lobe diagrams
in the spindle speed range (50-600 rad/s),
where the pocket structure near the natural frequencies is most pronounced.
In this critical region,
the differences between the stability predictions from the two methods become apparent,
with the proportional damping assumption yielding predictions
that deviate from the true system behavior.

These results demonstrate that forcing the proportional damping assumption
on inherently nonproportional systems—as is standard practice
in commercial EMA software—can lead to substantial errors
in the reconstructed dynamic model.
For machining applications where chatter stability margins are often narrow,
such errors could result in incorrect stability predictions
and suboptimal process parameters.

## Conclusions

Nonproportional damping occurs in various machining configurations,
including systems equipped with tuned mass dampers for chatter suppression,
multi-cutter setups, and structures with closely spaced modes and high damping ratios.
While TMDs are designed to suppress chatter,
accurately characterizing their dynamics through EMA requires models
that can handle nonproportional damping.
The compatibility condition $\text{Im}(\Phi^T \Phi) = 0$
must be satisfied by all physical systems to prevent
infinite acceleration at excitation onset.
However, mainstream commercial EMA software systematically ignores this constraint
when fitting complex mode shapes to nonproportional damping systems.

Our example with a 2-DoF turning tool equipped with a TMD demonstrates
that forcing the proportional damping assumption on nonproportional systems
can lead to substantial errors in the reconstructed FRFs and stability lobe predictions.
Such errors could compromise chatter suppression strategies
and process optimization efforts.

The open-source Indar package provides researchers and engineers
with accessible tools to properly identify and characterize
nonproportional damping in experimental data.
As machining processes continue to advance toward higher precision
and more complex configurations,
accurate modal models that respect the true damping structure
become increasingly essential for reliable chatter prediction and suppression.

## Acknowledgments

This research was supported by the Momentum MSCA Programme,
funded by the Horizon Europe programme
under the HORIZON-MSCA-2023-COFUND [grant No. 101179854],
and co-funded by the Hungarian Academy of Sciences (MTA).

## References

[Baldelomar 2026] Baldelomar, L. M., et al. Understanding modal interactions in non-classically damped linear oscillators with closely spaced modes. Journal of Sound and Vibration, 2026, Vol. 638, pp 119818. ISSN 0022-460X

[Balmes 1997] Balmes, E. New results on the identification of normal modes from experimental complex modes. Mechanical Systems and Signal Processing, 1997, Vol. 11, No. 2, pp 229-243. ISSN 0888-3270

[Balmes 2019] Balmes, E., et al. Structural Dynamics Toolbox User's Guide. Paris: SDTools, 2019. Available at www.sdtools.com

[Bellos 2025] Bellos, J., et al. Theoretical evaluation and experimental validation of an innovative dynamic analysis method for combined structures with Tuned Mass Dampers. In: M. Papadrakakis and M. Fragiadakis, eds. Proceedings of 10th ECCOMAS Thematic Conference on Computational Methods in Structural Dynamics and Earthquake Engineering, Rhodes Island, 15-18 June, 2025. Athens: National Technical University of Athens, pp 5562-5576. ISBN 978-618-5827-06-9

[Den Hartog 1956] Den Hartog, J. P. Mechanical Vibrations. 4th ed. New York: McGraw-Hill, 1956.

[Ewins 2000] Ewins, D.J. Modal testing: theory, practice and application. Baldock: Research Studies Press Ltd., 2000. ISBN 0863802184

[Hasselman 1976] Hasselman, T.K. Modal coupling in lightly damped structures. AIAA Journal, 1976, Vol. 14, No. 11, pp 1627-1628. ISSN 0001-1452

[Hintze 2023] Hintze, W., et al. Model based design of tuned mass dampers for boring bars of small diameter. Procedia CIRP, 2023, Vol. 117, pp 193-198. ISSN 2212-8271

[Indar 2026] Ponce-Vanegas, F. Indar Documentation. ReadTheDocs, 2026 [online]. Created on 20 May 2026. Available at indar.readthedocs.io

[Muñoa 2015] Muñoa, J., et al. Tuneable mass dampers with variable stiffness for chatter suppression. In: Proceedings of the 4th International Conference on Virtual Machining Process Technology, Vancouver, 2-5 June 2015. Vancouver: University of British Columbia

[Muñoa 2016] Muñoa, J., et al. Chatter suppression techniques in metal cutting. CIRP Annals, 2016, Vol. 65, No. 2, pp 785-808. ISSN 0007-8506

[Neugebauer 2010] Neugebauer, R., et al. Advanced state space modeling of non-proportional damped machine tool mechanics. CIRP Journal of Manufacturing Science and Technology, 2010, Vol. 3, No. 1, pp 8-13. ISSN 1755-5817

[Pawade 2008] Pawade, R. S., Joshi, S. S. and Brahmankar, P. K. Effect of machining parameters and cutting edge geometry on surface integrity of high-speed turned Inconel 718. International Journal of Machine Tools and Manufacture, 2008, Vol. 48, No. 1, pp 15-28. ISSN 0890-6955

[Ponce-Vanegas 2026] Ponce-Vanegas, F., et al. Revisiting Experimental Modal Analysis: leakage-free estimation, pole identification, and non-proportional damping. BIRD, 2026. Available at http://hdl.handle.net/20.500.11824/2165

[Ponce-Vanegas 2026b] Ponce-Vanegas, F. On complex mode shapes and their identification. Journal of Physics: Conference Series, 2026, Vol. 3190, No. 1, pp 012009. ISSN 1742-6596

[Reith 2017] Reith, M.J. and Stepan, G. Effect of non-proportional damping on the dynamics and stability of multi-cutter turning systems. International Journal of Machine Tools and Manufacture, June 2017, Vol. 117, pp 23-30. ISSN 0890-6955

[Yang 2010] Yang, Y., Munoa, J. and Altintas, Y. Optimization of multiple tuned mass dampers to suppress machine tool chatter. International Journal of Machine Tools and Manufacture, 2010, Vol. 50, No. 9, pp 834-842. ISSN 0890-6955