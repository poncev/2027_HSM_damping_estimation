Project Handover: Stability Lobes for Turning with a 2-DoF Tool + Tuned Mass Damper
Problem Setup

The mechanical system consists of:

M=[m100m2],C=[c1+c2−c2−c2c2],K=[k1+k2−k2−k2k2].M = \begin{bmatrix} m_1 & 0 \\ 0 & m_2 \end{bmatrix}, \quad C = \begin{bmatrix} c_1+c_2 & -c_2 \\ -c_2 & c_2 \end{bmatrix}, \quad K = \begin{bmatrix} k_1+k_2 & -k_2 \\ -k_2 & k_2 \end{bmatrix}.

Coordinate x1x_1 is the tool displacement and x2x_2 is the absorber displacement.

The regenerative cutting force acts only on the tool coordinate:

D=Eh[1000]=Eh e1e1T.D = Eh \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} = Eh\,e_1 e_1^T.

The equations have been normalized with spindle angle:

θ=Ωt,\theta=\Omega t,

so derivatives transform as:

ddt=Ωddθ,d2dt2=Ω2d2dθ2.\frac{d}{dt}=\Omega\frac{d}{d\theta}, \qquad \frac{d^2}{dt^2}=\Omega^2\frac{d^2}{d\theta^2}.

The delay is therefore constant:

τ=1\tau=1

(one revolution in angular coordinates).

The governing equation is

Ω2Mq′′+ΩCq′+Kq=D [q(θ−1)−q(θ)].\Omega^2Mq''+\Omega Cq'+Kq = D\,[q(\theta-1)-q(\theta)].
Important Discussion Outcome

Initially there was confusion about whether the delay had been absorbed into MM and CC.

It has not.

The factors Ω2\Omega^2 and Ω\Omega appear only because of the change of independent variable from time to spindle angle.

The characteristic equation is therefore still delay-dependent through

e−λ,e^{-\lambda},

where

q=veλθ.q=v e^{\lambda \theta}.

Substituting gives

det⁡(Ω2λ2M+ΩλC+K−Eh(e−λ−1)e1e1T)=0.\det \Big( \Omega^2\lambda^2M + \Omega\lambda C + K - Eh(e^{-\lambda}-1)e_1e_1^T \Big) = 0.
Key Simplification (Rank-1 Delay Matrix)

Since

D=Eh e1e1TD=Eh\,e_1e_1^T

is rank one, the matrix determinant lemma reduces the characteristic determinant to a scalar equation.

Define

G(λ,Ω)=(Ω2λ2M+ΩλC+K)−1.G(\lambda,\Omega) = \left( \Omega^2\lambda^2M + \Omega\lambda C + K \right)^{-1}.

Let

G11=e1TGe1.G_{11} = e_1^TGe_1.

Then the exact characteristic equation becomes

1−Eh G11(λ,Ω) (e−λ−1)=0.1-Eh\,G_{11}(\lambda,\Omega)\, (e^{-\lambda}-1) = 0.

This reduction is exact.

No SDM, FDM, or D-decomposition is required.

The entire stability problem reduces to the scalar receptance G11G_{11}.

Stability Boundary

On the Hopf boundary:

λ=jβ.\lambda=j\beta.

Define

H(β,Ω)=G11(jβ,Ω)(e−jβ−1).H(\beta,\Omega) = G_{11}(j\beta,\Omega) \bigl(e^{-j\beta}-1\bigr).

The characteristic equation becomes

1−EhH=0.1-Eh H=0.

Separating real and imaginary parts:

ℑ(H)=0,\Im(H)=0, Eh=1ℜ(H).Eh=\frac{1}{\Re(H)}.

Hence for a fixed spindle speed Ω\Omega:

Solve
ℑ(H)=0\Im(H)=0

for β\beta.

Compute
h=1Eℜ(H).h = \frac{1} {E\Re(H)}.

Keep only positive hh.

The smallest positive hh is the stability boundary.

Numerical Strategy Discussed
Option 1: Brute-Force Root Search

For fixed Ω\Omega:

Compute

f(β)=ℑ(H).f(\beta) = \Im(H).

Evaluate on a dense vectorized grid.

Find sign changes.

Use Brent's method / scipy.optimize.root_scalar / brentq to refine roots.

Compute corresponding hkh_k.

Set

hlim=min⁡(hk>0).h_{\rm lim} = \min(h_k>0).

This is robust.

Option 2: Continuation (Preferred)

An important observation:

When

Ω=0,\Omega=0,

the roots are known analytically.

At Ω=0\Omega=0,

G11=real constant,G_{11} = \text{real constant},

therefore

ℑ(H)=−G11sin⁡β.\Im(H) = -G_{11}\sin\beta.

Hence

βk=kπ.\beta_k=k\pi.

This gives explicit initialization for all branches.

The idea proposed was:

Start from Ω=0\Omega=0.
Initialize roots
βk(0)=kπ.\beta_k(0)=k\pi.
Increase Ω\Omega gradually.
For each branch, use the previous solution as initial guess.
Track
βk(Ω)\beta_k(\Omega)

instead of repeatedly scanning the whole axis.

This is essentially a continuation method and is expected to work well because roots move continuously with Ω\Omega.

A predictor-corrector continuation method may improve efficiency.

Important Observation About Trivial Roots

At

β=2πn,\beta=2\pi n, e−jβ−1=0.e^{-j\beta}-1=0.

Therefore

H=0H=0

for all spindle speeds.

Consequently

ℑ(H)=0\Im(H)=0

for all spindle speeds.

Thus:

β=2πn\beta=2\pi n

is always a root of the imaginary-part equation.

However:

1−EhH=01-EhH=0

becomes

1=0,1=0,

which is impossible.

Equivalently:

h=∞.h=\infty.

Therefore these roots are "trivial phase roots" and do not produce finite stability limits.

Open Question Encountered

While plotting ℑ(H)\Im(H), more than one nontrivial root was sometimes observed between consecutive even roots 2πn2\pi n.

Originally it was assumed there should be only one branch between them.

This assumption is not justified.

The equation actually solved is

−GRsin⁡β+GI(cos⁡β−1)=0,-G_R\sin\beta + G_I(\cos\beta-1) = 0,

where

G11=GR+jGI.G_{11}=G_R+jG_I.

Since GRG_R and GIG_I vary strongly near resonances and antiresonances of the TMD-equipped structure, additional roots may legitimately appear.

This issue remained unresolved and should be investigated further.

Possible causes:

Genuine extra branches caused by resonant behavior of G11G_{11}.
Numerical artifacts near the trivial roots 2πn2\pi n.

Suggested diagnostic:

Plot

ℑ(H),GR,GI\Im(H), \quad G_R, \quad G_I

in intervals where multiple roots appear.

Discussion on Classical Altintas vs Angular Formulation

The original question was whether it is better to:

Classical formulation
s2M+sC+K−D(e−sτ−1)=0s^2M+sC+K-D(e^{-s\tau}-1)=0

with

τ=1Ω,\tau=\frac1\Omega,

or

Angular formulation
Ω2λ2M+ΩλC+K−D(e−λ−1)=0.\Omega^2\lambda^2M+\Omega\lambda C+K-D(e^{-\lambda}-1)=0.

Conclusions:

Classical Altintas

Pros:

Explicit phase/depth formulas.
FRF independent of spindle speed.
Easier analytical derivation.

Cons:

Delay changes with spindle speed.
Angular formulation

Pros:

Fixed delay (τ=1\tau=1).
Better suited to root continuation.
More natural if equations are already normalized by spindle angle.

Cons:

G11G_{11} depends on Ω\Omega.
Classical closed-form Altintas formulas are lost.

For this project, because the goal is to generate stability maps numerically for many absorber configurations, the angular formulation was judged perfectly acceptable.

Intended Final Goal

Generate stability lobe diagrams

(Ω,h)(\Omega,h)

for several tuned mass damper configurations:

(m2,k2,c2).(m_2,k_2,c_2).

Proposed computational pipeline:

For each absorber design:

For each spindle speed Ω\Omega:

Track or compute all roots βk\beta_k of
ℑ(H)=0.\Im(H)=0.
Compute
hk=1Eℜ(Hk).h_k = \frac1{E\Re(H_k)}.

Keep positive values.

Define

hlim=min⁡khk.h_{\rm lim} = \min_k h_k.
Plot
hlim(Ω).h_{\rm lim}(\Omega).

This curve is the stability boundary.

Notes on Implementation

Python recommendations discussed:

Vectorize evaluation of f(β)f(\beta).
Use NumPy arrays for G11G_{11}.
Use scipy.optimize.brentq or root_scalar.
Root finding itself should remain scalar even if function evaluation is vectorized.

Continuation from kπk\pi at Ω=0\Omega=0 is likely more elegant and faster than repeated global scanning.