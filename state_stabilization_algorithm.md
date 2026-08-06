# A Reformulated AUTOSCATTER for Dissipative Gaussian State Stabilization

## What changes and what stays

AUTOSCATTER is two nested machines. The **outer** machine is a graph search: start from the fully connected graph, prune edges, file each graph as *valid* or *invalid*, and let those verdicts propagate (validity upward, invalidity downward). The **inner** machine is an *oracle*: handed one graph, it decides valid or invalid. Everything worth keeping — the efficient enumeration, the pruning, the interpretable irreducible graphs — lives in the outer machine, and the outer machine is provably correct. The danger lives entirely in the inner machine, because the original oracle is a gradient-descent optimizer that can report a false *invalid* when it merely gets stuck.

So the plan is: **keep the outer search essentially unchanged, and replace the oracle.** We can do this because state stabilization is easier than scattering synthesis in exactly the way that matters. In the scattering problem the target depends on the Hamiltonian through a tangled matrix inverse, forcing a guess-and-check optimizer. In our problem the target is a *state* — a covariance matrix — and once we fix it, the stationarity condition becomes **linear** in the Hamiltonian. A linear problem is solved in one shot, with no local minima and no false negatives, and it can even certify impossibility.

The rest of this document derives the moment equations, performs the reframing, shows how the *full* covariance is determined from a target specified only on the signal modes, states the new continuous oracle in two forms (an exact linear solve and a certifying SDP), and gives the discrete search that wraps it.

---

## 1. The model and the moment equations

Take $n$ modes total ($N$ target modes plus auxiliaries). Collect the quadratures into $x = (q_1,p_1,\dots,q_n,p_n)^\top$ with the canonical commutator $[x_i,x_j] = i\,\Omega_{ij}$, where

$$
\Omega = \bigoplus_{k=1}^{n}\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}, \qquad \Omega^\top = -\Omega, \qquad \Omega^2 = -\mathbb{1}.
$$

The dynamics are a Gaussian Lindblad equation with a quadratic Hamiltonian and jump operators linear in the quadratures,

$$
H = \tfrac12\, x^\top G\, x \quad (G = G^\top \text{ real}), \qquad L_\mu = c_\mu^\top x \quad (c_\mu \in \mathbb{C}^{2n}).
$$

Stack the jump vectors as the rows of $C$ (so row $\mu$ is $c_\mu^\top$) and define the Hermitian, positive-semidefinite **dissipation matrix**

$$
\Upsilon \;=\; C^\dagger C \;=\; \sum_\mu c_\mu^* c_\mu^\top \;\succeq\; 0 .
$$

Two facts about $\Upsilon$ carry the whole reformulation. First, it is the *only* object the dissipators enter through at the level of moments — individual jump operators never appear, only the sum $\Upsilon$. Second, **any** Hermitian PSD $\Upsilon$ factorizes as $C^\dagger C$, and $\operatorname{rank}\Upsilon$ equals the number of independent dissipative channels. So "$\Upsilon\succeq 0$" is exactly the condition of physical realizability, and its rank is the channel count — which is precisely the quantity your single-channel / flat-Williamson-spectrum results already control.

### Drift and diffusion

Working in the Heisenberg (adjoint) picture, the first moments obey $\frac{d}{dt}\langle x\rangle = A\langle x\rangle$. The Hamiltonian contribution is obtained from $i[H,x_k]$:

$$
i[H,x_k] = (\Omega G\, x)_k \;\Rightarrow\; A_H = \Omega G,
$$

and the dissipative contribution from $\sum_\mu \big(L_\mu^\dagger x_k L_\mu - \tfrac12\{L_\mu^\dagger L_\mu, x_k\}\big)$, whose commutators $[x_k,L_\mu]$ and $[L_\mu^\dagger,x_k]$ are c-numbers, collapses to

$$
A_D = \tfrac{i}{2}\,\Omega\big(\Upsilon^* - \Upsilon\big) = \Omega\,\operatorname{Im}\Upsilon ,
$$

using that $\Upsilon$ Hermitian means $\operatorname{Re}\Upsilon$ is symmetric and $\operatorname{Im}\Upsilon$ is antisymmetric, so $\Upsilon^*-\Upsilon = -2i\operatorname{Im}\Upsilon$. Hence the **drift matrix**

$$
\boxed{\,A \;=\; \Omega\big(G + \operatorname{Im}\Upsilon\big)\,}.
$$

The symmetrized covariance $V_{ij} = \tfrac12\langle\{x_i,x_j\}\rangle - \langle x_i\rangle\langle x_j\rangle$ then evolves by the standard drift–diffusion law

$$
\dot V \;=\; A V + V A^\top + D , \qquad \boxed{\,D \;=\; \Omega\,(\operatorname{Re}\Upsilon)\,\Omega^\top \;\succeq\; 0\,}.
$$

Three things to notice.

**(a) Drives drop out.** Linear driving terms shift the first moments only; mean and covariance decouple, so $V$ never sees a drive amplitude. This is unconditional.

**(b) The scale freedom is a *joint* $(G,\Upsilon)$ gauge, not a freedom in $\Upsilon$ alone.** It is tempting — and wrong — to say "$V$ is blind to absolute rates." Rescaling the dissipators alone, $\Upsilon\mapsto s\Upsilon$, multiplies $A_D$ and $D$ by $s$ but leaves $A_H=\Omega G$ untouched, so it changes the balance between coherent and dissipative dynamics and *does* move $V$. What is genuinely invariant is the simultaneous rescaling

$$
(G,\Upsilon) \;\longmapsto\; (sG,\, s\Upsilon), \qquad s>0
\quad\Longrightarrow\quad (A,D)\mapsto(sA,sD) \quad\Longrightarrow\quad V \text{ unchanged},
$$

since $(sA)V+V(sA)^\top+sD = s\,[AV+VA^\top+D]$. Only the *ratios* of couplings to rates matter. Two consequences used later: the solution set of $(\star\star)$ is a cone (if $(G,\Upsilon)$ solves it so does $(sG,s\Upsilon)$), so one overall scale may be fixed by fiat without loss; and when $\Upsilon$ is held fixed and $G$ solved for, changing the auxiliary rate $\kappa$ simply rescales the returned $G$ — which is why any finite $\kappa$ is as good as any other *provided $G$ moves with it*.

**(c) Rank.** $D\succeq 0$ inherits its rank from $\operatorname{Re}\Upsilon$, hence from the channel count.

---

## 2. The reframing: fix the state, solve for the generator

Stationarity is the continuous Lyapunov equation

$$
A V + V A^\top + D \;=\; 0 . \tag{$\star$}
$$

The *forward* oracle treats $(G,\Upsilon)$ as known, solves $(\star)$ for $V$, and compares to $V_{\text{target}}$. That is the guess-and-check route, and it is where the original algorithm's false negatives come from: solving $(\star)$ forward requires $A$ to be Hurwitz, so off the stability region the map $(G,\Upsilon)\mapsto V$ is undefined and the optimizer is descending a landscape with a ragged boundary.

Invert the roles. We *know* $V$. Substitute it into $(\star)$ and read the equation as a condition on the **unknowns $G$ and $\Upsilon$**. Expanding $A$ and $D$, and using $(G+\operatorname{Im}\Upsilon)^\top = G - \operatorname{Im}\Upsilon$ and $\Omega^\top=-\Omega$,

$$
\Omega(G+\operatorname{Im}\Upsilon)V \;-\; V(G-\operatorname{Im}\Upsilon)\Omega \;+\; \Omega(\operatorname{Re}\Upsilon)\Omega^\top \;=\; 0 . \tag{$\star\star$}
$$

Every term is **linear** in the unknowns $G$, $\operatorname{Im}\Upsilon$, $\operatorname{Re}\Upsilon$. That single observation is the engine of the whole method. The tangled forward map has become a linear equation, and linear equations do not have local minima.

It is convenient to separate the Hamiltonian terms from the dissipator terms. Define the linear operator

$$
\mathcal{L}_V(G) \;=\; \Omega G V - V G \Omega ,
$$

and move the $\Upsilon$-dependence to the right-hand side:

$$
\mathcal{L}_V(G) \;=\; B(\Upsilon), \qquad
B(\Upsilon) \;=\; -\Big[\Omega\,(\operatorname{Im}\Upsilon)\,V \;+\; V\,(\operatorname{Im}\Upsilon)\,\Omega \;+\; \Omega\,(\operatorname{Re}\Upsilon)\,\Omega^\top\Big]. \tag{1}
$$

### 2.1 The state-completion step: which $V$ goes into $(\star\star)$

Linearity is not free. It holds **only because $V$ enters $(\star\star)$ as a known constant matrix**. The term $\Omega G V$ has entries $\sum_{k,l}\Omega_{ik}G_{kl}V_{lj}$; if some $V_{lj}$ is itself unknown, that is a product of two unknowns and the equation is *bilinear*, not linear. No SVD, no SDP, and every local minimum the method was designed to eliminate comes straight back.

And $V$ in $(\star\star)$ is the **full** $2n\times 2n$ covariance of signal *and* auxiliary modes, whereas the problem specifies only the signal block. Partition ($T$ = target modes, $A$ = auxiliaries):

$$
V = \begin{pmatrix} V_{TT} & V_{TA} \\ V_{TA}^\top & V_{AA}\end{pmatrix},
$$

with $V_{TT}=V_{\text{target}}$ **given**, and $V_{TA}$ (signal–drain correlations) and $V_{AA}$ (the drain's own steady state) **unknown** — they are outputs of the dynamics, not inputs. A drain whose state you had to specify in advance would not be a drain.

Restricting $(\star\star)$ to its $TT$ block does not evade this. Since $\Omega$ is block diagonal ($\Omega_{TA}=0$),

$$
(\Omega G V)_{TT} \;=\; \Omega_T\big(G_{TT}V_{TT} \;+\; G_{TA}V_{AT}\big),
$$

and the second term is unknown coupling times unknown correlation. Nor may $G_{TA}$ be discarded: that block *is* the stabilization mechanism, and zeroing it disconnects the drain entirely.

The unknown blocks must therefore be eliminated, not ignored. For **pure targets** they can be, exactly, in two steps.

**Step 1 — purity kills $V_{TA}$.** If a marginal state is pure it is uncorrelated with everything else: $\rho_T$ pure $\Rightarrow \rho_{TA}=\rho_T\otimes\rho_A$, and this holds even when the global state is mixed. For Gaussian states that is precisely

$$
V_{\text{target}} \text{ pure} \;\Longrightarrow\; V_{TA}=0 .
$$

Not an assumption — a theorem, and it costs nothing for the targets of interest (two-mode squeezed states, unequally-squeezed EPR states, Cayley cluster states are all pure by construction).

**Step 2 — gauge plus drain purity pins $V_{AA}$.** Step 1 alone is insufficient: with $V_{TA}=0$ the $TT$ block is clean, $(\Omega GV)_{TT}=\Omega_T G_{TT}V_{TT}$, but the $TA$ block reads

$$
(\Omega G V)_{TA} \;=\; \Omega_T\big(G_{TT}\underbrace{V_{TA}}_{=0} + G_{TA}V_{AA}\big) \;=\; \Omega_T\,G_{TA}\,V_{AA},
$$

still unknown times unknown. Remove $V_{AA}$ as follows.

*(i) Gauge.* A local symplectic $S$ acting only on the auxiliary modes maps $V\mapsto (\mathbb{1}\oplus S)V(\mathbb{1}\oplus S)^\top$ and correspondingly redefines $G$ and $\Upsilon$ — both of which are being solved for, so the redefinition costs nothing. Because $V_{TA}=0$ it touches only $V_{AA}\mapsto SV_{AA}S^\top$, so by Williamson we may take

$$
V_{AA} \;=\; \operatorname{diag}(\nu_1,\nu_1,\dots,\nu_{n_{\text{aux}}},\nu_{n_{\text{aux}}}), \qquad \nu_m\ge\tfrac12 .
$$

The unknown block is now $n_{\text{aux}}$ scalars.

*(ii) Purity of the drain.* In a dark-state scheme the target modes lie in the kernel of every jump operator and are dynamically decoupled from the drain in steady state — which is the same statement as $V_{TA}=0$. The drain then relaxes to its own bath state: vacuum, or squeezed vacuum if it is given a squeezed (Bogoliubov) bath. Either is **pure**, so $\nu_m=\tfrac12$ and

$$
\boxed{\;V \;=\; V_{\text{target}} \;\oplus\; \tfrac12\,\mathbb{1}_{2n_{\text{aux}}}\;}
$$

is fully determined. Every term of $(\star\star)$ is now linear in $(G,\Upsilon)$ and the oracles of §4 apply verbatim.

Note that the completed system is generically **over-determined**: $(\star\star)$ is a symmetric $2n\times 2n$ equation, i.e. $n(2n+1)$ independent scalar constraints, against $\dim\mathcal{S}_G+\dim\mathcal{S}_\Upsilon$ unknowns. That is exactly why a nonzero least-squares residual is informative rather than an artifact of an under-specified problem.

**Fallback when the drain purity argument is not available.** Keep $V_{AA}=\operatorname{diag}(\nu_m)$ symbolic. The pair $(\nu,G)$ enters $(\star\star)$ bilinearly, but **for each fixed $\nu$ the problem is linear again**, so the oracle runs unchanged on a grid over $\nu\in[\tfrac12,\nu_{\max}]^{n_{\text{aux}}}$. With one auxiliary mode this is a single one-dimensional scan. Certificates remain valid pointwise: "infeasible at this $\nu$" is proved; "infeasible for all $\nu$" is not, so a grid-wide failure is reported as *undecided* (§4C), never as invalid.

**Scope boundary.** The construction above requires the target block to be exactly pure. It fails, and with it the exactness of the whole method, in two cases:

- **Mixed target.** $V_{TA}\ne 0$, Step 1 does not apply, and $(\star\star)$ is genuinely bilinear.
- **Nonzero intrinsic loss on the signal modes, $\Gamma_M>0$.** The target block is then not exactly pure at steady state, so again $V_{TA}\ne0$. This is the *same* physical situation §6 treats as "limit-only," but it is a logically separate consequence: even before asking whether a finite-parameter solution exists, the linearization itself has stopped being valid. Both are handled by the §6 idealize-then-perturb route, which sets $\Gamma_M=0$, solves exactly here, and reintroduces $\Gamma_M$ as a perturbation.

The idealized regime — pure target, $\Gamma_{\text{signal}}=0$, auxiliary drains on vacuum or squeezed-vacuum baths — is where the method is rigorous, and it is precisely the regime in which the flat-spectrum and universality results are stated.

---

## 3. Stability is automatic — there is nothing to filter

Before the algorithm, the fact that removes half of AUTOSCATTER's post-processing. Suppose $(\star)$ holds with a genuine covariance matrix $V \succ 0$ and $D \succeq 0$. Let $w$ be any left eigenvector of $A$, $w^\dagger A = \lambda w^\dagger$, and set $s = w^\dagger V w > 0$. Because $A,V$ are real and $V=V^\top$, the scalar $z = w^\dagger V A^\top w$ satisfies $z^* = w^\dagger A V w = \lambda s$, so $z = \bar\lambda s$, and therefore

$$
w^\dagger\big(AV + VA^\top\big)w \;=\; (\lambda+\bar\lambda)\,s \;=\; 2\,\operatorname{Re}(\lambda)\,s \;=\; -\,w^\dagger D\, w \;\le\; 0 .
$$

Hence $\operatorname{Re}\lambda \le 0$ for every eigenvalue, and $\operatorname{Re}\lambda = 0$ forces $w^\dagger D w = 0$, i.e. a dark mode. So a positive-definite solution of the stationarity equation *is its own stability certificate*, modulo exactly the non-degeneracy (no-dark-mode) condition you already treat as load-bearing. This is the Zippilli–Vitali uniqueness argument (Horn–Johnson Thm 2.4.7) rewritten in the quadrature basis.

The consequence for the algorithm: **a genuine, attractive solution cannot be unstable.** AUTOSCATTER filters unstable solutions afterward because its scattering targets include amplifiers that sit right at the instability edge; state stabilization has no analogue, because "relaxes to a fixed state and stays there" *is* the definition of stability. So there is no stability *tolerance* and no amplifier filter — but "genuine" must mean *attractive*, and the one thing the oracle must still do is select against the marginal, dark-mode (frozen) configurations that also solve the stationarity equation. That selection is the load-bearing use of the non-degeneracy condition. Note what §3 does *not* give: it rules out $\operatorname{Re}\lambda>0$, so the residual risk is marginality, never instability. The filter is therefore a search for a strictly-Hurwitz member of a solution set all of whose members are already at worst marginal — and it is built into the verdicts of §4.

---

## 4. The continuous optimization (the new oracle)

The oracle answers one yes/no question about a graph: *does a Hamiltonian living on this graph stabilize the target?* A graph is a pattern of allowed matrix entries. Since $H=\tfrac12 x^\top G x$ is fixed by $G$ and each entry is a coupling, "$H$ lives on $\mathcal{G}$" means $G_{ij}=0$ for every absent edge, and removing an edge pins one more entry to exactly zero — nothing rescaled, which is why the search variables must be the couplings themselves, not logarithms or ratios. The admissible matrices therefore form a *linear subspace* $\mathcal{S}_G$ whose dimension is the number of allowed couplings; define $\mathcal{S}_\Upsilon$ the same way for the dissipator.

Hardware constraints are more of the same: "passive mechanical block," "nearest-neighbour hopping only," "reservoir on this site" each say certain entries vanish or are linearly tied. Intersecting subspaces is a subspace, so the graph and all hardware constraints fuse into one admissible $\mathcal{S}_G$ (and $\mathcal{S}_\Upsilon$) — *hard* by construction, never soft penalties the optimizer could trade against the target. The oracle then just checks whether the linearized stationarity equation from §2 — with the completed $V$ of §2.1 — has a solution inside that subspace. Two forms, used together, plus the shared verdict rule of §4C.

### 4A. Exact solve for a fixed dissipator structure — linear least squares

Fix the dissipator directions the graph allows (single channel: one vector $c$ up to scale and phase; generally a low-rank $\Upsilon$ in $\mathcal{S}_\Upsilon$). Then $B(\Upsilon)$ in (1) is known and the equation is *linear in $G$*. Vectorizing with $\operatorname{vec}(MXN)=(N^\top\otimes M)\operatorname{vec}(X)$, $V=V^\top$, $\Omega^\top=-\Omega$:

$$
\operatorname{vec}\!\big(\mathcal{L}_V(G)\big) \;=\; \underbrace{\big[(V\otimes\Omega) + (\Omega\otimes V)\big]}_{\displaystyle \mathbb{L}}\,\operatorname{vec}(G).
$$

Restrict to $\mathcal{S}_G$ with a selection matrix $P$ ($\operatorname{vec}(G)=Pg$) and solve $g^\star=\arg\min_g\|\mathbb{L}P\,g-\operatorname{vec}(B(\Upsilon))\|^2$, which the SVD solves globally. The residual gives the first half of the verdict:

- residual $> 0$: **no** Hamiltonian on this graph realizes the target with this dissipator structure. Note the qualifier: this is a certificate about the *fixed* $\Upsilon$, not about the graph, so it does not stamp the graph invalid — it hands control to 4B.
- residual $= 0$ (to $\sim 10^{-14}$): a stationary solution exists on the graph.

Residual zero is necessary but **not sufficient**, and this is the one place §3's non-degeneracy condition does real work. The solution set is generally an affine space (a particular solution plus the nullspace of $\mathbb{L}P$) and can contain *spurious* members — most importantly a decoupled/frozen configuration whose drift $A$ has eigenvalues on the imaginary axis. Such a point is stationary but not attractive: it leaves a target mode undriven rather than stabilizing it — exactly the dark mode §3 warns about. (Concretely: a single target mode, one auxiliary loss channel, $\Gamma_M=0$; the minimum-norm least-squares point is $G=0$, the mode simply frozen, while the genuine active witness lives elsewhere in the same solution set with $A$ Hurwitz and machine-precision residual.) So the verdict is completed by the **attractivity filter** of §4C, run over the affine solution set $g^\star+\operatorname{null}(\mathbb{L}P)$.

No initial guess, no restarts, no local minima, no $N_{\text{rep}}$ — the false-negative risk AUTOSCATTER mitigates by rerunning is gone from the *linear algebra*. The only addition over a bare linear solve is the attractivity selection, a finite-dimensional eigenvalue search over the (usually small) solution nullspace, whose failure mode is handled explicitly in §4C.

### 4B. Joint certifying oracle — a semidefinite feasibility program

Version 4A fixes the dissipator structure. To decide a graph *without* committing to a particular structure — and to obtain a **sound proof of impossibility across all admissible dissipators** — exploit that $(\star\star)$ is jointly linear in $(G,\Upsilon)$ while the only nonlinearity, $\Upsilon = C^\dagger C$, is the convex constraint $\Upsilon\succeq 0$. Drop the rank constraint and solve the feasibility SDP

$$
\text{find } G\in\mathcal{S}_G,\ \Upsilon = \Upsilon^\dagger \in \mathcal{S}_\Upsilon \quad
\text{s.t.} \quad
\Omega(G+\operatorname{Im}\Upsilon)V - V(G-\operatorname{Im}\Upsilon)\Omega + \Omega(\operatorname{Re}\Upsilon)\Omega^\top = 0,\ \ \Upsilon\succeq 0 .
$$

**The Hurwitz condition must not be placed inside this program.** With $V\succ0$ fixed, "$A$ Hurwitz" is by §3 equivalent to the absence of a dark mode, i.e. to $(A,D)$ having no unobservable mode on the imaginary axis. That is a rank/detectability condition, not a linear matrix inequality; adding it would make the program nonconvex and destroy both the global solve and the dual certificate — the two properties the whole reformulation exists to obtain. Keep it out of the program and recover it afterwards:

- **Infeasible** $\Rightarrow$ **invalid**, with a dual certificate. This is sound *because* the program is a relaxation: the Hurwitz-free feasible set contains every physical solution, so if it is empty then no Hamiltonian and no dissipator with any number of channels realizes the target on this graph. This is the verdict you propagate downward without fear.
- **Feasible** $\Rightarrow$ **candidate only.** Feasibility proves a stationary point exists on the graph, not an attractive one — the same gap as 4A's zero residual. Hand the (convex) feasible set to §4C.

Note the asymmetry, and that it is the right way round. *Invalid* is certified by duality; *valid* is witnessed by exhibiting a concrete $(G,\Upsilon)$ and checking $\operatorname{Re}\lambda(A)<0$ directly, which is an exact finite computation. Soundness of pruning depends only on the invalid side, so the search's correctness guarantee is untouched by the fact that the valid side needs a search.

Channel-count control enters as a rank bound on $\Upsilon$: rank one is the single-channel case (your flat-spectrum branch), and the minimum feasible rank is the minimum channel count, matching the distinct-symplectic-eigenvalue counting you have already established. Rank constraints are nonconvex, so if you need a *specific* channel count you either fix the rank and use 4A, or add a low-rank surrogate; but for *pruning*, the rank-free SDP is what you want because its infeasibility certificate is the strongest possible sound rejection.

### 4C. The attractivity filter and a three-valued verdict

Both 4A and 4B end with a set $\mathcal{F}$ of stationary solutions — affine in 4A, convex in 4B — every member of which is at worst marginally stable by §3. The remaining task is to decide whether $\mathcal{F}$ contains a strictly Hurwitz member:

$$
\exists\,(G,\Upsilon)\in\mathcal{F} \ \text{ with }\ \max_i\operatorname{Re}\lambda_i\big(\Omega(G+\operatorname{Im}\Upsilon)\big) < 0\;?
$$

This is a small nonconvex problem, and it must not be allowed to fabricate an *invalid*. Dark-mode solutions are non-generic in $\mathcal{F}$, so in practice: sample $K$ points of $\mathcal{F}$ (random points of the affine set; random feasible points of the SDP) and accept the first Hurwitz one; if all $K$ are marginal, maximize the stability margin $-\max_i\operatorname{Re}\lambda_i$ over $\mathcal{F}$ directly. But a failed search proves nothing, so the oracle returns **three** verdicts, not two:

| verdict | evidence | propagates |
|---|---|---|
| **VALID** | explicit witness $(G,\Upsilon)$ with $A$ Hurwitz | upward, to all supergraphs |
| **INVALID** | nonzero residual across all admissible $\Upsilon$ / SDP dual certificate | downward, to all subgraphs |
| **UNDECIDED** | $\mathcal{F}\ne\emptyset$ but no Hurwitz member found | **nowhere** — reported, never propagated |

This is what preserves the soundness claim. A false *invalid* is impossible because that verdict is issued only from a certificate; the worst the attractivity search can do is downgrade a genuine valid to undecided, which is a *completeness* gap confined to a single graph rather than a soundness failure that would poison the whole down-set. Undecided graphs are reported to the user for inspection and, if desired, re-run with a larger $K$; they are never used to prune.

### Why this makes pruning a theorem, not a hope

Removing an edge shrinks $\mathcal{S}_G,\mathcal{S}_\Upsilon$ to a coordinate subspace. In 4A the least-squares residual over a smaller subspace can only be *larger*; in 4B a feasible set that is empty stays empty under further constraints. So "invalid propagates to every subgraph" is a property of the oracle itself, exactly matching the monotonicity of the filing system.

The attractivity filter does not break this, and it is worth recording why, since the filter is the one non-algebraic step. Write $\mathcal{F}(\mathcal{G})$ for the solution set on graph $\mathcal{G}$, and $\mathcal{H}(\mathcal{G})\subseteq\mathcal{F}(\mathcal{G})$ for its Hurwitz members. For $\mathcal{G}'\subseteq\mathcal{G}$ we have $\mathcal{F}(\mathcal{G}')\subseteq\mathcal{F}(\mathcal{G})$ (a smaller subspace), hence $\mathcal{H}(\mathcal{G}')\subseteq\mathcal{H}(\mathcal{G})$. So $\mathcal{H}(\mathcal{G})=\emptyset$ forces $\mathcal{H}(\mathcal{G}')=\emptyset$: *invalid* still propagates down. Conversely a witness on $\mathcal{G}$ embeds unchanged into any supergraph, so *valid* still propagates up. Both directions of the filing system survive the filter intact.

Contrast this with the gradient-descent oracle, where a subspace restriction could flip a spurious success into a spurious failure and break the match.

---

## 5. The discrete optimization (the search)

The outer loop is AUTOSCATTER's, with two modifications — down from three, because removing the seed is the point. Both concern the oracle and the integrity check, not the search structure.

**(i) Verdicts are certificates.** A graph is stamped *invalid* only when the oracle returns a nonzero least-squares residual across all admissible dissipators (4A into 4B) or an SDP infeasibility certificate (4B) — never because a search "gave up." A graph on which only the attractivity search failed is stamped *undecided* (§4C) and left out of both files. Since the reformulated oracle cannot manufacture a false *invalid*, this is the single property that makes cold-start discovery safe: the search can lose a graph only if that graph genuinely admits no Hamiltonian on it.

**(ii) No seeds — cold start from the fully connected root.** We insert nothing about Kronwald, Zippilli–Vitali, or Koga–Yamamoto in advance. The search begins from the fully connected graph and independently rediscovers every scheme reachable at finite parameters, because each such scheme produces its own zero-residual / feasible verdict with no prior knowledge required. This is deliberate. Seeding was originally a hedge against the old oracle's false negatives — redundant now by (i) — and a hedge that inserts the literature's answers and then reports them as output would quietly compromise the discovery claim. Seeding's *other* job, warm-starting the libraries, is recovered without any injected prior by memoizing the oracle's own past verdicts (below): a warm start over prior *computation*, never over prior *knowledge*.

**(iii) Bidirectional agreement is the correctness guarantee.** Run the search both top-down (prune edges from the root) and bottom-up (grow edges from the sparsest graph) on the same fixed mode count, for every $n$ you can afford, and assert the two produce identical valid/invalid/undecided partitions. Under a sound oracle they *must*: the verdict on each graph is a deterministic, direction-independent function of the graph, so the partition cannot depend on traversal order. (Make the attractivity search deterministic — fixed RNG seed per graph — so that *undecided* is reproducible too; otherwise compare only the valid/invalid files and report undecided counts separately.) This upgrades the check from the original "cheap empirical sanity test" to the primary integrity guarantee. A disagreement can no longer mean "one direction lost a scheme"; it can only mean an *implementation* bug — a mis-set tolerance, a wrong selection matrix $P$, a faulty SDP call, or a broken up-set/down-set propagation. It is a regression test *on* soundness rather than a hedge against unsoundness, and it delivers exactly what seeding was reaching for — a demonstration that nothing is lost — without telling the algorithm any answers.

Otherwise the loop is unchanged: increment auxiliary modes until the smallest fully connected graph is valid, then BFS through edge changes, checking each graph against the libraries first and calling the oracle only on genuinely new graphs, filing valid graphs with all extensions and invalid graphs with all subgraphs.

Two boundaries keep the "discover everything" claim precise. First, the cold interior search finds everything reachable *at finite parameters on the graph family*; limit-only schemes such as Kronwald live on the boundary of parameter space and are not — and never were going to be — found by the direct solve on the bare graph. Dropping their seed does not lose them, it relocates them to the §6 asymptotic module where they belong. Second, the price of the cold start is compute: re-deriving every graph is cheap at small $n$ and grows with $n$. The memoization cache mitigates this without touching discovery, since it caches only the oracle's own deterministic verdicts.

### Pseudocode

**State completion** `CompleteV(V_target, n_aux)` — §2.1:

```
assert purity(V_target) == 1                  # else: bilinear, out of scope (§2.1)
V ← V_target ⊕ (1/2)·I_{2·n_aux}              # V_TA = 0 by purity; V_AA = 1/2 by gauge + drain purity
return V
# fallback if drain purity is not argued: return the family
#   V(ν) = V_target ⊕ diag(ν_1,ν_1,…), ν_m ≥ 1/2
# and run the oracle once per ν on a grid; verdicts are pointwise in ν,
# so a grid-wide failure is UNDECIDED, never INVALID.
```

**Continuous oracle** `Oracle(𝒢, V)`:

```
build selection P for 𝒢's Hamiltonian support ∩ hardware constraints
# --- fast path: fixed single-/low-rank dissipator structure ---
fix Υ from 𝒢's dissipator support (rank r)
assemble 𝕃 = (V⊗Ω) + (Ω⊗V);  b = vec(B(Υ))
solve  g* = lstsq(𝕃P, b)  via SVD
if ‖𝕃P g* − b‖ ≤ tol:
    ℱ ← g* + null(𝕃P)                        # affine solution set
    (ok, G) ← AttractivityFilter(ℱ, Υ)        # §4C
    if ok:  return VALID, G, Υ
# residual > 0, or only marginal solutions: this decided Υ, not 𝒢 → go certify
# --- certifying path: decide across ALL admissible dissipators ---
# NOTE: the Hurwitz condition is deliberately NOT a constraint of this SDP.
#       It is not SDP-representable; including it would destroy convexity and
#       the dual certificate. It is applied afterwards, by the filter. (§4B)
ℱ ← {(G,Υ) : G∈S_G, Υ=Υ†∈S_Υ, Υ⪰0, (⋆⋆) holds}      # convex feasibility SDP
if ℱ = ∅:                  return INVALID, dual_certificate   # sound: relaxation is empty
(ok, G, Υ) ← AttractivityFilter(ℱ)                            # §4C
if ok:                     return VALID, G, Υ = C†C
else:                      return UNDECIDED                    # never propagated
```

**Attractivity filter** `AttractivityFilter(ℱ, ·)` — §4C:

```
for k in 1..K:                                  # deterministic RNG seeded per graph
    (G,Υ) ← sample(ℱ)
    if max Re λ( Ω(G + Im Υ) ) < 0:  return (true, G, Υ)
(G,Υ) ← argmax_{ℱ} [ −max Re λ( Ω(G + Im Υ) ) ]   # small nonconvex polish
return (max Re λ < 0, G, Υ)
# a failure here is NOT evidence of invalidity — caller must return UNDECIDED
```

**Discrete search** `Search(V_target)`:

```
Valid, Invalid, Undecided ← ∅, ∅, ∅        # no seeds — cold start
cache ← ∅                                  # memoized verdicts: compute reuse only, no priors
Oracle*(𝒢) ≡ cache[𝒢] if present, else cache[𝒢] ← Oracle(𝒢, CompleteV(V_target, n_aux))

# fix the mode count once, from the top-down root search
n_aux ← 0
while Oracle*(fully_connected(n_aux)).verdict ≠ VALID:  n_aux += 1
root ← fully_connected(n_aux)             # VALID by construction

# --- primary pass: prune from the root ---
(Valid_td, Invalid_td, Undec_td) ← BFS(direction = prune)

# --- correctness pass: grow from the sparsest graph on the SAME n_aux ---
if affordable(n_aux):
    (Valid_bu, Invalid_bu, Undec_bu) ← BFS(direction = grow)
    assert Valid_td == Valid_bu and Invalid_td == Invalid_bu
    assert Undec_td == Undec_bu            # requires deterministic filter (§5(iii))
    # disagreement → implementation bug (tolerance / P / SDP / propagation),
    #               NOT a lost scheme

report Undecided                           # for inspection / re-run at larger K
return irreducible elements of Valid_td
```

with the shared, direction-dual BFS body:

```
BFS(direction):
    Valid, Invalid, Undecided ← ∅, ∅, ∅
    frontier ← {root}                        if direction = prune
             ← {sparsest admissible graph}   if direction = grow   # typically INVALID

    while frontier nonempty:
        𝒢 ← pop(frontier);  skip if 𝒢 ∈ Valid or 𝒢 ∈ Invalid
        (v, G, Υ | cert) ← Oracle*(𝒢)
        if v = VALID:
            add 𝒢 and all extensions (supergraphs) to Valid
            if some couplings ≈ 0:  add the reduced graph + its extensions
            if direction = prune:  enqueue edge-removal neighbours   # keep descending
            # grow: stop — the whole up-set is already valid
        elif v = INVALID:                                            # + certificate
            add 𝒢 and all subgraphs to Invalid
            if direction = grow:   enqueue edge-addition neighbours  # keep ascending
            # prune: stop — the whole down-set is already invalid
        else:                                                        # UNDECIDED
            add 𝒢 to Undecided
            enqueue neighbours in the current direction              # propagate NOTHING;
            # the up-/down-set is unresolved, so keep exploring past it
    return (Valid, Invalid, Undecided)
```

The two passes are exact duals: prune descends through valid territory until it meets the invalid boundary; grow ascends through invalid territory until it meets the valid boundary. Both converge on the same object — the set of irreducible (minimal) valid graphs — which is why their agreement is a genuine check and not a tautology of shared code.

---

## 6. Limit-only targets (Kronwald and friends): a separate module

It is worth being precise about when a target genuinely has no finite solution, because it is subtler than "Kronwald is a limit."

**A pure target is not limit-only by itself.** By the universality theorem, a pure Gaussian target is the exact, attractive steady state of an $(N{+}1)$-mode system with one auxiliary, one vacuum-loss channel, and an active Hamiltonian — and the §4 oracle finds exactly that at finite parameters, provided you (i) idealize the intrinsic target-mode loss to zero and (ii) let the Hamiltonian search range freely. Setting the mechanical decay $\Gamma_M=0$ is legitimate and is the clean way to sit at the ideal point directly. It is also what makes §2.1's state completion exact, so the two idealizations are the same idealization: at $\Gamma_M=0$ with a pure target we have $V_{TA}=0$ and $V=V_{\text{target}}\oplus\tfrac12\mathbb{1}$, and $(\star\star)$ is genuinely linear. The auxiliary rate $\kappa$ may then be set to any convenient finite value — not because $V$ ignores absolute rates (it does not; see §1(b)), but because the solve returns $G$ *together with* $\kappa$ up to the joint $(G,\Upsilon)\mapsto(sG,s\Upsilon)$ gauge, so a different $\kappa$ returns a proportionally rescaled $G$ describing the same physics. The solve yields the active witness with machine-precision residual and a Hurwitz $A$. (Numerically verified for the single-mode squeezed target: residual $\sim 10^{-16}$, $\max\operatorname{Re}\lambda(A)<0$, forward steady state matching the target to $\sim 10^{-14}$.)

**"Limit-only" appears only when you force extra structure** — and it is always a property of the constraint set, never of the target in the abstract. Two triggers:

- **A forced second channel.** If the intrinsic mechanical loss must be modelled as its own vacuum-loss channel ($\Gamma_M>0$), its dark state (mechanical vacuum) conflicts with the squeezed target. Two things then break at once, and they should be kept distinct. *First*, the state completion of §2.1 fails: the target block is no longer exactly pure at steady state, so $V_{TA}\ne0$ and $(\star\star)$ is bilinear rather than linear — the oracle's exactness is gone before any question of existence is asked. *Second*, even granting a completed $V$, no Hamiltonian makes the pure state exactly stationary: the least-squares residual is nonzero and vanishes only as the relative rate $\Gamma_M/\kappa\to 0$ (equivalently $\mathcal{C}\to\infty$). This is the origin of Kronwald's residual thermal occupation, and it is consistent with the single-channel/flat-spectrum theorem — a *forced* second channel with a conflicting dark state pushes the steady state off the pure (flat-spectrum) point until its relative rate vanishes.
- **A constrained Hamiltonian.** If $G$ is restricted to what the physical bichromatic drive produces after adiabatic elimination, the exact witness lies outside the admissible subspace, and only the boundary is reachable within it.

So on the unconstrained graph with $\Gamma_M=0$ the oracle returns an exact finite solution — one the physical Kronwald scheme only *approximates*. Both oracle versions return "unreachable" only when one of the two triggers above is imposed; that is AUTOSCATTER's own second acknowledged failure mode ("realizable only in a limiting case"), a target driven to the boundary of parameter space *by its constraints*.

When you are committed to the constrained/physical model, handle it as a distinct module rather than by loosening tolerances (a loose tolerance accepts spurious near-misses and still rejects the genuine limit point — no threshold sorts this out, because a limit point is distinguished by the *trend* of the residual along a ray, not its value at one point):

- **Idealize, then perturb (preferred when you only need existence).** Set $\Gamma_M=0$, solve exactly for the active witness, and re-introduce the physical $\Gamma_M>0$ as a perturbation whose fidelity cost scales with $\Gamma_M/\kappa$. This needs no adiabatic elimination and no sweep, and it is the route that keeps §2.1's linearization intact — the perturbation is applied to the *answer*, never inside the solve.
- **Pre-eliminate by hand.** Perform the adiabatic elimination that produces the effective reduced model — the bichromatic-drive squeezed-reservoir block — and run the oracle on the *reduced* graph. Consistent with your conclusion that these blocks are human pre-specified rather than auto-discovered (elimination removes a mode and redefines $G,\Upsilon$, which is outside the search's edge-add/remove move set).
- **Asymptotic sweep.** Parametrize a ray toward the boundary (the compatible path: $\Gamma_M\to 0$ at fixed $G<\kappa$) and verify the residual $\to 0$ along it, extracting the convergence rate as the figure of merit. The rate is *scheme-dependent* — Kronwald's own squeezed-variance floor scales as $\sim\sqrt{(1+2n_{th})/\mathcal{C}}$, not the $\sim r/\mathcal{C}$ of the ZV cluster-state analysis — so extract it per scheme rather than quoting a universal form. Note the stiffness caveat: as $\Gamma_M\to0$ the Liouvillian gap closes while the cavity rate stays large, so $A$ becomes ill-conditioned exactly where you need the answer; this is why idealize-then-perturb is cleaner when the algebra allows it.

**Mixed targets.** For completeness, the other way out of §2.1's scope is a target that is mixed to begin with. It is not a limiting case and no sweep recovers it: $V_{TA}\ne0$ identically, so the method's linearity fails structurally. Either purify the target (dilate to a larger pure target on signal + ancilla and stabilize that) or accept a bilinear solve with the corresponding loss of certificates.

---

## 7. Honest bookkeeping for your supervisor

- **Standard / borrowed.** The outer search architecture is AUTOSCATTER's, used as intended. The moment equations $A=\Omega(G+\operatorname{Im}\Upsilon)$, $D=\Omega(\operatorname{Re}\Upsilon)\Omega^\top$ are standard Gaussian-Lindblad results. The stability-from-positive-definite-Lyapunov argument in §3 is textbook (Horn–Johnson) and is the quadrature-basis version of the Zippilli–Vitali uniqueness proof. That a pure marginal is uncorrelated with its complement (§2.1, Step 1) is elementary quantum information; Williamson's theorem supplies the gauge reduction in Step 2. Linearity of the stationarity condition in $G$ for fixed dissipators is implicit in the Koga–Yamamoto construction.
- **Novel here.** The reframing that makes the *oracle* exact — fixing $V$ to linearize $(\star\star)$ and thereby replace gradient descent with a global linear solve (4A) or a certifying SDP (4B) — is the contribution. Three supporting pieces make it stand up and are part of the same contribution: the **state-completion argument** (§2.1) that recovers the full $V$ from a target specified only on the signal modes, without which the linearization is not actually available; the **separation of the Hurwitz condition from the convex program** (§4B–4C), which is what preserves the dual certificate; and the **three-valued verdict** (§4C), which confines the one remaining nonconvex step to a completeness gap on a single graph instead of letting it manufacture false invalids. Together they convert AUTOSCATTER's best-effort enumeration (false negatives mitigated by reruns) into a search whose pruning is *sound by certificate*. The claim that this yields provable soundness, and the SDP-with-dual-certificate as the pruning oracle, are not in any of the source papers and should be presented as your extension.
- **A discovery claim, not a confirmation.** Because the oracle is sound (i), the search runs cold — no known scheme is seeded (§5(ii)) — so recovering Zippilli–Vitali and Koga–Yamamoto is an *output* of the search rather than an input to it. This is a strictly stronger validation than seeding-and-confirming, and it depends on the soundness result above: without a certifying oracle, a cold start could silently lose a genuine scheme, and seeding would be the only defence. The bidirectional-agreement test (§5(iii)) is the empirical demonstration that it does not — with a sound oracle the top-down and bottom-up partitions must coincide, so their agreement checks the *implementation*, not the search's completeness.
- **Scope, stated up front.** Two caveats belong next to the claim rather than in a footnote. Cold discovery covers everything reachable at finite parameters; limit-only schemes (Kronwald) remain a separately-handled §6 module by their nature, not by omission. And the exactness of the oracle requires a **pure target with $\Gamma_{\text{signal}}=0$** (§2.1), which is what makes the completed $V$ known and $(\star\star)$ linear; mixed targets and finite intrinsic loss are bilinear and fall back to §6's idealize-then-perturb. Both restrictions are properties of the *problem class*, not of the algorithm's implementation, and both are exactly the regime in which the flat-spectrum and universality results are already stated — so the method is exact precisely where the theory it is meant to explore lives.

The one-line summary: keep their search, replace their guesser with a solver. The state target is what makes the solver possible, and it is the reason your adaptation can be made strictly more rigorous than the original rather than a straight port of it.
