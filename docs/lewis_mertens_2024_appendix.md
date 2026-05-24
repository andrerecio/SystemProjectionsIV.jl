# Dynamic Identification Using System Projections and Instrumental Variables — ONLINE APPENDIX

**Daniel J. Lewis · Karel Mertens**

---

## Contents

- **I. Testing the Null Hypothesis of Weak Instruments**
  - I.1 Weak IV Representation of the SP-IV Estimator
  - I.2 Definition of Weak Instruments
  - I.3 Characterizing the Boundary of the Weak Instrument Set
  - I.4 Null Hypothesis
  - I.5 Test Statistic and Critical Values
- **II. Additional Simulation Results**
  - II.1 IRF Estimates in the Simulations
  - II.2 Simulation Results Using Three Instruments ($N_z = 3$)
  - II.3 Simulation Results for Generalized SP-IV estimators
  - II.4 Simulation Results for Alternative 2SLS Specifications
  - II.5 Measuring Effective Instrument Strength

---

## I. Testing the Null Hypothesis of Weak Instruments

This section describes the weak instruments test in the SP-IV model discussed in Section 2.2 of the main text. The test nests the popular bias-based test of Stock and Yogo (2005) when $H = 1$. The development is analogous to the weak instruments test in Lewis and Mertens (2022).

**Notation.** $\|U\|_2$ is the spectral norm of $U$; $\mathbb{P}_n$ is the set of positive definite $n \times n$ matrices; $\mathbb{O}_{n \times m}$ is the set of $n \times m$ orthogonal real matrices; $K_{n,m}$ is the $n \times m$ commutation matrix; and $R_{n,m} = I_n \otimes \text{vec}(I_m)$. Note $R'_{n,m} R_{n,m} = m I_N$.

### I.1 Weak IV Representation of the SP-IV Estimator

The SP-IV estimator using the general restriction matrix is

$$\hat{\beta} = \left(R'_{K,H}(Y^\perp_H P_{Z^\perp} Y^{\perp\prime}_H \otimes I_H) R_{K,H}\right)^{-1} R'_{K,H} \text{vec}(y^\perp_H P_{Z^\perp} Y^{\perp\prime}_H), \tag{I.1}$$

where $P_{Z^\perp} = Z^{\perp\prime}(Z^\perp Z^{\perp\prime})^{-1} Z^\perp$.

**Assumption 4 (weak instruments).** $\Theta_Y = C/\sqrt{T}$ where $C \in \mathbb{R}^{HK \times N_z}$ is a fixed matrix and $R_{K,H}(CC' \otimes I_H) R_{K,H}$ is of full rank.

**Assumption 5.** The following limits hold as $T \to \infty$:
- $u^\perp_H u^{\perp\prime}_H / T \xrightarrow{p} \Sigma_{u^\perp_H} \in \mathbb{P}_H$ (5.a)
- $u^\perp_H v^{\perp\prime}_H / T \xrightarrow{p} \Sigma_{u^\perp_H v^\perp_H} \in \mathbb{R}^{H \times HK}$
- $v^\perp_H v^{\perp\prime}_H / T \xrightarrow{p} \Sigma_{v^\perp_H} \in \mathbb{P}_{HK}$
- $T^{-1/2} [\text{vec}((Z^\perp Z^{\perp\prime})^{-1/2} Z^\perp w^{\perp\prime}_H)', \text{vec}((Z^\perp Z^{\perp\prime})^{-1/2} Z^\perp v^{\perp\prime}_H)']' \xrightarrow{d} N(0, W \otimes I_{N_z})$ (5.b)
- $\hat{W} \xrightarrow{p} W$ (5.c)

where $W = \begin{bmatrix} W_1 & W_{12} \\ W'_{12} & W_2 \end{bmatrix} \in \mathbb{P}_{(K+1)H}$.

Here $w^\perp_H = y^\perp_H - (\beta' \otimes I_H) \Theta_Y Q^{-1/2} Z^\perp$ are reduced-form errors with covariance $W_1$; $v^\perp_H = Y^\perp_H - \Theta_Y Q^{-1/2} Z^\perp$ are first-stage errors with covariance $W_2$.

Define the random variables $\eta_1$ ($H \times N_z$) and $\eta_2$ ($HK \times N_z$):

$$\begin{bmatrix} \text{vec}(\eta_1) \\ \text{vec}(\eta_2) \end{bmatrix} \sim N\left(\begin{bmatrix} 0_{HN_z} \\ \text{vec}(C) \end{bmatrix}, S \otimes I_{N_z}\right) \tag{I.3}$$

where $S \in \mathbb{P}_{(K+1)H}$ is partitioned analogously to $W$:
$$S_1 = W_1 + (\beta' \otimes I_H) W_2 (\beta \otimes I_H) - (\beta' \otimes I_H) W'_{12} - W_{12}(\beta \otimes I_H)$$
$$S_{12} = W_{12} - (\beta' \otimes I_H) W_2, \quad S_2 = W_2 \tag{I.4}$$

**Proposition 6.** Under Assumptions 4 and 5, $s_{ZY} \xrightarrow{d} \eta_2$ and $s_{Zy} \xrightarrow{d} (\beta' \otimes I_H) \eta_2 + \eta_1$, and thus

$$\hat{\beta} - \beta \xrightarrow{d} \beta^* = \left(R'_{K,H}(\eta_2 \eta'_2 \otimes I_H) R_{K,H}\right)^{-1} R'_{K,H} \text{vec}(\eta_1 \eta'_2).$$

Since $\beta^*$ converges to a quotient of quadratic forms in normal random variables, $\hat{\beta}$ is **not a consistent estimator** of $\beta$.

**Definition 1 (Concentration matrix).**
$$\Lambda = \frac{1}{N_z} \Phi^{-1/2} R_{K,H}(CC' \otimes I_H) R_{K,H} \Phi^{-1/2}$$
where $\Phi = R'_{K,H}(S_2 \otimes I_H) R_{K,H}$.

### I.2 Definition of Weak Instruments

**Definition 2 (Bias criterion).**
$$B = \text{Tr}(S_1)^{-1/2} \|\mathbb{E}[\beta^*]' \Phi^{1/2}\|_2$$

Following Stock and Yogo (2005), the $\ell_2$-norm aggregates the $K$ elements of the bias through a quadratic loss function, with $\Phi$ effectively standardizing the regressors. The bias criterion reaches a maximum of unity in a worst case (perfect linear dependence of errors on second-stage regressors, completely uninformative instruments).

**Definition 3 (Weak instrument set).**
$$B_\tau(W) = \{C \in \mathbb{R}^{N \times K}, \beta \in \mathbb{R}^N : B \geq \tau\} \tag{I.6}$$

The weak instrument set contains values of $\beta$ and first-stage parameters $C$ such that the bias $B$ exceeds tolerance level $\tau$.

### I.3 Characterizing the Boundary of the Weak Instrument Set

The bias criterion can be decomposed as $B = \|h \rho\|_2$, where
$$h = H \mathbb{E}\left[\left(R'_{K,H}(S(l + \psi)(l + \psi)' S' \otimes I_H) R_{K,H}\right)^{-1} R'_{K,H}\left(S(l + \psi) \psi' S^{-1} \otimes I_H\right)\right]$$
$$\rho = (\Phi^{-1/2} \otimes I_{H^2}) \text{vec}(S_{12}) / \sqrt{\text{Tr}(S_1)}$$

with $l = S_2^{-1/2} C$, $\psi = S_2^{-1/2}(\eta_2 - C)$, $\text{vec}(\psi) \sim N(0, I_{KHN_z})$, $S = ((\Phi/H)^{-1/2} \otimes I_H) S_2^{1/2}$.

Following Montiel-Olea and Pflueger (2013), we adopt a **Nagar (1959) approximation** to $h$ around $\psi = 0$, denoted $h_n$. The Nagar bias is $B_n = \|h_n \rho\|_2$. We base our test on the bound

$$B_n \leq \lambda_{\min}^{-1} \mathcal{B}(W), \tag{I.8}$$

where $\lambda_{\min} = \text{mineval}\{\Lambda\}$.

### I.4 Null Hypothesis

The test of the null hypothesis of weak instruments is based on testing whether the minimum eigenvalue of $\Lambda$ is less than or equal to a threshold $\lambda^*_{\min}(\tau)$:

$$H_0: \lambda_{\min} \in \mathcal{H}(W) \quad \text{vs.} \quad H_1: \lambda_{\min} \notin \mathcal{H}(W), \tag{I.11}$$

where $\mathcal{H}(W) = \{\lambda_{\min} \in \mathbb{R}_+ : \lambda_{\min} \leq \lambda^*_{\min}(\tau)\}$ and $\lambda^*_{\min}(\tau) = \mathcal{B}(W) / \tau$.

### I.5 Test Statistic and Critical Values

**Proposition 7.** Define the test statistic
$$g = N_z^{-1} \text{mineval}\{\hat{\Phi}^{-1/2}(Y^\perp_H P_{Z^\perp} Y^{\perp\prime}_H) \hat{\Phi}^{-1/2}\},$$
where $\hat{\Phi} = R'_{K,H}(\hat{W}_2 \otimes I_H) R_{K,H}$.

Under Assumptions 4 and 5,
$$g \xrightarrow{d} \text{mineval}\{R'_{K,H}(\zeta \otimes I_K) R_{K,H} / (H N_z)\},$$
where the $KH \times KH$ random matrix $\zeta = S(l + \psi)(l + \psi)' S'$ has a non-central Wishart distribution: $\zeta \sim \mathcal{W}(N_z, \Sigma, \Omega)$, with $N_z$ degrees of freedom, covariance $\Sigma = SS' \in \mathbb{P}_{KH}$, and noncentrality matrix $\Omega = \Sigma^{-1} S l l' S'$.

Critical values are obtained from a **bounding limiting distribution**, following Stock and Yogo (2005) and Lewis and Mertens (2022). We use the class of approximating distributions proposed by **Imhof (1961)**, matching the first three cumulants. We select the Imhof distribution with the largest critical value at significance level $\alpha$ subject to the constraints that:
- The first cumulant $\kappa_1 = H N_z (1 + \lambda_{\min})$ matches the target distribution;
- The second and third cumulants respect the analytical upper bounds.

The resulting critical value is **conservative** relative to the unknown critical value from the true limiting distribution.

---

## II. Additional Simulation Results

### II.1 IRF Estimates in the Simulations

Figures II.1 (small sample, $T = 250$) and II.2 ($T = 5000$) show the true model impulse responses to a one s.t.d. contractionary monetary policy shock, together with mean IRF estimates and 2.5%/97.5% percentiles across 5000 simulations from the Smets and Wouters (2007) model.

**Three columns:** Distributed Lag, LP with controls, VAR.
**Top row:** IRFs of inflation. **Bottom row:** IRFs of the output gap (real marginal cost).

**Key findings:**
- **Small sample ($T = 250$):** The DL estimates display smaller small-sample bias than LP and VAR but have a wider 95% range at shorter horizons. Consistent with Li et al. (2021), the VAR estimates have a narrower range than LP with controls, particularly at longer horizons.
- **Large sample ($T = 5000$):** The DL and LP estimates show essentially no bias. Consistent with Montiel Olea and Plagborg-Møller (2021), the VAR estimates show no bias for horizons up to the lag length of the VAR (four). The SW model does not have a finite-order VAR representation in $X_t$, so the finite-order VAR restrictions result in bias at horizons beyond the lag length.

### II.2 Simulation Results Using Three Instruments ($N_z = 3$)

In addition to the monetary policy shock, this section uses the **government spending shock** and the **risk premium shock** from the SW model.

**Table II.1: Mean and Variance of parameter estimates, $N_z = 3$**

*Panel a. Mean Parameter Estimates*

| | T=250 | | | T=500 | | | T=5000 | | |
|---|---|---|---|---|---|---|---|---|---|
| Estimator | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
| True Value | 0.15 | 0.85 | 0.05 | 0.15 | 0.85 | 0.05 | 0.15 | 0.85 | 0.05 |
| OLS | 0.47 | 0.47 | 0.00 | 0.48 | 0.48 | 0.00 | 0.48 | 0.48 | 0.00 |
| **H = 8** | | | | | | | | | |
| 2SLS | 0.40 | 0.56 | 0.00 | 0.36 | 0.63 | 0.00 | 0.23 | 0.82 | 0.02 |
| SP-IV LP | 0.39 | 0.56 | 0.00 | 0.36 | 0.63 | 0.00 | 0.22 | 0.82 | 0.02 |
| SP-IV LP-C | 0.40 | 0.55 | 0.02 | 0.36 | 0.63 | 0.03 | 0.20 | 0.81 | 0.04 |
| SP-IV VAR | 0.34 | 0.69 | 0.01 | 0.29 | 0.75 | 0.02 | 0.20 | 0.83 | 0.04 |
| **H = 20** | | | | | | | | | |
| 2SLS | 0.45 | 0.51 | 0.00 | 0.43 | 0.55 | 0.00 | 0.28 | 0.76 | 0.01 |
| SP-IV LP | 0.44 | 0.51 | 0.00 | 0.42 | 0.56 | 0.00 | 0.28 | 0.76 | 0.01 |
| SP-IV LP-C | 0.44 | 0.51 | 0.01 | 0.43 | 0.56 | 0.01 | 0.27 | 0.76 | 0.02 |
| SP-IV VAR | 0.35 | 0.69 | 0.01 | 0.31 | 0.74 | 0.01 | 0.23 | 0.82 | 0.02 |

*Panel b. Standard Deviation of Parameter Estimates*

| | T=250 | | | T=500 | | | T=5000 | | |
|---|---|---|---|---|---|---|---|---|---|
| Estimator | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
| **H = 8** | | | | | | | | | |
| 2SLS | 0.09 | 0.10 | 0.03 | 0.08 | 0.09 | 0.03 | 0.06 | 0.05 | 0.02 |
| SP-IV LP | 0.09 | 0.10 | 0.04 | 0.09 | 0.09 | 0.03 | 0.06 | 0.05 | 0.02 |
| SP-IV LP-C | 0.09 | 0.10 | 0.06 | 0.09 | 0.09 | 0.05 | 0.06 | 0.05 | 0.03 |
| SP-IV VAR | 0.11 | 0.13 | 0.05 | 0.11 | 0.11 | 0.05 | 0.06 | 0.05 | 0.03 |
| **H = 20** | | | | | | | | | |
| 2SLS | 0.04 | 0.04 | 0.01 | 0.04 | 0.04 | 0.01 | 0.04 | 0.04 | 0.01 |
| SP-IV LP | 0.05 | 0.05 | 0.02 | 0.04 | 0.04 | 0.01 | 0.04 | 0.04 | 0.01 |
| SP-IV LP-C | 0.04 | 0.05 | 0.02 | 0.04 | 0.04 | 0.02 | 0.04 | 0.04 | 0.01 |
| SP-IV VAR | 0.09 | 0.11 | 0.02 | 0.09 | 0.10 | 0.02 | 0.05 | 0.04 | 0.02 |

**Key takeaways:**
- Relative performance of estimators is qualitatively the same as with single instrument.
- Bias improvements from IV vs OLS are **smaller** with three instruments.
- Using additional instruments **lowers the variance** of all estimators.
- → The choice of the number of instruments involves a **bias-variance trade-off**.

**Table II.2: Empirical size of nominal 5% tests, $N_z = 3$**

| | H=8 | | | H=20 | | |
|---|---|---|---|---|---|---|
| Test | T=250 | T=500 | T=5000 | T=250 | T=500 | T=5000 |
| Wald 2SLS | 83.1 | 79.2 | 58.9 | 100.0 | 99.9 | 94.3 |
| Wald SP-IV LP | 84.3 | 80.4 | 60.4 | 100.0 | 99.9 | 93.8 |
| Wald SP-IV LP-C | 75.8 | 62.4 | 22.7 | 100.0 | 99.8 | 83.0 |
| Wald SP-IV VAR | 39.2 | 28.3 | 13.3 | 86.7 | 76.7 | 54.1 |
| AR SE-IV | 16.9 | 11.4 | 4.3 | 60.0 | 36.3 | 6.4 |
| AR SP-IV LP | 7.0 | 5.7 | 4.7 | 14.3 | 8.0 | 5.0 |
| AR SP-IV LP-C | 7.0 | 5.6 | 4.5 | 16.9 | 9.2 | 5.1 |
| **AR SP-IV VAR** | **3.9** | **5.1** | **4.8** | **6.5** | **5.2** | **4.6** |
| KLM SE-IV | 2.7 | 4.3 | 4.3 | 0.0 | 7.2 | 5.0 |
| KLM SP-IV LP | 5.7 | 5.2 | 5.3 | 7.6 | 6.5 | 5.3 |
| KLM SP-IV LP-C | 7.3 | 5.5 | 5.6 | 11.4 | 7.6 | 6.1 |
| KLM SP-IV VAR | 6.9 | 6.6 | 4.9 | 11.7 | 8.5 | 5.5 |

The robust SP-IV tests appear less affected by many-moment distortions than the robust tests for the single-equation specification with DL instruments.

### II.3 Simulation Results for Generalized SP-IV estimators

The (feasible) generalized SP-IV (GSP-IV) estimators are based on a 2-step procedure: first estimate baseline SP-IV and $\hat{\Sigma}^\perp_u$ using (19), then use this to obtain GSP-IV as in (B.1). GSP-IV is also the feasible 2-step efficient GMM estimator.

**Table II.3: Standard deviation of parameter estimates, GSP-IV**

| | T=250 | | | T=500 | | | T=5000 | | |
|---|---|---|---|---|---|---|---|---|---|
| | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
| **H = 8, $N_z$ = 1** | | | | | | | | | |
| GSP-IV LP | 0.33 | 0.46 | 0.24 | 0.27 | 0.40 | 0.24 | 0.12 | 0.08 | 0.09 |
| GSP-IV LP-C | 0.36 | 0.33 | 0.31 | 0.31 | 0.22 | 0.28 | 0.12 | 0.06 | 0.08 |
| GSP-IV VAR | 0.36 | 0.41 | 0.34 | 0.33 | 0.27 | 0.32 | 0.13 | 0.06 | 0.09 |
| **H = 20, $N_z$ = 1** | | | | | | | | | |
| GSP-IV LP | 0.15 | 0.18 | 0.07 | 0.12 | 0.14 | 0.06 | 0.07 | 0.05 | 0.03 |
| GSP-IV LP-C | 0.10 | 0.11 | 0.06 | 0.09 | 0.09 | 0.05 | 0.08 | 0.05 | 0.03 |
| GSP-IV VAR | 0.24 | 0.28 | 0.14 | 0.21 | 0.19 | 0.13 | 0.12 | 0.06 | 0.06 |
| **H = 8, $N_z$ = 3** | | | | | | | | | |
| GSP-IV LP | 0.11 | 0.12 | 0.04 | 0.09 | 0.10 | 0.03 | 0.06 | 0.05 | 0.02 |
| GSP-IV LP-C | 0.10 | 0.10 | 0.05 | 0.09 | 0.08 | 0.05 | 0.06 | 0.05 | 0.03 |
| GSP-IV VAR | 0.11 | 0.12 | 0.05 | 0.10 | 0.11 | 0.05 | 0.06 | 0.05 | 0.03 |
| **H = 20, $N_z$ = 3** | | | | | | | | | |
| GSP-IV LP | 0.04 | 0.04 | 0.01 | 0.03 | 0.03 | 0.01 | 0.04 | 0.04 | 0.01 |
| GSP-IV LP-C | 0.03 | 0.03 | 0.01 | 0.03 | 0.03 | 0.01 | 0.04 | 0.04 | 0.01 |
| GSP-IV VAR | 0.07 | 0.09 | 0.02 | 0.07 | 0.09 | 0.02 | 0.05 | 0.04 | 0.02 |

**Conclusion:** Although GSP-IV is theoretically asymptotically more efficient, the feasible versions do **not generally improve performance in practice**, at least in realistic sample sizes. For $N_z = 1$, most GSP-IV variances slightly exceed those of their SP-IV counterparts. Likely cause: estimation error in the $H \times H$ weighting matrix, which itself depends on the only weakly identified estimate $\hat{\beta}$. → Little motivation to prefer GSP-IV over SP-IV in practice.

### II.4 Simulation Results for Alternative 2SLS Specifications

Three alternative versions of 2SLS with controls:
- **2SLS-C**: adds $X_{t-1}$ as controls to both stages of 2SLS with DL instruments.
- **2SLS-CL**: adds $X_{t-H}$ as controls to both stages of 2SLS with DL instruments.
- **2SLS-CZ**: does not add controls to 2SLS, but uses a DL of $z^\perp_t$ (the residual in the regression of $z_t$ on $X_{t-1}$) as instruments.

**Table II.4: Lag Endogenous Instrument, $T = 5000$**

| Estimator | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
|---|---|---|---|
| True Value | 0.15 | 0.85 | 0.05 |
| OLS | 0.48 | 0.48 | 0.00 |
| **H = 8** | | | |
| 2SLS | 0.27 | 0.58 | −0.09 |
| 2SLS-C | 0.19 | 0.87 | −0.06 |
| 2SLS-CL | 0.20 | 0.83 | 0.01 |
| 2SLS-CZ | 0.16 | 0.84 | 0.05 |
| SP-IV LP-C | 0.16 | 0.84 | 0.05 |
| SP-IV VAR | 0.12 | 0.83 | 0.09 |
| **H = 20** | | | |
| 2SLS | 0.24 | 0.76 | −0.02 |
| 2SLS-C | 0.21 | 0.84 | −0.06 |
| 2SLS-CL | 0.23 | 0.81 | 0.01 |
| 2SLS-CZ | 0.23 | 0.81 | 0.02 |
| SP-IV LP-C | 0.23 | 0.81 | 0.02 |
| SP-IV VAR | 0.17 | 0.83 | 0.05 |

**Key findings:**
- **2SLS-C** reduces overall bias relative to 2SLS, but still produces a severely biased estimate for $\lambda$. Problem: including $X_{t-1}$ as controls weakens identifying information from lags of instruments.
- **2SLS-CL** generates bias improvements but not to the same extent as SP-IV LP-C and VAR.
- **2SLS-CZ** is the only version of 2SLS that successfully removes the lag endogeneity bias. In large samples, it generates the same parameter estimates as SP-IV LP-C.

**Table II.5: Fully Exogenous Instruments**

| | T=250 | | | T=500 | | | T=5000 | | |
|---|---|---|---|---|---|---|---|---|---|
| Estimator | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
| True Value | 0.15 | 0.85 | 0.05 | 0.15 | 0.85 | 0.05 | 0.15 | 0.85 | 0.05 |
| **H = 8** | | | | | | | | | |
| 2SLS | 0.27 | 0.51 | 0.01 | 0.24 | 0.61 | 0.00 | 0.17 | 0.83 | 0.04 |
| 2SLS-C | 0.31 | 0.69 | −0.05 | 0.27 | 0.76 | −0.04 | 0.18 | 0.89 | −0.11 |
| 2SLS-CL | 0.30 | 0.58 | 0.02 | 0.26 | 0.69 | 0.03 | 0.16 | 0.83 | 0.05 |
| 2SLS-CZ | 0.27 | 0.52 | 0.01 | 0.24 | 0.62 | 0.01 | 0.16 | 0.84 | 0.05 |
| SP-IV LP-C | 0.29 | 0.64 | 0.05 | 0.25 | 0.74 | 0.04 | 0.16 | 0.84 | 0.05 |

When $z_t$ is fully exogenous, 2SLS-CZ offers no improvement over 2SLS in small samples — unlike SP-IV with controls, there is **no improvement in the effective instrument strength**.

### II.5 Measuring Effective Instrument Strength

We approximate the concentration matrix $\Lambda$ for each estimator by setting $T = 1{,}000{,}000$ in the SW model. The minimum eigenvalue of $\Lambda$ is a sufficient statistic for the worst-case bias. Reported is this minimum eigenvalue relative to the $\tau = 0.10$, $\alpha = 0.05$ critical values, specific to each estimator/specification.

**Table II.6: Measures of instrument strength, $T = 500$**

| Estimator | $N_z = 1$ | $N_z = 3$ |
|---|---|---|
| **H = 8** | | |
| 2SLS | 0.02 | 0.02 |
| SP-IV LP | 0.02 | 0.02 |
| **SP-IV LP-C** | **0.17** | **0.17** |
| SP-IV VAR | 0.12 | 0.13 |
| **H = 20** | | |
| 2SLS | 0.02 | 0.01 |
| SP-IV LP | 0.02 | 0.01 |
| **SP-IV LP-C** | **0.08** | **0.05** |
| SP-IV VAR | 0.03 | 0.06 |

**Key takeaways:**
- SP-IV LP-C and SP-IV VAR have **effectively stronger instruments** due to the inclusion of controls.
- 2SLS and SP-IV LP (without controls) are essentially identical.
- SP-IV LP-C makes marginally better use of the instruments than SP-IV VAR.
- $H = 20$ delivers weaker identification than $H = 8$, due to flattening of IRFs at longer horizons.
- Effect of additional instruments depends on the estimator: additional shocks provide more identifying variation, but the critical values to which measures are benchmarked also increase with $N_z$.

The reported measures of instrument strength are overall relatively small, yet estimator performance is not considerably worse — because the tests are based on **worst-case bias**, and there is no reason to believe the model parameters are in the neighborhood of this worst case.

---

## Online Appendix References

- Imhof, J.P. (1961). Computing the Distribution of Quadratic Forms in Normal Variables. *Biometrika*, 48(3-4), 419–426.
- Lewis, D.J., & Mertens, K. (2022). A Robust Test for Weak Instruments with Multiple Endogenous Regressors. FRB New York Staff Reports 1020.
- Li, D., Plagborg-Møller, M., & Wolf, C.K. (2021). Local Projections vs. VARs: Lessons From Thousands of DGPs. arXiv 2104.00655.
- Montiel-Olea, J.L., & Pflueger, C. (2013). A Robust Test for Weak Instruments. *JBES*, 31(3), 358–369.
- Montiel Olea, J.L., & Plagborg-Møller, M. (2021). Local Projection Inference is Simpler and More Robust Than You Think. *Econometrica*, 89(4), 1789–1823.
- Muirhead, R.J. (1982). *Aspects of Multivariate Statistical Theory*. John Wiley & Sons.
- Nagar, A.L. (1959). The Bias and Moment Matrix of the General k-Class Estimators. *Econometrica*, 27(4), 575–595.
- Smets, F., & Wouters, R. (2007). Shocks and Frictions in US Business Cycles. *American Economic Review*, 97(3), 586–606.
- Staiger, D., & Stock, J.H. (1997). Instrumental Variables Regression with Weak Instruments. *Econometrica*, 65(3), 557–586.
- Stock, J., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In Andrews (Ed.), *Identification and Inference for Econometric Models*. Cambridge UP.
- Stock, J.H., & Watson, M.W. (2012). Disentangling the Channels of the 2007-2009 Recession. *Brookings Papers*, Spring 2012, 81–135.
- Sun, Y. (2014). Let's fix it: Fixed-b asymptotics versus small-b asymptotics. *Journal of Econometrics*, 178(P3), 659–677.
