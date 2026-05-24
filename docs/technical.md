# SP-IV: Mathematical Specification

This document provides the mathematical specification of the **System Projections on Instrumental Variables (SP-IV)** estimator of Lewis and Mertens (2024), as implemented in this package.

---

## Notation

| Symbol | Meaning |
|---|---|
| $T$ | Sample size |
| $H$ | Number of forecast horizons (leads) |
| $K$ | Number of endogenous regressors |
| $N_z$ | Number of instruments |
| $N_x$ | Dimension of predetermined control vector $X_{t-1}$ |
| $\otimes$ | Kronecker product |
| $\text{vec}(\cdot)$ | Column-stacking vectorization |
| $\text{Tr}(\cdot)$ | Trace |
| $P_A = A'(AA')^{-1}A$ | Projection matrix onto rows of $A$ |
| $M_A = I - P_A$ | Residualizing (annihilator) matrix |
| $\mathbb{P}_n$ | Set of $n \times n$ positive definite matrices |

**Restriction matrix.** Throughout we use

$$R_{K,H} = I_K \otimes \text{vec}(I_H) \in \mathbb{R}^{KH^2 \times K}.$$

When unambiguous, we write $R \equiv R_{K,H}$. Useful identities:
- $R' R = H \, I_K$
- For $U \in \mathbb{R}^{KH \times KH}$, the $(i,j)$-th element of $R'(U \otimes I_H) R$ is $\text{Tr}(U_{ij})$, where $U_{ij}$ is the $(i,j)$-th $H \times H$ block of $U$.
- For $U \in \mathbb{R}^{KH \times H}$, the $i$-th element of $R' \text{vec}(U')$ is $\text{Tr}(U_i)$, where $U_i$ is the $i$-th $H \times H$ row block.

---

## 1. Structural Model

The target structural equation is the scalar relation

$$y_t = \beta' Y_t + u_t, \qquad \beta \in \mathbb{R}^K, \tag{1}$$

with $\mathbb{E}[Y_t u_t] \neq 0$. The error $u_t$ admits the structural MA($\infty$) representation

$$u_t = \sum_{l=0}^\infty \mu_l' \varepsilon_{t-l}, \qquad \mathbb{E}[\varepsilon_t \varepsilon_t'] = I, \quad \mathbb{E}[\varepsilon_t \varepsilon_s'] = 0 \ \forall s \neq t.$$

Define $h$-step ahead residualized variables conditional on $X_{t-1}$:

$$
y_t^\perp(h) = y_{t+h} - \mathbb{E}[y_{t+h} \mid X_{t-1}],
\quad
Y_t^\perp(h) = Y_{t+h} - \mathbb{E}[Y_{t+h} \mid X_{t-1}],
$$
$$
u_t^\perp(h) = u_{t+h} - \mathbb{E}[u_{t+h} \mid X_{t-1}],
\quad
z_t^\perp = z_t - \mathbb{E}[z_t \mid X_{t-1}].
$$

The structural equation in residualized form is

$$y_t^\perp(h) = \beta' Y_t^\perp(h) + u_t^\perp(h), \qquad h = 0, \ldots, H-1. \tag{3}$$

---

## 2. Moment Conditions

### 2.1 Structural moments

SP-IV exploits $HN_z$ orthogonality conditions:

$$\mathbb{E}[u_t^\perp(h) \, z_t^\perp] = 0, \qquad h = 0, \ldots, H-1. \tag{4}$$

Stack the $H \times 1$ vectors $u_{H,t}^\perp = (u_t^\perp(0), \ldots, u_t^\perp(H-1))'$ and similarly $y_{H,t}^\perp$. Stack the $HK \times 1$ vector $Y_{H,t}^\perp = (Y_{H,t}^{1,\perp\prime}, \ldots, Y_{H,t}^{K,\perp\prime})'$, where $Y_{H,t}^{k,\perp}$ is the $H \times 1$ vector for the $k$-th endogenous variable. Then (4) compactly reads

$$\mathbb{E}[u_{H,t}^\perp(\beta) \otimes z_t^\perp] = 0, \qquad u_{H,t}^\perp(b) \equiv y_{H,t}^\perp - (b' \otimes I_H) Y_{H,t}^\perp. \tag{5}$$

### 2.2 Sufficient identification conditions

| Condition | Lag $l$ | Form |
|---|---|---|
| Contemporaneous exogeneity | $l = h$ | $\mu_l' \mathbb{E}[\varepsilon_{t+h-l} z_t'] = 0$ |
| Lead exogeneity | $l < h$ | $\mu_l' \mathbb{E}[\varepsilon_{t+h-l} z_t'] = 0$ |
| Lag exogeneity | $l > h$ | $\mu_l' \mathbb{E}[\varepsilon_{t+h-l} z_t'] = 0$ |

- **SE-IV (2SLS)** with $\{z_{t-h}\}_{h=0}^{H-1}$ requires all three.
- **SP-IV** with $X_{t-1}$ spanning past shocks in $u_t$ (so $u_t^\perp = \mu_0' \varepsilon_t$) requires only **contemporaneous + lead exogeneity** ($l \leq h$).

### 2.3 Forecasting moments

For a forecasting model $\mathbb{E}[\cdot \mid X_{t-1}] = g(\zeta) X_{t-1}$ linear in $X_{t-1}$ but possibly nonlinear in $\zeta$:

$$\mathbb{E}\left[X_{t-1} \otimes \left(y_{H,t}^{\perp\prime}(\zeta), \, Y_{H,t}^{\perp\prime}(\zeta), \, z_t^{\perp\prime}(\zeta)\right)'\right] = 0. \tag{6}$$

**Key fact (no two-step correction).** The Jacobian of (5) with respect to $\zeta$ vanishes at the truth because $u_{H,t}^\perp$ and $z_t^\perp$ are orthogonal to $X_{t-1}$ by construction. With a block-diagonal weighting matrix, the asymptotic variance of $\hat\beta$ does **not** depend on $\hat\zeta$. Plug-in estimation is valid without further adjustment.

---

## 3. The SP-IV Estimator

### 3.1 GMM problem

The objective function with structural-block weighting $\Phi_s(b, d) = I_H \otimes Q^{-1}$, $Q = \mathbb{E}[z_t^\perp z_t^{\perp\prime}]$, is

$$F_T(b) = \frac{1}{T}\left[\sum_{t=1}^T u_{H,t}^\perp(b) \otimes z_t^\perp\right]' \Phi_s \left[\sum_{t=1}^T u_{H,t}^\perp(b) \otimes z_t^\perp\right].$$

### 3.2 Closed-form estimator

Stack sample observations into

- $y_H^\perp$: $H \times T$ matrix with $y_t^\perp(h)$ in row $h+1$, column $t$
- $Y_H^\perp$: $HK \times T$ matrix, vertically stacking the $H \times T$ blocks $Y_H^{k,\perp}$, $k = 1, \ldots, K$
- $Z^\perp$: $N_z \times T$ matrix with $z_t^\perp$ in column $t$

Let $P_{Z^\perp} = Z^{\perp\prime}(Z^\perp Z^{\perp\prime})^{-1} Z^\perp$. The **SP-IV estimator** is

$$
\boxed{\;
\hat\beta = \left[R'\bigl(Y_H^\perp P_{Z^\perp} Y_H^{\perp\prime} \otimes I_H\bigr) R\right]^{-1} R' \, \text{vec}\bigl(y_H^\perp P_{Z^\perp} Y_H^{\perp\prime}\bigr).
\;}
\tag{9}
$$

**Equivalent characterization** — $\hat\beta$ minimizes

$$\text{Tr}\bigl(u_H^\perp \, P_{Z^\perp} \, u_H^{\perp\prime}\bigr), \qquad u_H^\perp = y_H^\perp - (\beta' \otimes I_H) Y_H^\perp,$$

i.e. it is the restricted IV estimator in the system

$$y_H^\perp = (\beta' \otimes I_H) Y_H^\perp + u_H^\perp, \tag{10}$$

with the restriction that the *same* $\beta$ applies at every horizon $h = 0, \ldots, H-1$.

### 3.3 IRF representation

Define the (scaled) reduced-form IRF coefficients

$$
\hat\Theta_Y = \frac{Y_H^\perp Z^{\perp\prime}}{T}\left(\frac{Z^\perp Z^{\perp\prime}}{T}\right)^{-1/2} \in \mathbb{R}^{HK \times N_z},
\qquad
\hat\Theta_y = \frac{y_H^\perp Z^{\perp\prime}}{T}\left(\frac{Z^\perp Z^{\perp\prime}}{T}\right)^{-1/2} \in \mathbb{R}^{H \times N_z}.
\tag{11}
$$

Rearrange into stacked form: $\boldsymbol{\hat\Theta}_y \in \mathbb{R}^{HN_z \times 1}$ and $\boldsymbol{\hat\Theta}_Y \in \mathbb{R}^{HN_z \times K}$:

$$
\boldsymbol{\hat\Theta}_Y = \bigl((Z^\perp Z^{\perp\prime}/T)^{-1/2} Z^\perp \otimes I_H / T\bigr) \mathbf{Y}_H^\perp,
\qquad
\boldsymbol{\hat\Theta}_y = \bigl((Z^\perp Z^{\perp\prime}/T)^{-1/2} Z^\perp \otimes I_H / T\bigr) \mathbf{y}_H^\perp,
\tag{12}
$$

where $\mathbf{y}_H^\perp = \text{vec}(y_H^\perp) \in \mathbb{R}^{TH}$ and $\mathbf{Y}_H^\perp = [\text{vec}(Y_H^{1,\perp}), \ldots, \text{vec}(Y_H^{K,\perp})] \in \mathbb{R}^{TH \times K}$.

Then

$$
\boxed{\;
\hat\beta = \bigl(\boldsymbol{\hat\Theta}_Y' \, \boldsymbol{\hat\Theta}_Y\bigr)^{-1} \boldsymbol{\hat\Theta}_Y' \, \boldsymbol{\hat\Theta}_y.
\;}
\tag{13}
$$

**Reading.** SP-IV is OLS of stacked IRF coefficients of $y_t$ on stacked IRF coefficients of $Y_t$.

### 3.4 Order condition

Because $\text{rank}\bigl(R'(\Theta_Y \Theta_Y' \otimes I_H) R\bigr) = \min\{K, H \cdot \text{rank}(\Theta_Y \Theta_Y')\}$, the order condition is

$$H N_z \geq K.$$

Adding leads of $y_t, Y_t$ substitutes for $N_z < K$ in the same way adding lags of $z_t$ does for SE-IV.

---

## 4. Implementation

### 4.1 Local projections (LP)

Let $X$ be the $N_x \times T$ matrix with $X_{t-1}$ in column $t$, $P_X = X'(XX')^{-1}X$, $M_X = I_T - P_X$. Define

$$
\hat y_H^\perp = y_H M_X, \qquad \hat Y_H^\perp = Y_H M_X, \qquad \hat Z^\perp = Z M_X. \tag{A.1}
$$

These are exactly the residuals from Jordà (2005) local projections of $y_{t+h}, Y_{t+h}$ on $z_t$ and $X_{t-1}$ at $h = 0, \ldots, H-1$ (by Frisch-Waugh-Lovell). Plug into (9) or, equivalently, into (13).

### 4.2 Vector autoregressions (VAR)

Suppose $X_t$ (which contains $y_t, Y_t, z_t$ and any additional variables) follows

$$X_t = A X_{t-1} + e_t. \tag{A.2}$$

Higher-order VARs are written in companion form. The OLS estimator is

$$\hat A = X^f X' (XX')^{-1}, \qquad X^f = [X_2, \ldots, X_{T+1}], \quad X = [X_1, \ldots, X_T].$$

The $h$-step ahead forecast errors are

$$\hat X_t^\perp(h) = \sum_{j=0}^h \hat A^{h-j} \hat e_{t+j}, \qquad \hat e_t = X_t - \hat A X_{t-1}. \tag{A.3}$$

Appropriate selection rows yield $\hat y_H^\perp$, $\hat Y_H^\perp$, $\hat Z^\perp$.

#### Imposing VAR dynamics on IRFs (recommended)

Let $\hat\Theta^{\text{VAR}}_{X,h} = \hat A^h \hat D_{1:N_z}$ be the VAR-implied IRFs at horizon $h$ to the $N_z$ identified shocks ($\hat D$ comes from the chosen identification scheme on $\hat e_t = \hat D \varepsilon_t$). Stack into

- $\hat\Theta^{\text{VAR}}_Y \in \mathbb{R}^{HK \times N_z}$: IRFs of $Y_t$
- $\hat\Theta^{\text{VAR}}_y \in \mathbb{R}^{H \times N_z}$: IRFs of $y_t$

Replace in (9):

$$
y_H^\perp P_{Z^\perp} Y_H^{\perp\prime} \;\longrightarrow\; \hat\Theta^{\text{VAR}}_y \bigl(\hat\Theta^{\text{VAR}}_Y\bigr)',
\qquad
Y_H^\perp P_{Z^\perp} Y_H^{\perp\prime} \;\longrightarrow\; \hat\Theta^{\text{VAR}}_Y \bigl(\hat\Theta^{\text{VAR}}_Y\bigr)'.
$$

This imposes the VAR restrictions consistently on both forecast errors and IRFs.

**Caveat — lag truncation.** If the DGP does not admit a finite-order VAR in $X_t$, IRFs at $h > p$ (lag length) are biased. The KLM test is robust to this; the AR test is **not** when VAR restrictions are imposed on IRFs (use "FE-only" version for AR).

---

## 5. Asymptotic Theory under Strong Identification

### 5.1 Assumptions

**Assumption 1** (forecasting). A unique solution $\zeta$ to (6) exists; the GMM estimator satisfies $\sqrt{T}(\hat\zeta - \zeta) \xrightarrow{d} N(0, V_\zeta)$; $\Phi(\beta, \zeta)$ is feasible and block-diagonal.

**Assumption 2** (structural).
- (a) $Z^\perp Z^{\perp\prime}/T \xrightarrow{p} Q$, positive definite.
- (b) $Y_H^\perp Z^{\perp\prime}/T \xrightarrow{p} \Theta_Y Q^{1/2}$, a real $HK \times N_z$ matrix.
- (c) $Z^\perp u_H^{\perp\prime}/T \xrightarrow{p} 0$ (exogeneity).
- (d) $R'(\Theta_Y \Theta_Y' \otimes I_H) R$ is fixed and full rank.

**Assumption 3** (CLT). $T^{-1/2} \text{vec}(Z^\perp u_H^{\perp\prime}) \xrightarrow{d} N(0, \Sigma_{u_H^\perp} \otimes Q)$, with $\Sigma_{u_H^\perp}$ full rank.

### 5.2 Consistency

**Proposition 4.** Under Assumptions 1 and 2, $\hat\beta \xrightarrow{p} \beta$.

### 5.3 Asymptotic normality

**Proposition 5.** Under Assumptions 1–3,

$$\sqrt{T}(\hat\beta - \beta) \xrightarrow{d} N(0, V_\beta), \tag{18}$$

with sandwich

$$
\boxed{\;
V_\beta = \bigl[R'(\Theta_Y \Theta_Y' \otimes I_H) R\bigr]^{-1} R' \bigl(\Theta_Y \Theta_Y' \otimes \Sigma_{u_H^\perp}\bigr) R \, \bigl[R'(\Theta_Y \Theta_Y' \otimes I_H) R\bigr]^{-1}.
\;}
$$

### 5.4 Feasible standard errors

Consistent estimator of $\Sigma_{u_H^\perp}$:

$$\hat\Sigma_{u_H^\perp} = \frac{\hat u_H^\perp \hat u_H^{\perp\prime}}{T - N_x - K}. \tag{19}$$

Replace $\Theta_Y \Theta_Y'$ by $Y_H^\perp P_{Z^\perp} Y_H^{\perp\prime} / T$ (or its VAR counterpart) in $V_\beta$ and plug in $\hat\Sigma_{u_H^\perp}$.

**No HAR adjustment needed.** When $X_{t-1}$ includes sufficient lags so that $z_t^\perp$ is serially uncorrelated, $u_{H,t}^\perp \otimes z_t^\perp$ is also serially uncorrelated and (19) suffices. This is unlike 2SLS, which requires HAR estimators due to mechanical autocorrelation in the overlapping lag sequence of $z_t$.

---

## 6. Weak Instruments Theory

### 6.1 Weak IV embedding

**Assumption 4.** $\Theta_Y = C/\sqrt T$, with $C \in \mathbb{R}^{HK \times N_z}$ fixed and $R_{K,H}(CC' \otimes I_H) R_{K,H}$ full rank.

### 6.2 Joint covariance

Define $w_H^\perp = y_H^\perp - (\beta' \otimes I_H) \Theta_Y Q^{-1/2} Z^\perp$ (reduced-form errors) and $v_H^\perp = Y_H^\perp - \Theta_Y Q^{-1/2} Z^\perp$ (first-stage errors). The joint covariance is

$$W = \begin{pmatrix} W_1 & W_{12} \\ W_{12}' & W_2 \end{pmatrix} \in \mathbb{P}_{(K+1)H},$$

with $W_1 = \text{Cov}(w_H^\perp)$, $W_2 = \text{Cov}(v_H^\perp)$, $W_{12} = \text{Cov}(w_H^\perp, v_H^{\perp\prime})$.

The "structural-side" covariance matrix is

$$
S = \begin{pmatrix} S_1 & S_{12} \\ S_{12}' & S_2 \end{pmatrix},
\qquad
\begin{aligned}
S_1 &= W_1 + (\beta' \otimes I_H) W_2 (\beta \otimes I_H) - (\beta' \otimes I_H) W_{12}' - W_{12}(\beta \otimes I_H) \\
S_{12} &= W_{12} - (\beta' \otimes I_H) W_2 \\
S_2 &= W_2.
\end{aligned}
\tag{I.4}
$$

### 6.3 Concentration matrix

**Definition 1.**

$$
\boxed{\;
\Lambda = \frac{1}{N_z} \Phi^{-1/2} R_{K,H}(CC' \otimes I_H) R_{K,H} \Phi^{-1/2}, \qquad \Phi = R_{K,H}'(S_2 \otimes I_H) R_{K,H} \in \mathbb{R}^{K \times K}.
\;}
$$

### 6.4 Bias criterion

**Proposition 6.** Under Assumptions 4 and 5, with $\beta^* = \hat\beta - \beta$:

$$\beta^* \xrightarrow{d} \left[R_{K,H}'(\eta_2 \eta_2' \otimes I_H) R_{K,H}\right]^{-1} R_{K,H}' \text{vec}(\eta_1 \eta_2'),$$

where $(\eta_1, \eta_2)$ are jointly normal with mean $(0, C)$ and covariance $S \otimes I_{N_z}$.

**Definition 2.** Bias criterion

$$B = \text{Tr}(S_1)^{-1/2} \, \bigl\| \mathbb{E}[\beta^*]' \Phi^{1/2} \bigr\|_2.$$

By Staiger-Stock-style approximation:

$$\mathbb{E}[\beta^*] \approx \frac{\text{vec}(S_{12})' R_{K,H} \Phi^{-1/2}}{\text{Tr}(S_1)^{1/2}} (I_K + \Lambda)^{-1} \Phi^{-1/2} \text{Tr}(S_1)^{1/2}. \tag{I.5}$$

The criterion is at most $1$ (worst case: errors are perfect linear combinations of second-stage regressors and $\Lambda = 0$).

**Definition 3.** Weak instrument set at tolerance $\tau$:

$$\mathcal{B}_\tau(W) = \{C, \beta : B \geq \tau\}.$$

### 6.5 Nagar approximation

The bias decomposes as $B = \|h \rho\|_2$, where (skipping the full expression of $h$, see Online Appendix I.3):

$$\rho = (\Phi^{-1/2} \otimes I_{H^2}) \text{vec}(S_{12}) / \sqrt{\text{Tr}(S_1)},$$
$$l = S_2^{-1/2} C, \quad \psi = S_2^{-1/2}(\eta_2 - C), \quad \text{vec}(\psi) \sim N(0, I_{KHN_z}).$$

The **Nagar (1959) approximation** $h_n$ expands around $\psi = 0$:

$$h_n = N_z^{-1} Q_\Lambda D_\Lambda^{-1/2} M_1 (D_\Lambda^{-1/2} Q_\Lambda \otimes L_0 \otimes I_K)(I_{KH} \otimes (I_{N_z} \otimes L_0) K_{N_z, HN_z} R_{H, N_z}) M_2,$$

with $\Lambda = Q_\Lambda D_\Lambda Q_\Lambda'$ the eigendecomposition, $M_1 = R_{K,K}'(I_{K^3} + (K_{K,K} \otimes I_K))$, $M_2 = R_{K,H} R_{K,H}'/(K+1) - I_{KH^2}$, $L_0 = (HN_z)^{-1/2} Q_\Lambda' \Lambda^{-1/2} R_{K, HN_z}'(S \, \text{vec}(l) \otimes I_{HN_z})$.

The Nagar bias $B_n = \|h_n \rho\|_2$ satisfies

$$B_n \leq \lambda_{\min}^{-1} \, \mathcal{B}(W), \qquad \lambda_{\min} = \text{mineval}\{\Lambda\}, \tag{I.8}$$

with

$$\mathcal{B}(W) = (N_z \sqrt H)^{-1} \sup_{L_0} \|M_1 (I_K \otimes L_0 \otimes I_K)(I_{KH} \otimes (I_{N_z} \otimes L_0) K_{N_z, HN_z} R_{H, N_z}) M_2 \Psi \|_2,$$
$$\Psi = S W_2^{-1/2} \bigl([W_{12} : W_2]' \otimes I_H\bigr) R_{K+1, H} \bigl(R_{K+1, H}'(W \otimes I_H) R_{K+1, H}\bigr)^{-1/2}. \tag{I.10}$$

### 6.6 Test of weak instruments

Hypotheses for tolerance $\tau$:

$$H_0: \lambda_{\min} \leq \lambda_{\min}^*(\tau) \quad \text{vs.} \quad H_1: \lambda_{\min} > \lambda_{\min}^*(\tau),$$

with $\lambda_{\min}^*(\tau) = \mathcal{B}(W)/\tau$.

**Proposition 7.** Test statistic

$$
\boxed{\;
g = N_z^{-1} \, \text{mineval}\bigl\{\hat\Phi^{-1/2} \bigl(Y_H^\perp P_{Z^\perp} Y_H^{\perp\prime}\bigr) \hat\Phi^{-1/2}\bigr\}, \qquad \hat\Phi = R_{K,H}'(\hat W_2 \otimes I_H) R_{K,H}.
\;}
$$

Under Assumptions 4 and 5,

$$g \xrightarrow{d} \text{mineval}\bigl\{R_{K,H}'(\zeta \otimes I_K) R_{K,H} / (H N_z)\bigr\},$$

where $\zeta = S(l+\psi)(l+\psi)'S' \sim \mathcal{W}(N_z, \Sigma, \Omega)$ is non-central Wishart, with covariance $\Sigma = SS' \in \mathbb{P}_{KH}$ and noncentrality matrix $\Omega = \Sigma^{-1} S l l' S'$.

### 6.7 Critical values via Imhof bounding

The exact limiting distribution of $g$ is unknown. Bound by the distribution of $\gamma' R'_{K,H}(\zeta \otimes I_H) R_{K,H} \gamma$, where $\gamma$ is the eigenvector of the minimum eigenvalue of $\Lambda$, $\gamma' \gamma = 1$.

**Theorem 1.** For $\zeta \sim \mathcal{W}(N_z, \Sigma, \Omega)$:

- $n$-th cumulant: $\kappa_n = 2^{n-1}(n-1)! \left[ N_z \text{Tr}\bigl(((\gamma \gamma' \otimes I_H) \Sigma)^n\bigr) + n \text{Tr}\bigl(((\gamma \gamma' \otimes I_H) \Sigma)^n \Omega\bigr) \right]$.
- For $n > 1$:
$$\kappa_n \leq 2^{n-1}(n-1)! \bigl[N_z \text{maxeval}\{R'_{K,H}(\Sigma^n \otimes I_H) R_{K,H}\} + n H N_z \lambda_{\min} \, \text{maxeval}\{\Sigma\}^{n-1}\bigr].$$

Use Imhof (1961) approximating distributions matching the first three cumulants. Pick the Imhof distribution with the largest critical value at level $\alpha$, subject to $\kappa_1 = HN_z(1 + \lambda_{\min})$ and the upper bounds on $\kappa_2, \kappa_3$. The resulting critical value is **conservative**.

---

## 7. Weak Instrument Robust Inference

### 7.1 Anderson-Rubin (S-statistic)

$$
\boxed{\;
\text{AR}(b) = (T - d_{\text{AR}}) \, \text{Tr}\bigl[u_H^\perp(b) P_{Z^\perp} u_H^{\perp\prime}(b) \bigl(u_H^\perp(b) M_{Z^\perp} u_H^{\perp\prime}(b)\bigr)^{-1}\bigr], \qquad d_{\text{AR}} = N_z + N_x.
\;}
\tag{20}
$$

Under $H_0: b = \beta$: $\text{AR}(\beta) \xrightarrow{d} \chi^2_{HN_z}$.

### 7.2 Kleibergen KLM

Define
- $\Xi = u_H^\perp(b) M_{Z^\perp} u_H^{\perp\prime}(b)$
- $\check u_H^\perp(b) = u_H^\perp(b) M_{Z^\perp}$
- $\check v_H^\perp = v_H^\perp M_{Z^\perp}$
- $\check Y_H = Y_H^\perp P_{Z^\perp} - \check v_H^\perp \check u_H^{\perp\prime}(b) \bigl(\check u_H^\perp(b) \check u_H^{\perp\prime}(b)\bigr)^{-1} u_H^\perp(b) P_{Z^\perp}$

$$
\text{K}(b) = (T - d_{\text{K}}) \, \text{vec}\bigl(\Xi^{-1} u_H^\perp(b) \check Y_H'\bigr)' R \bigl[R'(\check Y_H \check Y_H' \otimes \Xi^{-1} u_H^\perp(b) u_H^{\perp\prime}(b) \Xi^{-1}) R\bigr]^{-1} R' \text{vec}\bigl(\Xi^{-1} u_H^\perp(b) \check Y_H'\bigr).
\tag{21}
$$

Under $H_0$: $\text{K}(\beta) \xrightarrow{d} \chi^2_K$.

### 7.3 VAR-restricted versions

To impose VAR restrictions on the IRFs in (20), (21), replace:
- $\hat u_H^\perp(b) P_{\hat Z^\perp} \;\to\; \bigl(\hat\Theta^{\text{VAR}}_y - (b' \otimes I_H) \hat\Theta^{\text{VAR}}_Y\bigr) (Z M_X Z'/T)^{-1/2} Z M_X$
- $\hat Y_H^\perp P_{\hat Z^\perp} \;\to\; \hat\Theta^{\text{VAR}}_Y (Z M_X Z'/T)^{-1/2} Z M_X$
- $\hat u_H^\perp(b) M_{\hat Z^\perp} \;\to\; \hat u_H^\perp(b) - \bigl(\hat\Theta^{\text{VAR}}_y - (b' \otimes I_H) \hat\Theta^{\text{VAR}}_Y\bigr) (Z M_X Z'/T)^{-1/2} Z M_X$

For AR, the normalizing variance also requires the **Delta method** to account for the parametric VAR restriction.

---

## 8. Generalized SP-IV

Using the efficient weighting $\Phi_s(\beta, \zeta) = \Sigma_{u_H^\perp}^{-1} \otimes Q^{-1}$ yields the GLS / efficient-GMM estimator:

$$
\boxed{\;
\hat\beta_G = \left[R'\bigl(Y_H^\perp P_{Z^\perp} Y_H^{\perp\prime} \otimes \Sigma_{u_H^\perp}^{-1}\bigr) R\right]^{-1} R' \bigl(Y_H^\perp P_{Z^\perp} \otimes \Sigma_{u_H^\perp}^{-1}\bigr) \text{vec}(y_H^\perp P_{Z^\perp}).
\;}
\tag{B.1}
$$

With Assumption 2.d replaced by $R'(\Theta_Y \Theta_Y' \otimes \Sigma_{u_H^\perp}^{-1}) R$ full rank,

$$\sqrt T (\hat\beta_G - \beta) \xrightarrow{d} N\!\left(0, \bigl[R'(\Theta_Y \Theta_Y' \otimes \Sigma_{u_H^\perp}^{-1}) R\bigr]^{-1}\right). \tag{B.2}$$

Feasible: replace $\Sigma_{u_H^\perp}$ with (19) using residuals from a first-step $\hat\beta$. Iterate if desired.

**Continuously updating estimator (CUE).** Minimizer of the AR statistic in $b$. At $\hat\beta_{\text{CUE}}$, $\text{K}(\hat\beta_{\text{CUE}}) = 0$; AR and KLM confidence sets contain it.

**Practical note.** In the SW DGP simulations, GSP-IV does **not** reliably improve on baseline SP-IV in realistic samples due to estimation noise in $\hat\Sigma_{u_H^\perp}^{-1}$. Use with caution at small $T$ or weak $\hat\beta$.

---

## 9. Algorithmic Summary

```
INPUT:  data (y, Y, z, X); horizon H; lag policy for X; choice {LP, VAR}
OUTPUT: β̂, V̂_β, weak-IV test g, robust statistics AR(b), K(b)

1.  CONSTRUCT FORECAST ERRORS
    if LP:
        ŷ⊥_H, Ŷ⊥_H, Ẑ⊥  ←  residualize y_H, Y_H, Z on X_{t-1}  (eq. A.1)
    if VAR:
        Â  ←  OLS of X_t on X_{t-1}                              (eq. A.2)
        êₜ ←  X_t − Â X_{t-1}
        X̂⊥_t(h) ←  Σ_{j=0}^h Â^{h-j} êₜ₊ⱼ                      (eq. A.3)
        select rows for ŷ⊥_H, Ŷ⊥_H, Ẑ⊥
        OPTIONAL: compute Θ̂^VAR_y, Θ̂^VAR_Y from Â^h D̂_{1:N_z}
                   and impose VAR dynamics on the IRFs

2.  SP-IV ESTIMATOR
    A   ←  R'(Ŷ⊥_H P_Ẑ⊥ Ŷ⊥'_H ⊗ I_H) R           (K × K)
    b   ←  R' vec(ŷ⊥_H P_Ẑ⊥ Ŷ⊥'_H)               (K × 1)
    β̂  ←  A⁻¹ b                                  (eq. 9)

3.  ASYMPTOTIC VARIANCE (strong identification)
    û⊥_H        ←  ŷ⊥_H − (β̂' ⊗ I_H) Ŷ⊥_H
    Σ̂_{u⊥_H}  ←  û⊥_H û⊥'_H / (T − N_x − K)     (eq. 19)
    Θ̂Θ̂'      ←  Ŷ⊥_H P_Ẑ⊥ Ŷ⊥'_H / T
    V̂_β       ←  A⁻¹ [R'(Θ̂Θ̂' ⊗ Σ̂_{u⊥_H}) R] A⁻¹

4.  WEAK-INSTRUMENT TEST (Proposition 7)
    Ŵ₂  ←  v̂⊥_H v̂⊥'_H / T   with v̂⊥_H = Ŷ⊥_H − Π̂ Ẑ⊥
    Φ̂   ←  R'(Ŵ₂ ⊗ I_H) R
    g    ←  N_z⁻¹ mineval{ Φ̂⁻¹⸍² (Ŷ⊥_H P_Ẑ⊥ Ŷ⊥'_H) Φ̂⁻¹⸍² }
    compare to Imhof-bounding critical value at (τ, α)

5.  ROBUST INFERENCE (optional, for given test point b)
    AR(b) per (20)
    K(b)  per (21)
```

---

## References

- Lewis, D. J., & Mertens, K. (2024). *Dynamic Identification Using System Projections on Instrumental Variables.* Working paper.
- Lewis, D. J., & Mertens, K. (2022). *A Robust Test for Weak Instruments with Multiple Endogenous Regressors.* FRB New York Staff Reports 1020.
- Stock, J., & Yogo, M. (2005). *Testing for Weak Instruments in Linear IV Regression.* In Andrews (Ed.), *Identification and Inference for Econometric Models*. Cambridge UP.
- Montiel-Olea, J. L., & Pflueger, C. (2013). *A Robust Test for Weak Instruments.* JBES, 31(3), 358–369.
- Kleibergen, F. (2005). *Testing Parameters in GMM Without Assuming that They Are Identified.* Econometrica, 73(4), 1103–1123.
- Stock, J. H., & Wright, J. H. (2000). *GMM with Weak Identification.* Econometrica, 68(5), 1055–1096.
- Nagar, A. L. (1959). *The Bias and Moment Matrix of the General k-Class Estimators.* Econometrica, 27(4), 575–595.
- Imhof, J. P. (1961). *Computing the Distribution of Quadratic Forms in Normal Variables.* Biometrika, 48(3-4), 419–426.
- Jordà, Ò. (2005). *Estimation and Inference of Impulse Responses by Local Projections.* AER, 95(1), 161–182.
