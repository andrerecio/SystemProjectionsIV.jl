# Dynamic Identification Using System Projections on Instrumental Variables

**Daniel J. Lewis** (University College London) · **Karel Mertens** (FRB Dallas, CEPR)

*July 1, 2024*

---

## Abstract

We propose **System Projections on Instrumental Variables (SP-IV)** to estimate structural relationships using regressions of structural impulse responses obtained from local projections or vector autoregressions. Relative to IV with distributed lags of shocks as instruments, SP-IV imposes weaker exogeneity requirements and can improve efficiency and increase effective instrument strength relative to the typical 2SLS estimator. We describe inference under strong and weak identification. The SP-IV estimator outperforms other estimators of Phillips Curve parameters in simulations. We estimate the Phillips Curve implied by the main business cycle shock of Angeletos et al. (2020) and find that the impulse responses are consistent with weak but also relatively strong cyclical connections between inflation and unemployment.

**JEL classification:** E3, C32, C36.
**Keywords:** Structural Equations, Instrumental Variables, Impulse Responses, Robust Inference, Phillips Curve, Inflation Dynamics.

---

## Introduction

This paper studies the estimation of $\beta$ in structural time series equations of the form

$$y_t = \beta' Y_t + u_t, \tag{1}$$

where $y_t$ is a scalar observation of an outcome variable in period $t$, $Y_t$ is a $K \times 1$ vector of explanatory variables, $u_t$ is an error term (which may or may not be i.i.d.), and $\beta$ contains the $K$ structural parameters of interest. The explanatory variables $Y_t$ may contain contemporaneous variables but also lagged variables or agents' expectations of future variables that the econometrician may not measure well. We are interested in applications where $\mathbb{E}[Y_t u_t] \neq 0$, such that standard regression techniques yield inconsistent estimates of $\beta$ due to endogeneity.

Equation (1) nests a wide range of dynamic relationships of interest in macroeconomics. Consider the example of the **Hybrid New Keynesian Phillips Curve** ("Phillips Curve"),

$$\pi_t = \gamma_b \pi_{t-1} + \gamma_f \pi^e_{t+1} + \lambda \, gap_t + u_t, \tag{2}$$

where $\pi_t$ denotes inflation, $\pi^e_{t+1}$ is a measure of price setters' period $t$ expectation of inflation in $t+1$, and $gap_t$ is an output gap measure (the deviation of actual economic activity from the level without price rigidities). Equation (2) maps into the more general problem in (1) with $y_t = \pi_t$, $Y_t = [\pi_{t-1}, \pi^e_{t+1}, gap_t]'$ and $\beta = [\gamma_b, \gamma_f, \lambda]'$. The estimation of $\beta$ is complicated by a number of well-known problems that result in $\mathbb{E}[Y_t u_t] \neq 0$ (see Mavroeidis et al. 2014; McLeay and Tenreyro 2019; Barnichon and Mesters 2020):

- **Measurement error**: in practice the output gap and inflation expectations must be replaced with proxy measures.
- **Simultaneity**: the error term generally includes structural shocks that also influence the endogenous variables in $Y_t$.

A common approach in the literature is to rely on dynamics for identification and use lagged variables as instrumental variables. In the Phillips Curve application, it is typical to use $gap_{t-1}, gap_{t-2}, \ldots$ and $\pi_{t-2}, \pi_{t-3}, \ldots$, or lags of other readily available macroeconomic variables. Instrument exogeneity requires that the error term $u_t$ is uncorrelated with any of the instrumenting lagged macroeconomic variables. There is no general reason to believe that restrictions of this sort hold when the instruments are lags of standard macroeconomic variables. Lags of the output gap or inflation are, for example, not valid instruments for (2) in medium-scale macroeconomic models such as the Smets and Wouters (2007) model.

For this reason, Barnichon and Mesters (2020) propose IV with current and lagged values of **external measures of monetary policy shocks** as instruments. In general, however, the literature is rarely comfortable with imposing the strong assumption of unconditional lag exogeneity on external shock measures and typically avoids doing so by including a rich set of lagged macroeconomic controls in VARs and local projections (LPs). Unfortunately, when estimating structural equations rather than impulse responses, including such controls in conventional single-equation IV (SE-IV) regressions with a distributed lag (DL) of shocks as instruments **shrinks the explanatory power of the instruments to that of only the contemporaneous shock**, resulting in weaker or even under-identification.

In this paper, we propose a novel approach to identifying and estimating $\beta$ that allows the inclusion of lagged variables as controls without weakening identification. Specifically, we replace the single equation (1) with an $H$-dimensional **system of structural equations in forecast errors** of $y_t$ and $Y_t$, where $H$ is the number of leads. The forecast errors can be derived from a variety of forecasting models, including VARs or LPs with a rich set of controls. The contemporaneous values of the $N_z$ instrumental variables generate $H N_z$ moment conditions, which we solve in closed form for $\beta$, yielding a restricted IV estimator in the system of reduced form forecast errors. We refer to this methodology as **System Projections on Instrumental Variables (SP-IV)**.

SP-IV estimates structural equations on the basis of the relationships between empirical estimates of the dependent and independent variables' impulse responses to economic shocks. We show that SP-IV is equivalent to a straightforward regression of the IRF of $y_t$ on the IRFs of $Y_t$, where the IRFs can be obtained from a VAR, LPs, or other valid impulse response estimators. Intuitively, SP-IV finds the linear combination of IRFs of the endogenous variables to one or more suitably chosen structural shocks that most closely matches the IRFs of the dependent variable to the same shocks.

Depending on the data generating process (DGP), SP-IV has several advantages relative to SE-IV using a DL of instruments:

1. **Weaker exogeneity requirements**: with adequate controls, it requires only the weaker assumptions of contemporaneous and lead exogeneity of the instruments, compared to contemporaneous, lead, and lag exogeneity for SE-IV.
2. **Efficiency**: the use of forecast errors instead of raw variables can improve efficiency in estimating $\beta$.
3. **Stronger identification**: similar efficiency gains in the first stage can increase effective instrument strength, mitigating weak instrument problems.

As SP-IV is a GMM estimator, inference is straightforward under strong identification. We develop a first-stage test for instrument strength by extending the popular bias-based test in Stock and Yogo (2005) to the SP-IV setting. As instruments are often weak in practice, we propose weak instrument robust inference procedures based on the Anderson and Rubin (1949) AR statistic and Kleibergen's (2005) KLM statistic.

### Notation

$\otimes$ denotes the Kronecker product, $\text{Tr}(\cdot)$ the trace operator, $\text{vec}(\cdot)$ the vectorization operator, $\text{mineval}\{\cdot\}/\text{maxeval}\{\cdot\}$ the minimum/maximum eigenvalue, $\xrightarrow{p}$ convergence in probability, $\xrightarrow{d}$ convergence in distribution, and $P_X = X'(XX')^{-1}X$ the projection matrix.

---

## 1. System Projections on Instrumental Variables

We begin by reformulating the dynamic relationship in (1) in terms of forecast errors. Taking $h$-horizon leads and taking residuals after conditioning on an $N_x \times 1$ vector of predetermined variables $X_{t-1}$ yields

$$y^\perp_t(h) = \beta' Y^\perp_t(h) + u^\perp_t(h), \tag{3}$$

where $y^\perp_t(h) = y_{t+h} - \mathbb{E}[y_{t+h} \mid X_{t-1}]$, $Y^\perp_t(h) = Y_{t+h} - \mathbb{E}[Y_{t+h} \mid X_{t-1}]$, and $u^\perp_t(h) = u_{t+h} - \mathbb{E}[u_{t+h} \mid X_{t-1}]$. Let $z_t$ denote an $N_z \times 1$ vector of instrumental variables, and define $z^\perp_t = z_t - \mathbb{E}[z_t \mid X_{t-1}]$. We focus on applications that rely on dynamics for identification, exploiting orthogonality conditions between the error term $u_t$ and $z_t, z_{t-1}, \ldots$. Instead of the usual approach of imposing orthogonality between $z_{t-h}$ and $u_t$ for various $h \geq 0$, we impose

$$\mathbb{E}[u^\perp_t(h) z^\perp_t] = 0; \quad h = 0, \ldots, H-1. \tag{4}$$

Without conditioning on $X_{t-1}$ and under stationarity, the orthogonality conditions in (4) are equivalent to imposing $\mathbb{E}[u_t z_{t-h}] = 0$, as in conventional SE-IV. The key departure compared to using a DL of $z_t$ as instruments is that the moments in (4) are in terms of forecast errors after conditioning on the predetermined predictors $X_{t-1}$, where lags of $z_t$ may be included in $X_{t-1}$.

### 1.1 The Generalized Method of Moments Problem

The conditions in (4) provide a set of $H N_z$ moment conditions that can be used to identify the $K$ elements of $\beta$. Let $y^\perp_{H,t}$ and $u^\perp_{H,t}$ denote the $H \times 1$ vectors in which the $(h+1)$-th element is $y^\perp_t(h)$ or $u^\perp_t(h)$ respectively. Let $Y^\perp_{H,t}$ denote the $HK \times 1$ vector stacking the $H \times 1$ vectors $Y^{k,\perp}_{H,t}$, where $Y^k_t$ is the $k$-th variable in $Y_t$. The moment conditions are

$$\mathbb{E}[u^\perp_{H,t}(\beta) \otimes z^\perp_t] = 0, \tag{5}$$

where $u^\perp_{H,t}(b) \equiv y^\perp_{H,t} - (b' \otimes I_H) Y^\perp_{H,t}$ and the truth is $b = \beta$.

The moment conditions in (5) can be augmented to account for the estimation of the forecast errors. The forecasting moment conditions are

$$\mathbb{E}\left[X_{t-1} \otimes \left(y^{\perp\prime}_{H,t}(\zeta), Y^{\perp\prime}_{H,t}(\zeta), z^{\perp\prime}_t(\zeta)\right)'\right] = 0, \tag{6}$$

where the true value of $d$ is $\zeta$. The class of forecasting models considered is linear in $X_{t-1}$ but possibly nonlinear in parameters $d$, including LPs and VARs.

**Assumption 1.** There exists a unique solution $\zeta$ to the forecasting moments (6), which are linear in $X_{t-1}$; the associated GMM estimator satisfies $\sqrt{T}(\hat{\zeta} - \zeta) \xrightarrow{d} N(0, V_\zeta)$ for some feasible block-diagonal weighting matrix $\Phi(\beta, \zeta)$ and positive definite $V_\zeta$.

Under Assumption 1, the Jacobian of (5) with respect to $d$ is zero in expectation at $\zeta$, which, with the structure of $\Phi(\beta, \zeta)$, implies that the asymptotic variance of $\hat{\beta}$ depends only on the asymptotic variance of the sample counterpart of (5). This means that estimating forecast errors and plugging them into the structural moments yields a valid asymptotic variance.

### 1.2 The SP-IV Estimator

Our baseline estimator uses $\Phi_s(b, d) = I_H \otimes Q^{-1}$, where $Q = \mathbb{E}[z^\perp_t z^{\perp\prime}_t]$. In population the solution identifies $\beta$ as

$$\beta = \left(R'(\mathbb{E}[Y^\perp_{H,t} z^{\perp\prime}_t] Q^{-1} \mathbb{E}[Y^\perp_{H,t} z^{\perp\prime}_t]' \otimes I_H) R\right)^{-1} R' \text{vec}\left(\mathbb{E}[y^\perp_{H,t} z^{\perp\prime}_t] Q^{-1} \mathbb{E}[Y^\perp_{H,t} z^{\perp\prime}_t]'\right), \tag{8}$$

where $R = I_K \otimes \text{vec}(I_H)$. Let the $H \times T$ matrix $y^\perp_H$, the $HK \times T$ matrix $Y^\perp_H$, and the $N_z \times T$ matrix $Z^\perp$ collect the sample observations. The sample analog of (8) is

$$\hat{\beta} = \left(R'(Y^\perp_H P_{Z^\perp} Y^{\perp\prime}_H \otimes I_H) R\right)^{-1} R' \text{vec}(y^\perp_H P_{Z^\perp} Y^{\perp\prime}_H), \tag{9}$$

which minimizes the GMM objective with respect to $b$. That minimization problem is equivalent to minimizing $\text{Tr}(u^\perp_H P_{Z^\perp} u^{\perp\prime}_H)$, the sum of squared residuals in the system

$$y^\perp_H = (\beta' \otimes I_H) Y^\perp_H + u^\perp_H, \tag{10}$$

after projection on the instruments $z^\perp_t$. Thus, $\hat{\beta}$ is the restricted IV estimator in the system of equations in (10), where the only restriction is that $\beta$ applies at all horizons.

**Interpretation in terms of IRFs.** Consider the IRF estimates

$$\hat{\Theta}_Y = \frac{Y^\perp_H Z^{\perp\prime}}{T}\left(\frac{Z^\perp Z^{\perp\prime}}{T}\right)^{-1/2}; \quad \hat{\Theta}_y = \frac{y^\perp_H Z^{\perp\prime}}{T}\left(\frac{Z^\perp Z^{\perp\prime}}{T}\right)^{-1/2}, \tag{11}$$

which are OLS coefficients from regressing $Y^\perp_{H,t}$ and $y^\perp_{H,t}$ on standardized innovations to the instruments. Then the SP-IV estimator $\hat{\beta}$ is equivalent to

$$\hat{\beta} = (\hat{\Theta}'_Y \hat{\Theta}_Y)^{-1} \hat{\Theta}'_Y \hat{\Theta}_y, \tag{13}$$

so $\hat{\beta}$ is the slope in the **OLS regression of $\hat{\Theta}_y$ on $\hat{\Theta}_Y$** — the coefficients in a regression of the IRFs of $y_t$ and $Y_t$ to $z_t$, conditional on $X_{t-1}$.

The expression for $\hat{\beta}$ in (13) suggests a simple **two-stage procedure**:

1. **First stage**: estimate IRFs using instruments satisfying the exogeneity conditions.
2. **Second stage**: given a set of IRF estimates, the SP-IV estimator is obtained by regressing the IRF of the outcome variable, $y_t$, on the IRFs of the endogenous variables, $Y_t$.

For the Phillips curve example, the first stage estimates IRFs of inflation $\pi_t$ and the slack measure $gap_t$ to a monetary policy shock (or other aggregate demand shocks orthogonal to the cost-push term, $u_t$). In the second stage, the IRF of $\pi_t$ is regressed on the IRF of $gap_t$ as well as the IRFs of lagged and expected future inflation, $\pi_{t-1}$ and $\pi^e_{t+1}$. The latter can be obtained simply by lagging and leading the IRF of $\pi_t$ by one horizon.

### 1.3 SP-IV versus 2SLS Implementations of SE-IV

The standard SE-IV approach for identifying $\beta$ in (1) with $z_t, \ldots, z_{t-H+1}$ as instruments exploits the $H N_z$ orthogonality conditions

$$\mathbb{E}[u_t z_{t-h}] = 0; \quad h = 0, \ldots, H-1. \tag{14}$$

Barnichon and Mesters (2020) observe that the 2SLS estimates equal the estimates from OLS regression of the IRF of $y_t$ on the IRFs of $Y_t$ obtained from DL regressions. The 2SLS estimator with a DL of shocks can — like SP-IV — be interpreted in terms of a regression with IRFs. In 2SLS, however, the IRFs come from single-equation DL models without additional controls, whereas in SP-IV the IRFs can be obtained from **LPs with controls $X_{t-1}$ or from VARs**.

Depending on the DGP, the ability to accommodate controls yields three further potential advantages of SP-IV.

#### 1. Weaker Exogeneity Requirements for $z_t$

Adopt the impulse-propagation paradigm representing $y_t$ and $Y_t$ in terms of linear combinations of current and past realizations of structural shocks $\varepsilon_t$. Given the representations for $y_t$ and $Y_t$, (1) implies that the error term $u_t$ is generally also an MA($\infty$) in the structural shocks:

$$u_t = \mu'_0 \varepsilon_t + \mu'_1 \varepsilon_{t-1} + \mu'_2 \varepsilon_{t-2} + \ldots. \tag{15}$$

**Proposition 1.** Suppose (15) and stationarity hold:

- The **SE-IV** exogeneity condition with lags of $z_t$ holds when
  $$\mu'_l \, \mathbb{E}[\varepsilon_{t+h-l} z'_t] = 0; \quad l = 0, \ldots, \infty; \quad h = 0, \ldots, H-1. \tag{16}$$

- Suppose $X_{t-1}$ spans past shocks included in $u_t$ such that $u^\perp_t = \mu'_0 \varepsilon_t$. The **SP-IV** exogeneity condition holds when
  $$\mu'_l \, \mathbb{E}[\varepsilon_{t+h-l} z'_t] = 0; \quad l = 0, \ldots, h; \quad h = 0, \ldots, H-1. \tag{17}$$

Following Stock and Watson (2018), we denote the sufficient conditions in (16) with $l > h$ as **lag exogeneity**, with $l = h$ as **contemporaneous exogeneity**, and with $l < h$ as **lead exogeneity**. The SE-IV exogeneity condition requires all three. In contrast, the SP-IV exogeneity condition is implied by only contemporaneous and lead exogeneity, since by assumption conditioning on $X_{t-1}$ eliminates the influence of all past realizations of $\varepsilon_t$ on $u^\perp_t(h)$.

**Concrete example.** Consider the Phillips Curve in (2) with Romer and Romer's (2004) monetary policy surprises $z^{RR}_t$ as instruments. Assume the error term follows $u_t = \rho_u u_{t-1} + \upsilon_t$. Unless $\rho_u = 0$, $u_t$ depends on $\upsilon_t$ and all lags. Suppose the regression generating $z^{RR}_t$ is misspecified by omitting lags of inflation. Then $z^{RR}_t$ generally depends on lags of $\upsilon_t$, and the lag exogeneity requirement is not satisfied. However, by including lags of inflation amongst predictors $X_{t-1}$, the exogeneity requirements for SP-IV remain satisfied as long as contemporaneous and lead exogeneity hold.

#### 2. Efficiency Gains

**Proposition 2.** Suppose $z_t$ is i.i.d. and independent of $u_t$; then:

(i) If $u_t$ is i.i.d., or if $X_{t-1}$ is empty or otherwise uninformative for $u_t, \ldots, u_{t+H-1}$, SP-IV is asymptotically as efficient as 2SLS.

(ii) $\hat{\beta}$ is asymptotically more efficient than $\hat{\beta}_{2SLS}$ if $\text{maxeval}(\Sigma_{u_H}) > \text{maxeval}(\Sigma_{u^\perp_H})$.

Intuitively, the SP-IV estimator is more efficient than 2SLS if the variances of the forecast errors $u^\perp_t(h)$ at $h = 0, \ldots, H-1$ are small relative to the variance of the error term $u_t$. Efficiency gains from SP-IV are more likely when $u_t$ is more persistent, so $X_{t-1}$ explains a larger fraction of the variance of $u_t, \ldots, u_{t+H-1}$.

#### 3. Stronger Identification

**Proposition 3.** Assume $K = 1$ and $z_t$ is i.i.d. and satisfies (16):

(i) Unless $Y^\perp_t(h) = Y_{t+h}$ for $h = 0, \ldots, H-1$, the concentration parameter for SP-IV with controls $X_{t-1}$ is larger than for SP-IV without controls.

(ii) If $\text{Tr}(\Sigma_{v^\perp_H})/H < \sigma^2_\omega$, the concentration parameter for SP-IV is larger than that for 2SLS.

The effective instrument strength can increase relative to 2SLS depending on the persistence and predictability of the errors, as well as on $H$.

### 1.4 Consistency of the SP-IV Estimator

**Assumption 2.** The following hold:

- (a) $Z^\perp Z^{\perp\prime}/T \xrightarrow{p} \mathbb{E}[z^\perp_t z^{\perp\prime}_t] = Q$, positive definite;
- (b) $Y^\perp_H Z^{\perp\prime}/T \xrightarrow{p} \mathbb{E}[Y^\perp_{H,t} z^{\perp\prime}_t] = \Theta_Y Q^{1/2}$;
- (c) $Z^\perp u^{\perp\prime}_H/T \xrightarrow{p} \mathbb{E}[z^\perp_t u^{\perp\prime}_{H,t}] = 0$;
- (d) $R'(\Theta_Y \Theta'_Y \otimes I_H) R$ is a fixed matrix with full rank.

The order condition is $H N_z \geq K$: adding leads of $y_t$ and $Y_t$ makes up for $N_z < K$ just as adding lags of $z_t$ as instruments does for SE-IV.

**Proposition 4.** Under Assumptions 1 and 2, $\hat{\beta} \xrightarrow{p} \beta$.

---

## 2. Inference for SP-IV

### 2.1 Inference under Strong Instruments

**Assumption 3.** $T^{-1/2} \text{vec}(Z^\perp u^{\perp\prime}_H) \xrightarrow{d} N(0, \Sigma_{u^\perp_H} \otimes Q)$, where $\Sigma_{u^\perp_H}$ is full rank.

**Proposition 5.** Under Assumptions 1–3,

$$\sqrt{T}(\hat{\beta} - \beta) \xrightarrow{d} N(0, V_\beta), \tag{18}$$

where $V_\beta = (R'(\Theta_Y \Theta'_Y \otimes I_H) R)^{-1} R'(\Theta_Y \Theta'_Y \otimes \Sigma_{u^\perp_H}) R \, (R'(\Theta_Y \Theta'_Y \otimes I_H) R)^{-1}$.

A natural consistent estimator is

$$\hat{\Sigma}_{u^\perp_H} = \hat{u}^\perp_H \hat{u}^{\perp\prime}_H / (T - N_x - K). \tag{19}$$

Including adequate lags in $X_{t-1}$ (which can include lags of $z_t$) obviates the need for an autocorrelation-robust estimate by eliminating autocorrelation in $z^\perp_t$. This is **not the case for 2SLS**, which generally requires autocorrelation-robust methods due to mechanical autocorrelation in the overlapping lag sequence of $z_t$.

### 2.2 A Test for Weak Instruments

We derive a bias-based test of instrument strength for SP-IV analogous to the popular Stock and Yogo (2005) test. We consider a Nagar approximation of the bias under weak instrument asymptotics. Weak instruments are defined as those for which the bias in $\hat{\beta}$ is at least $\tau$ percent of a worst-case benchmark. The test statistic is similar to Cragg and Donald (1993), and rejects when the statistic exceeds the level-$\alpha$ critical value of a bounding distribution. The test nests Stock and Yogo (2005) when $H = 1$.

### 2.3 Weak Instrument Robust Inference

**AR Statistic.** The "S-statistic" of Stock and Wright (2000) extends AR to GMM:

$$AR(b) = (T - d_{AR}) \, \text{Tr}\left(u^\perp_H(b) P_{Z^\perp} u^\perp_H(b)' \left(u^\perp_H(b) M_{Z^\perp} u^\perp_H(b)'\right)^{-1}\right), \tag{20}$$

$$AR(\beta) \xrightarrow{d} \chi^2_{H N_z}.$$

**KLM Statistic.** When $H N_z > K$, the AR statistic can have poor power. The Kleibergen (2005) KLM statistic addresses this:

$$K(b) = (T - d_K) \, \text{vec}(\Xi^{-1} u^\perp_H(b) \check{Y}'_H)' R \left[R'(\check{Y}_H \check{Y}'_H \otimes \Xi^{-1} u^\perp_H(b) u^{\perp\prime}_H(b) \Xi^{-1}) R\right]^{-1} R' \text{vec}(\Xi^{-1} u^\perp_H(b) \check{Y}'_H), \tag{21}$$

$$K(\beta) \xrightarrow{d} \chi^2_K.$$

---

## 3. Performance of SP-IV in Model Simulations

The objective in all simulations is to estimate the parameters of the Phillips Curve in (2) using data generated from the Smets and Wouters (2007) (SW) model. An important feature is that the shocks underlying the error term $u_t$ explain a very large fraction of the variance of inflation. This means that, in realistic sample sizes, the weak instrument problem is generally severe.

In the SW model, the error term in (2) is the ARMA(1,1) process

$$u_t = \rho_u u_{t-1} + \varepsilon^p_t - \mu_p \varepsilon^p_{t-1}, \quad \rho_u = 0.99, \ \mu_p = 0.83 \tag{22}$$

Inverting the autoregressive term yields $u_t = \varepsilon^p_t + \rho_u(1 - \mu_p) \varepsilon^p_{t-1} + \ldots$, so $u_t$ depends on the entire history of price markup shocks. Lagged endogenous variables are not valid instruments.

We use a monetary policy shock as $z_t$. We consider two sets of simulations: one with a **lag-endogenous instrument** and one with the **true model monetary policy shock**.

We use a realistic set of controls consisting of seven endogenous model variables: the short-term interest rate, inflation, marginal cost, output, consumption, investment, and the real wage. Both LP and VAR implementations of SP-IV include four lags of these variables in $X_{t-1}$.

### 3.1 Simulations with Violations of Lag Exogeneity

We simulate "Romer and Romer (2004) instruments" consisting of the true monetary policy shocks augmented with a linear function of inflation over the past four quarters (calibrated against the actual RR series, $R^2 = 0.08$).

**Table 1: Results with Lag Endogenous Instrument, $T = 5000$**

*Mean Estimates*

| Estimator | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
|---|---|---|---|
| True Value | 0.15 | 0.85 | 0.05 |
| OLS | 0.48 | 0.48 | 0.00 |
| **H = 8** | | | |
| 2SLS | 0.27 | 0.58 | **−0.09** |
| SP-IV LP | 0.26 | 0.60 | **−0.08** |
| SP-IV LP-C | 0.16 | 0.84 | 0.05 |
| SP-IV VAR | 0.12 | 0.83 | 0.09 |
| **H = 20** | | | |
| 2SLS | 0.24 | 0.76 | −0.02 |
| SP-IV LP | 0.24 | 0.75 | −0.02 |
| SP-IV LP-C | 0.23 | 0.81 | 0.02 |
| SP-IV VAR | 0.17 | 0.83 | 0.05 |

*Empirical Size of Nominal 5% Tests*

| Test | H = 8 | H = 20 |
|---|---|---|
| Wald 2SLS | 55.0 | 96.0 |
| Wald SP-IV LP | 61.2 | 96.0 |
| Wald SP-IV LP-C | 9.3 | 34.9 |
| Wald SP-IV VAR | 5.5 | 13.3 |
| AR SE-IV | 72.5 | 72.3 |
| AR SP-IV LP | 67.6 | 54.4 |
| **AR SP-IV LP-C** | **4.6** | **5.8** |
| **AR SP-IV VAR** | **4.9** | **4.5** |
| KLM SE-IV | 82.5 | 85.9 |
| KLM SP-IV LP | 81.5 | 72.4 |
| **KLM SP-IV LP-C** | **5.2** | **5.5** |
| **KLM SP-IV VAR** | **4.9** | **4.6** |

The 2SLS estimates are strongly biased (wrong sign for $\lambda$!). SP-IV without controls is similarly biased. **SP-IV LP-C and VAR**, which condition on $X_{t-1}$, produce estimates with the correct sign and values close to the truth. The conditioning step adequately protects against the violation of lag exogeneity.

### 3.2 Small Sample Performance

Now we use the **true monetary policy shocks** as instruments to level the playing field.

**Table 2: Mean parameter estimates**

| | T=250 | | | T=500 | | | T=5000 | | |
|---|---|---|---|---|---|---|---|---|---|
| Estimator | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ | $\gamma_b$ | $\gamma_f$ | $\lambda$ |
| True Value | 0.15 | 0.85 | 0.05 | 0.15 | 0.85 | 0.05 | 0.15 | 0.85 | 0.05 |
| OLS | 0.47 | 0.47 | 0.00 | 0.48 | 0.48 | 0.00 | 0.48 | 0.48 | 0.00 |
| **H = 8** | | | | | | | | | |
| 2SLS | 0.27 | 0.51 | 0.01 | 0.24 | 0.61 | 0.00 | 0.17 | 0.83 | 0.04 |
| SP-IV LP | 0.26 | 0.51 | 0.01 | 0.24 | 0.60 | 0.00 | 0.17 | 0.83 | 0.04 |
| SP-IV LP-C | 0.29 | 0.64 | **0.05** | 0.25 | 0.74 | 0.04 | 0.16 | 0.84 | 0.05 |
| SP-IV VAR | 0.22 | 0.80 | 0.03 | 0.18 | 0.84 | 0.05 | 0.12 | 0.83 | 0.09 |
| **H = 20** | | | | | | | | | |
| 2SLS | 0.39 | 0.53 | 0.00 | 0.36 | 0.61 | 0.00 | 0.23 | 0.80 | 0.01 |
| SP-IV LP | 0.38 | 0.53 | 0.00 | 0.35 | 0.61 | 0.00 | 0.23 | 0.80 | 0.01 |
| SP-IV LP-C | 0.41 | 0.55 | 0.01 | 0.37 | 0.64 | 0.01 | 0.23 | 0.81 | 0.02 |
| SP-IV VAR | 0.27 | 0.80 | 0.01 | 0.23 | 0.84 | 0.02 | 0.17 | 0.83 | 0.05 |

For $T = 250$, $H = 8$: the LP-C implementation of SP-IV averages exactly the true $\lambda = 0.05$. The VAR implementation also delivers substantial bias improvements. **Reductions in small sample bias by adopting SP-IV LP-C or SP-IV VAR are sizeable**.

**Variance.** SP-IV with controls is asymptotically more efficient than 2SLS when $u_t$ is persistent. For $T = 5000$, $H = 8$, the standard deviations of SP-IV LP-C are uniformly smaller than 2SLS. The relative efficiency diminishes for larger $H$.

**Inference.** Wald tests for 2SLS show meaningful size distortions, increasing with $H$ and $N_z$. The robust SP-IV tests (AR and KLM) are generally well-sized; size distortions with large $H N_z$ are milder than for SE-IV versions. We recommend avoiding very large $H N_z$ also when using SP-IV. $H = 20$ appears as an upper bound for $T = 250$.

---

## 4. Application to the Phillips Curve with U.S. Data

We estimate the Phillips curve using U.S. monthly data (Jan 1978 – Feb 2020, 506 observations):

$$\pi^{1m}_t = (1 - \gamma_f) \pi^{1y}_{t-1} + \gamma_f \pi^{1y}_{t+12} + \lambda U_t + u_t, \tag{23}$$

where $\pi^{1m}_t$ is the annualized monthly Core CPI change, $\pi^{1y}_t$ is the year-over-year Core CPI change, and $U_t$ is the unemployment rate. We restrict $\gamma_b + \gamma_f = 1$ (no long-run trade-off). Maximum horizon: 3 years (36 months), using $h = 0, 3, 6, \ldots, 33$ (12 quarterly horizons).

**VAR controls (6 lags):** annualized monthly Core CPI, unemployment, 12-month change in log industrial production, 12-month PPI change, 3-month Treasury rate, 10-year Treasury rate.

**Instrument:** monthly version of the **Angeletos et al. (2020) Main Business Cycle (MBC) Shock**, identified by maximizing the contribution to cyclical unemployment fluctuations in the frequency domain.

**First-Stage Test Results** (showing test statistic $g$ and 5% critical value $cv$ for the joint test of both endogenous regressors):

| Instrument | 2SLS $g$ / $cv$ | SP-IV $g$ / $cv$ |
|---|---|---|
| **MBC** | 2.3 / 21.9 | **6.3** / 22.0 |
| RR (Romer-Romer) | 0.3 / 20.2 | 0.4 / 21.2 |
| GK (Gertler-Karadi) | 3.5 / 21.0 | 0.1 / 22.3 |
| MAR (Miranda-Agrippino-Ricco) | 0.1 / 19.7 | 0.0 / 21.0 |
| JK (Jarociński-Karadi) | 0.6 / 19.9 | 0.0 / 22.3 |

The MBC shock test statistic is 6.3 for SP-IV (vs 2.3 for 2SLS), illustrating how SP-IV amplifies the signal of the instrument. **Each monetary policy shock measure is too weak to be useful for identifying the Phillips curve in this sample**.

**Point Estimates and Confidence Sets**

- Point estimates: $\gamma_f = 0.65$ (2SLS) vs $0.66$ (SP-IV); $\lambda = -0.14$ (2SLS) vs $-0.15$ (SP-IV).
- 2SLS confidence sets do not reject any plausible values of $\gamma_f$ or rule out a wide range of $\lambda$.
- **SP-IV confidence sets are much sharper**: rule out values of $\gamma_f$ meaningfully below 0.5 or above 1. The 90% set for $\lambda$ ranges from about −0.6 to slightly above 0.

**Conclusion of application:** The confidence sets are consistent with weak but **also relatively strong cyclical connections** between inflation and unemployment. The business cycle anatomy of Angeletos et al. (2020) does not provide clear-cut evidence that inflation and activity are largely disconnected.

---

## 5. Concluding Remarks and Future Research

SP-IV can help identify a wide variety of structural relationships in macroeconomics, such as Euler equations, wage Phillips curves, monetary or fiscal policy rules, and aggregate production functions. It can be used to conduct inference on ratios of impulse response coefficients (Okun coefficients, sacrifice ratios, multipliers, etc.). The methodology could be extended to panel data settings and cross-sectional applications. Future work can also develop Bayesian implementations or methods to optimally select horizons.

---

## Appendix A. Practical Implementation of SP-IV with LPs or VARs

### Local Projections

Define the projection matrix $P_X = X'(XX')^{-1}X$ and residualizing matrix $M_X = I_T - P_X$. The forecast errors are:

$$\hat{y}^\perp_H = y_H M_X, \quad \hat{Y}^\perp_H = Y_H M_X, \quad \hat{Z}^\perp = Z M_X. \tag{A.1}$$

By Frisch-Waugh-Lovell, this is equivalent to estimating Jordà (2005) local projections of $y_{t+h}$ and $Y_{t+h}$ on $z_t$ and $X_{t-1}$.

### Vector Autoregressions

Suppose $X_t = A X_{t-1} + e_t$. The standard estimator is $\hat{A} = X^f X'(XX')^{-1}$, leading to $h$-step ahead forecast errors:

$$\hat{X}^\perp_t(h) = \sum_{j=0}^{h} \hat{A}^{h-j} \hat{e}_{t+j}. \tag{A.3}$$

**Preferred implementation** with structural VARs: select the elements corresponding to $y_t$ and $Y_t$ in $\hat{\Theta}^{VAR}_{X,h}$ to form $\hat{\Theta}_y$ and $\hat{\Theta}_Y$, then obtain the SP-IV estimator from the regression of impulse responses as in (13). This imposes the VAR dynamics on both the reduced form forecast errors and the impulse responses.

**Lag truncation in the VAR.** When the DGP does not admit a finite-order VAR representation, IRFs at horizons exceeding the lag length are biased. For inference: the AR test is affected by lag truncation bias when VAR restrictions are imposed on the IRFs (use "FE only" version); the KLM test is more robust to lag truncation bias.

## Appendix B. Generalized and CUE SP-IV

The efficient GMM estimator uses $\Phi_s(\beta, \zeta) = (\Sigma^{-1}_{u^\perp_H} \otimes Q^{-1})$:

$$\hat{\beta}_G = \left(R'\left(Y^\perp_H P_{Z^\perp} Y^{\perp\prime}_H \otimes \Sigma^{-1}_{u^\perp_H}\right) R\right)^{-1} R'\left(Y^\perp_H P_{Z^\perp} \otimes \Sigma^{-1}_{u^\perp_H}\right) \text{vec}(y^\perp_H P_{Z^\perp}). \tag{B.1}$$

The continuously updating (CUE) GMM estimator minimizes the AR statistic with respect to $b$. The KLM statistic is zero at the CUE estimator, so both AR and KLM confidence sets contain the CUE.

## Appendix C. Impact of Estimation Error on Inference

The expected Jacobian of (5) with respect to $d$ at $d = \zeta$ is:

$$\mathbb{E}\left[\frac{\partial z^\perp_t \otimes u^\perp_{H,t}}{\partial \zeta'}\right] = 0, \tag{C.5}$$

because $u^\perp_{H,t}$ and $z^\perp_t$ are orthogonal to $X_{t-1}$ by construction. Together with block-diagonal $\Phi$, this implies that **plugging in estimated forecast errors and using the standard formula yields a valid asymptotic variance, without further adjustment**.

## Appendix D. Proof of Proposition 2

The asymptotic variance of SP-IV is:
$$aV\!ar(\hat{\beta}) = (\Theta'_Y \Theta_Y)^{-1} \Theta'_Y \left(I_{N_z} \otimes \text{var}(u^\perp_{H,t})\right) \Theta_Y (\Theta'_Y \Theta_Y)^{-1}. \tag{D.1}$$

For 2SLS:
$$aV\!ar(\hat{\beta}_{2SLS}) = (\Theta'_Y \Theta_Y)^{-1} \Theta'_Y \Omega(z_{H,t} u_t) \Theta_Y (\Theta'_Y \Theta_Y)^{-1}, \tag{D.2}$$

where $\Omega(z_{H,t} u_t) = \sum_{l=-H+1}^{H-1} (I_{N_z} \otimes \iota_{-l}) \gamma_l$. Under stationarity each $H \times H$ block equals the autocovariance matrix $\Sigma_{u_H}$. The result follows.

## Appendix E. Proof of Proposition 3

Under weak instrument asymptotics $\Theta_Y = C/\sqrt{T}$:

- 2SLS concentration parameter: $\text{Tr}(CC')/(H N_z \sigma^2_\omega)$
- SP-IV without controls: $\text{Tr}(CC')/(N_z \text{Tr}(\Sigma_{v_H}))$
- SP-IV with controls: $\text{Tr}(CC')/(N_z \text{Tr}(\Sigma_{v^\perp_H}))$

$\text{Tr}(\Sigma_{v_H})$ is larger than $\text{Tr}(\Sigma_{v^\perp_H})$ unless $X_{t-1}$ is completely irrelevant.

---

## References

- Anderson, T.W., & Rubin, H. (1949). *Annals of Mathematical Statistics*, 20(1), 46–63.
- Andrews, I. (2016). *Econometrica*, 84(6), 2155–2182.
- Andrews, I., Stock, J.H., & Sun, L. (2019). *Annual Review of Economics*, 11(1), 727–753.
- Angeletos, G.-M., Collard, F., & Dellas, H. (2020). *American Economic Review*, 110(10), 3030–70.
- Barakchian, S.M., & Crowe, C. (2013). *Journal of Monetary Economics*, 60(8), 950–966.
- Barnichon, R., & Mesters, G. (2020). *Quarterly Journal of Economics*, 135(4), 2255–2298.
- Bauer, M.D., & Swanson, E.T. (2022). NBER WP 29939.
- Bekker, P.A. (1994). *Econometrica*, 62(3), 657–681.
- Bloom, N. (2009). *Econometrica*, 77(3).
- Cieslak, A. (2018). *Review of Financial Studies*, 31(9), 3265–3306.
- Coibion, O. (2012). *AEJ: Macroeconomics*, 4(2), 1–32.
- Cragg, J.G., & Donald, S.G. (1993). *Econometric Theory*, 9(2), 222–240.
- Del Negro, M., Lenza, M., Primiceri, G.E., & Tambalotti, A. (2020). *Brookings Papers*.
- Fieldhouse, A.J., & Mertens, K. (2023). FRB Dallas WP 2305.
- Galí, J., & Gambetti, L. (2020). Central Bank of Chile.
- Galí, J., & Gertler, M. (1999). *Journal of Monetary Economics*, 44(2), 195–222.
- Gertler, M., & Karadi, P. (2015). *AEJ: Macroeconomics*, 7(1), 44–76.
- Gilchrist, S., & Zakrajšek, E. (2012). *American Economic Review*, 102(4), 1692–1720.
- Han, C., & Phillips, P.C.B. (2006). *Econometrica*, 74(1), 147–192.
- Jarociński, M., & Karadi, P. (2020). *AEJ: Macroeconomics*, 12(2), 1–43.
- Jordà, Ò. (2005). *American Economic Review*, 95(1), 161–182.
- Jordà, Ò., & Kozicki, S. (2011). *International Economic Review*, 52(2), 461–487.
- Kilian, L., & Lütkepohl, H. (2017). *Structural VAR Analysis*. Cambridge UP.
- Kleibergen, F. (2002). *Econometrica*, 70(5), 1781–1803.
- Kleibergen, F. (2005). *Econometrica*, 73(4), 1103–1123.
- Kuttner, K.N. (2001). *Journal of Monetary Economics*, 47(3), 523–544.
- Lewis, D.J., & Mertens, K. (2022). FRB New York Staff Reports 1020.
- Li, D., Plagborg-Møller, M., & Wolf, C.K. (2021). arXiv 2104.00655.
- Lloyd, S., & Manuel, E. (2023).
- Mavroeidis, S., Plagborg-Møller, M., & Stock, J.H. (2014). *Journal of Economic Literature*, 52(1), 124–88.
- McLeay, M., & Tenreyro, S. (2019). *NBER Macroeconomics Annual*, 34.
- Mertens, K., & Ravn, M.O. (2013). *American Economic Review*, 103(4), 1212–47.
- Mikusheva, A. (2021). World Congress of Econometric Society.
- Miranda-Agrippino, S., & Ricco, G. (2021). *AEJ: Macroeconomics*, 13(3), 74–107.
- Montiel-Olea, J.L., & Pflueger, C. (2013). *JBES*, 31(3), 358–369.
- Montiel Olea, J.L., & Plagborg-Møller, M. (2021). *Econometrica*, 89(4), 1789–1823.
- Moreira, M.J. (2003). *Econometrica*, 71(4), 1027–1048.
- Nakamura, E., & Steinsson, J. (2018). *QJE*, 133(3), 1283–1330.
- Newey, W.K., & Windmeijer, F. (2009). *Econometrica*, 77(3), 687–719.
- Plagborg-Møller, M., & Wolf, C.K. (2021). *Econometrica*, 89(2), 955–980.
- Plagborg-Møller, M., & Wolf, C.K. (2022). *JPE*, 130(8), 2164–2202.
- Ramey, V. (2016). In Taylor & Uhlig (Eds.), *Handbook of Macroeconomics*.
- Romer, C.D., & Romer, D.H. (2004). *American Economic Review*, 94(4), 1055–1084.
- Rothenberg, T.J., & Leenders, C.T. (1964). *Econometrica*, 32(1/2), 57–76.
- Smets, F., & Wouters, R. (2007). *American Economic Review*, 97(3), 586–606.
- Stock, J., & Yogo, M. (2005). In Andrews (Ed.), *Identification and Inference*. Cambridge UP.
- Stock, J.H., & Watson, M.W. (2012). *Brookings Papers*, Spring 2012, 81–135.
- Stock, J.H., & Watson, M.W. (2018). *Economic Journal*, 128(610), 917–948.
- Stock, J.H., & Wright, J.H. (2000). *Econometrica*, 68(5), 1055–1096.
- Sun, Y. (2014). *Journal of Econometrics*, 178(P3), 659–677.
- Swanson, E.T. (2021). *Journal of Monetary Economics*, 118, 32–53.
