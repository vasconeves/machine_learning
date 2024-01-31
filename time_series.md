# Time Series

## Introduction

A **time series** is a special kind of statistical data. It is a collection of numerical measurements called **observations**

$$
x_1,x_2,...,x_t,...,x_n \in \mathbb{R}
$$

that are indexed by a time stamp $t=1,...,n$. The time stamps form a deterministic sequence and are regularly spaced in time with equal intervals between any two adjacent stamps. 

The observations are modeled mathematically as realizations of a corresponding series of random variables (r.v.),

$$
X_1,X_2,...,X_t,...,X_n : \Omega \rightarrow \mathbb{R}
$$

and are defined on some common probability space ($\Omega,P$). 

Then, what is observed is a particular outcome of a specialized probability model:

$$
x_1 = X_1(\omega),...,x_t = X_t(\omega),...,x_n = X_n(\omega),
$$

for an outcome $\omega in \Omega$ in some probability space ($\Omega,P$).

So, we have one r.v. for each time stamp and one observation for each r.v.. All r.v. are defined on a common probability space, so we can speak of probabilities of **joint events** that involve any number of these r.v. The realizations that come from the real world occur sequentially ($x_t$ before $x_{t+1}$), where the time interval between any observation is the same.

### Dependence in Time Series

The most important feature of time series data is that we make **no assumption about independence of these r.v.** In fact most time series are dependent., typically because past realizations influence future observations. The main goal of time-series analysis is to first model and then estimate from data (guided by a model) the dependence structure of these r.v.

**Statistical dependence in a time series is a double edged sword!** On the one hand, dependence **helps** us make predictions about the future realizations from knowledge of the past realizations. On the other hand, dependence poses technical challenges in the distributional analysis of the estimators, because there is less statistical information in dependent data about the data generating process **(e.g. The LLN and the CLT do not apply here!)**

Let's consider the following examples:

- **Economic data**: stock prices, inflation rate, GDP, emplyment rate, interest rate, exchange rates;
- **Biometric data**: heart rate, blood pressure, weight, fMRI;
- **Environmental data**: temperature, precipitation, pressure, humidity, pollution;
- **Sound data**: speech, music, pollution;

![](pics/johnson.png)
*Johnson & Johnson quarterly earnings per share*
![](pics/air_traffic.png)
*global air traffic passengers*
![](pics/exchange_rates.png)
*exchange rates GBP to NZ dollar*
![](pics/pressure_and_fish.png)
*Air pressure and fish population*
![](pics/price_index_usa.png)
*Price index USA*

In all these examples, data takes the form of discrete measurements of a real world phenomena that evolves continuously in time. A general probabilistic model to describe such phenomena is called a **stochastic process**, which is simply a collection of r.v. indexed by either a continuous or discrete time parameter $t$. A time series can therefore be considered as a **single realization** of a stochastic process. Each r.v. $X_t$ has a marginal distribution $P_t$. The process ${X_t}_{t>0}$ as a whole also has a probability law, which can be thought of as the **joint** distribution of all the $X_t$s.

## Deterministic dependencies in time series

Two of the most important deterministic dependence features in a time series are a **trend** and the **seasonal variation**. 

Let 

$$
\mu_X(t) = \mathbb{E}[X_t]
$$

denote the mean function of the marginal distributions of the series. We can also write the mean function as

$$
\mu_X(t) = m_X(t) + s_X(t),
$$

where $m_X(t)$ is the **trend** of the series, the non-constant and non-cyclical component of $\mu_X(t)$ and $s_X(t)$ is the **seasonal** variation of the series, the cyclical component of $mu_X(t)$ that repeats itself at a fixed time interval. 

### Example 0: White noise

A basic building block of time series models is the white noise process. The simplest example is the i.i.d. data $W_t \sim \mathcal{N}(0,\sigma^2)$.

![](pics/white_noise.png)

*Does the series above has a trend or seasonality?*

No! White noise does not have deterministic features.

We can now add a noise term to our model. Thus, we can write that

$$
X_t = m_X(t) + s_X(t) + W_t,
$$

where $W_t$ is a white noise time series. 

In the Figure below we have a time series with a seasonal variation and some noise. We can readily observe the cyclical pattern by visual inspection.

![](pics/seasonal1.png)

However, when we increase the noise term, it starts to be challenging to discern the deterministic pattern.

![](pics/seasonal2.png)

Now consider the following time series with a linear trend and some noise.

![](pics/linear1.png)

If we add more noise, the trend starts to be more difficult to observe.

![](pics/linear2.png)

The same applies to a time series with a quadratic trend and some noise...

![](pics/quadratic1.png)

and the same with more noise.

![](pics/quadratic2.png)

Again the same for an exponential trend.

![](pics/exponential1.png)

![](pics/exponential2.png)

Let's now mix both deterministic features.

In the plot below we have both features: a linear trend and a seasonal variation with some noise.

![](pics/deterministic1.png)

As we increase the noise, it is harder to discern the signal.

![](pics/deterministic2.png)

And yet another example, this time for a quadratic trend with a cyclical variation with some noise...

![](pics/deterministic3.png)

...and with a lot of noise!

![](pics/deterministic4.png)

Enough of examples!

## Stochastic dependence

Very often time series observations that are close to each other in time are **strongly correlated**.  For many, this correlation decays as the time distance between observations increases, while the variation of $X_t$ stays constant over time. For others, the correlation of $X_t$ with future observations stay constant, while the total variation of the series accumulates and increases with time. 

We can write this description as the **autocovariance** function of a time series. Therefore,

$$
\gamma_X(s,t) = Cov[X_s,X_t] = \mathbb{E}\left[ (X_s-\mu_X(s))(X_t-\mu_X(t))\right],
$$

decribes the linear stochastic association between the terms of the series. The **marginal variance** function

$$
\gamma_X(t,t) = Var[X_t] = \mathbb{E}\left[ (X_t-\mu_x(t))^2\right],
$$

describes the magnitude of the random fluctuations in the series at different time points. Realizations of time series with strong stochastic dependencies tend to look much smoother and more regular than the white noise process with the same marginal variance function.

## Objective of time series analysis

The **goal** of time series analysis is to understand and explore the deterministic and stochastic dependencies of the stochastic process that generates the data.

Specifically we need statistical tools and models to:

- Detect the trend $m_X(t).
- Detect the seasonal variation $s_X(t)$ and determine its period.
- Understand the correlation structure $\gamma_X(t,s)$ within the same time series.
- Understand the correlation structure between two different time series.

In many applications the important goal is to be able to forecast future observations.

Let's take a look at real data. *What properties does the series has?*

![](pics/airline.png)

Here we have air passenger data. We observe a linear trend, a seasonal variation of period 4 and an increase in variance over time. 

## Stationarity

In order to do statistical estimation and inference in time series we need certain technical conditions to be met. In other words, we need to insure that the observations along a single realization of the process are representative of all possible realizations of the process, so that we are able to estimate population parameters for the whole process (e.g. expectations, variances, correlations). Also, we need conditions that allow us to extrapolate statistical models fitted to observations from the past ntoto the future.

While time series data are not i.i.d it is possible to do inference via **time averages** along a sample path and allowing for stochastic dependencies. 

A time series $X_t$ is **strongly stationary** if the joint distribution of $X_t,...,X_{t+n}$ is the same as the joint distribution of $X_{t+h},...,X_{t+n+h}$,

$$
(X_t,...,X_{t+n}) = (X_{t+h},...,X_{t+n+h})
$$

for all integers $n$, time stamps $t$, and time shifters $h$.

**Weak stationarity** requires that only the first two moments of the series (i.e. the mean and the variance/covariance) be constant in time. Thus,

$$
\mathbb{E}[X_t] = \mu_X, Var(X_t) = \sigma_X^2, Cov(X_s,X_t) = \gamma_X(|s-t|),
$$

for all time stamps $s$ and $t$. A weakly stationary time series is simply called stationary.

Observations about stationarity:

1. Stationarity requires that the joint distribution of the series remains fixed throughout time. This implies that the stochastic dependencies in the series remain the same throughout time. Therefore, if we estimate a model based on observations $from t=1,...,n$, then stationarity allows us to use the fitted model to predict observations for $t>n$.
2. If, in addition, the stochastic dependents in the terms of the series dies down sufficiently quickly as the time gap between observations increases, we can use these observations tod o statistical estimation by relying on appropriate generalizations of the LLN and the CLT. In other words, if the observations are far enough from each other in time allowing them to be nearly independent from each other, then the path sample averages behave similarly to the sample averages of i.i.d r.v..
3. Stationarity rules out dependence of the series on the time index $t$. Therefore, it rules out trends and seasonal variation.

For these reasons it is always important to work with stationary time series. However, many time series are not stationary so it is necessary to transform them into a stationary one. This typically requires the use of regression methods or averaging techniques to remove the trend, seasonal variation and noise variation components. 

## Identifying and removing non-stationarity features

When we have a non-stationary time series we may observe one or all of the following features:

- Trend: non-constant expected value
- Periodical oscillations (seasonal effect)
- Non-constant variance
- Changes in the dependency structure

To detect these features we can plot the time series and plot its autocovariance. The autocovariance plot allows us not only to observe the trends and seasonal effects but also any changes in the dependency structure.

Let's consider the following techniques of transforming time series data:

- Linear regression $\rightarrow$ can remove a linear trend.
- Non-parametric regression model (e.g. kernel smoothing, polynomial fitting, k-nearest neighbors, series regression...).
  - can remove linear and non-linear trends as well as seasonal variations
- Periodic function as regression model $\rightarrow$ can remove seasonal variations.
- differencing the data one or more times $\rightarrow$ can remove linear and non-linear trends as well as increasing variances.
$$
Y_t = \nabla X_t = X_t - X_{t-1}
\rightarrow \text{removes linear trend} \\
$$
$$
\nabla^2 X_t = \nabla X_t - \nabla X_{t-1} = X_t - 2 X_{t-1} + X_{t-2} \rightarrow \text{removes quadratic trend}
$$
$$
\text{If } \{X_t\}_t \text{ is integrated of order p, then } \{\nabla^p X_t\}_t \text{ is stationary.}
$$
- applying a variance reduction transformation (square root or log transformation) $\rightarrow$ can remove increasing variance.
- smoothing (i.e. applying moving averages).
- applying fourier analysis $\rightarrow$ can remove seasonal trends.

**Note: after the transformation the remainder of the data should be stationary, with mean zero.**

### Example

Let's observe the following plot on the left. Here we can detect a linear trend and a seasonal variation by visual inspection. 

On the right side we have two plot showing the application of two distinct  techniques. On the top plot we have the result of detrending by linear regression and on the bottom the result of differencing.

![](pics/detrending.png)

We can observe that, while both methods were successful in removing the linear trend, in this case, the differencing also removed other patterns from the data. *So, in general, which method should we choose?* Well, it depends!

Let's check out the pros and cons of each method. The decomposition or detrending of the data allows a much greater control of the process because we know where each component lies. In other words, it is very interpretable. When we do differentiation we lose that interpertability. On the other hand, in the first method you need to parametrize and do some choices, while in the differentiation there's no need of that. Also, multiple differencing shortens your time series, careful.

## Estimation on stationary time series

Let $X_t$ denote a stationary time series. We obtain estimators of the mean, variance and autocovariance functions by replacing the expectations with sample averages. Therefore,

$$
\hat{\mu} = \frac{1}{n}\sum_{t=1}^n X_t
$$
$$
\hat{\sigma}^2 = \frac{1}{n}\sum_{t=1}^n(X_t-\hat{\mu})^2
$$
$$
\hat{\gamma}(h) = \frac{1}{n}\sum_{t=1}^{n-h}(X_t-\hat{\mu})(X_{t+
h}-\hat{\mu}) \text{ for } 1\le h \le n.
$$

If the series is stationary, then each observation in the sample average contributes with statistical information about the common parameters. Most time series are dependent, but when the stochastic dependences in the series decay sufficiently fast, as the time distance between terms gets large, then the sample averages have a similar asymptotic behavior as in LLN and CLT for i.i.d. data. 

We can also write the sample autocorrelation function as

$$
\hat{\rho}(h) = \hat{\gamma}(h)/\hat{\gamma}(0) \text{ for } 1\le h \le n.
$$

**Thus, under mild technical conditions we have good estimators of the mean, variance and autocovariance functions of a stationary time series.**

## Autocorrelation as a diagnostic tool

The autocovariance function (ACF) is a very powerful statistical tool to study the dependence properties of a time series. Visualizing the ACF is the second step after visualizing the series itself. 

Properties of the ACF:

- symmetric.
- measures *linear* dependence of $X_t,X_s$.
- relates to smoothness
- for weakly stationary series: $\gamma_X(t,t+h) = \gamma_X(0,h) = \gamma_X(h)$.

### Example

In the following diagram we can observe a very simple example. We want to plot the ACF of a signed time series. The first step is to calculate the autocovariance function of all possible step sizes $h$, from $h=1$ to $h=4$. We note that the average of the time series is zero.

![](pics/acf_example1.png)

From here we can draw the "correlogran" as shown below which is just the value of the autocovariance as a function of the lag $h$.

![](pics/acf_example2.png)

This diagram shows a negative component (-3/7) at lag $h=2$ and a positive component (2/7) at lag $h=4.

Let's take a look at another example, this time a simulation with short term correlations and noise. The top plot shows the time series while the bottom plot shows the ACF. The horizontal dashed blue lines represent the noise levels.

![](pics/acf_example3.png)

Here we can observe that this time series exhibits short-term correlations which is often the case in stationary series. Also, after $h=5$ the longer term correlations are indistinguishable from the noise.

In the case of a series with a trend as shown below, the case is different.

![](pics/acf_example4.png)

Here, the ACF shows correlations throughout the $h$ range. This means that the points are all correlated and this indicates an existence of a trend which is obviously the case.

In the case where we don't have any trend but only a seasonal variation, such as the case of the de-trended $CO_2$ Mauna Loa data shown below.

![Alt text](pics/acf_example5.png)

In the ACF plot we observe a cyclical cycle which is the pattern for the existence of a seasonal variation in the time series.

The next picture shows three cases of a time series (left) and the corresponding ACF (right). This shows that through ACF we can make precise diagnostics on the non-stationarity nature of the signal components in our data.

![](pics/acf_example6.png)


On the top panel, the ACF signal steadily decreases with time lag, revealing a mixture between a trend and a seasonal variation. In the middle panel, the time series was already detrended but a seasonal variation persists in the data. On the bottom plot there is no evidence of non-stationarity components.

## The white noise model (revisited)

The simplest time series model is the **white noise process** $\{W_t\}_t$ of r.v. that have zero mean, the same variance $\sigma_W^2$, and zero correlations. Therefore,

$$
\mu_W(t) = \mathbb{E}[W_t] = 0
$$
$$
\gamma_W(t,t) = Var(X_t) = \sigma_W^2
$$
$$
\gamma_W(t,s) = Cov(X_t,X_s) = 0, \text{ for t } \ne s.
$$

Also,

- It is oftern i.i.d. e.g. Gaussian
- Stationary
- Checking for white noise: $\hat{\rho}(h) is approximately \mathbb{N}(0,\frac{1}{n}) under mild conditions.


The path of a white noise process and its ACF function can be represented by the following figure

![](image.png)

Here we observe that the full strength of the ACF signal is concentrated when there is no correlation ($Lag=0$).

The main purpose of the white noise is to model the "best" case residuals that contain no information after we perform all possible procedures in our data. 

The distribution of the estimator of a white noise source is

$$
\hat{\gamma}_W(h) \sim \mathcal{N}\left(0,\frac{\sigma_W^2}{n}\right)
$$

which means that we do not expect to see the theoretical ACF function exactly as our estimate but only approximately up to estimation error.

## Autoregressive model

A time series \{X_t\}_t is an **autoregressive process** of order $p$, denoted AR(p) if

$$
X_t = \phi_1X_{t-1} + \phi_2X_{t-2} + ... + \phi_pX_{t-p} + W_t,
$$

where $\{W_t\}_t$ is a white noise process, and $W_t$ is uncorrelated with $X_s$ for $s < t$.

**Note:** the definition of the model is recursive, meaning we can relate $X_t$ to any previous term of the series $X_{t-h} by substituting the above expression for X_{t-1} on the right side of the equation and so on. Because of this recursive nature, **all** terms of the series are **dependent**. This fact is reflected on the ACF. The ACF of a **stationary autoregressive process** is **non-zero** for all time shifters $h$ and decays to zero exponentially as $h$ increases, as shown in the plot below.

![](pics/models_autoregression.png)

Let's see another example. The following autoregressive model 

$$
X_t = 1.5X_{t-1} - 0.75X_{t-2} + W_t
$$

generates the following ACF. Again, we observe the exponentially decay as $h$ increases.

![](pics/models_autoregression2.png)


## Random walk model

A time series $\{X_t\}_{t\ge 1}$ is a random walk if the value of $X_t$ is obtained from the value of $X_{t-1}$ by adding a random perturbation $W_t$ (white noise) that is independent of the past history of the series $\{X_s\}_{s<t}$. Thus,

$$
X_t = X_{t-1} + W_t
$$

A time series $\{Y_t\}_{t\ge 1}$ is a random walk **with drift** if it is equal to the sum of a random walk process with a deterministic linear trend. Therefore,

$$
Y_t = \delta t + X_t = \delta + Y_{t-1} + W_t,
$$

where $Y_{t-1} = \delta(t-1) + X_{t-1}$.

In the following plot we can observe the differences between the random walk model with and without drift.

![](pics/models_random_walk.png)

Due to the inherent random nature of white noise, all random walks are different as shown in the figure below.

![](pics/models_random_walk1.png)
![](pics/models_random_walk2.png)

### Statistic of random walk

To compute the basic statistics of the random walk it is useful to write $X_t$ as a sum of perturbations that accumulate over time.

$$
X_t = X_{t-1} + W_t
\\
= \left[X_{t-2} + W_{t-1} \right] + W_t
\\
\vdots
\\
= X_0 + \sum_{h=1}^t W_h
$$

Similarly, for the random walk with drift we have

$$
Y_t = \delta + Y_{t-1} + W_t
\\
= \delta + \left[\delta + Y_{t-2} + W_{t-1} \right] + W_t
\\
\vdots
\\
\delta t + Y_0 + \sum_{h=1}^t W_h.
$$

Using these representations we can find the marginal mean function, the covariance function and the autocorrelation function.

$$
\mu_X(t) = \mathbb{E}[X_t] = \mathbb{E}\left[X_0 + \sum_{h=1}^t W_h\right]
$$
$$
\mu_X(t) = \mathbb{E}[X_0].
$$
$$
\sigma_X^2(t) = Var(X_t) = Var\left(X_0 + \sum_{h=1}^t W_h\right) 
$$
$$
= Var(X_0) + \sum_{h=1}^t \left[2Cov(X_0,W_h) + Var(W_h)\right] + 2 \sum_{1 \le h < j \le t} Cov(W_h,W_j) 
$$
$$
\sigma_X^2(t) = Var(X_0) + t\sigma_W^2,
$$

since $W_h$ is uncorrelated with $X_0$ and with $W_j$ for $j \ne h$.

$$
\gamma_X(s,t) = Cov(X_s,X_t) = Cov\left(X_0 + \sum_{h=1}^s W_h, X_0+\sum_{h=1}^t W_h\right) 
$$
$$
= Var(X_0) + \sum_{h=1}^{min(s,t)} Var(W_h)
$$
$$
\gamma_X(s,t) = Var(X_0) + min(s,t)\sigma_W^2.
$$

**Note: The random walk is not stationary because the variance is growing with time and the autocovariance depends on the smallest of the two time stamps rather than on the difference.**

**Note2: However $\nabla X_t$ *is* stationary.**

## Moving average model

A time series \{X_t\}_t is a **moving average process** of order $q$, denoted by MA(q) if it can be represented as a weighted moving average

$$
X_t = W_t + \theta_1 W_{t-1} + \theta_2 W_{t-2} + ... + \theta_q W_{t-q}
$$

$$
X_t = \sum_{h=0}^q \theta_h W_{t-h}
$$

of a white noise series $\{W_t\}_t$.

The following figure shows three plots (top part) and corresponding ACFs(bottom part). The plot on the left is a representation of white noise, the middle plot depicts a moving average model of order 1 and on the left we can observe a representation of a moving average model of order 7. 

![](pics/models_MA.png)

From the definition of a moving average time series we can write that the autocovariance function is

$$
\gamma_X(h) = Cov\left(\sum_{j=0}^q \theta_j W_{t-j}, \sum_{k=0}^q \theta_k W_{t+h-k} \right)
$$
$$
\gamma_X(h) = \sum_{j=0}^{q-h} \theta_j\theta_{j+h}\sigma_W^2, \text{ for } 0 \le h \le k
$$

because $W_j$ is uncorrelated with $W_k$ for $j \ne k$.

**Notes:**

- $\mathbb{E}[X_t] = 0$.
- Autocovariance $\gamma$ depends only on $|s-t| \implies$ stationarity.
- ACF reflects order: $\gamma(s,t) = 0 if |s-t| > q.
- ACF distinguishes MA and AR models as shown in the figure below.

![](pics/acf_comparison.png)

## ARMA Model

A time series $\{X_t\}_{t \ge l}$ is a **moving average autoregressive process** of orders $p,q$, denoted by ARMA(p,q), if it is a sum of an AR(p) component with a MA(q) component. Thus

$$
X_t = \phi_1X_{t-1} + \phi_2X_{t-2} + ... + \phi_pX_{t-p} + W_t + \theta_1 W_{t-1} + \theta_2 W_{t-2} + ... + \theta_q W_{t-q}
$$

A time series $\{X_t\}_{t \ge l}$ is an ARIMA(p,d,q) model if the difference of order d, \{\nabla^d X_t\}_{t\ge l} is an ARMA(p,q) model.

And so on...

## Regression and time series

We can always try and fit a regression in our time series. Such a model can be written as

$$
X_t = \beta_1 z_{t1} + \beta_2 z_{t2} + ... + W_t = \mathbf{\beta^Tz_t} + W_t
$$

Examples:

- linear trend $\implies X_t = \beta_1 + \beta_2 t + W_t$ 
- AR(2) model $\implies X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + W_t$
- include external regressors $\implies X_t = \beta_1 X_{t-1} + \beta_2 Y_t + W_t$

To fit any model we can do least squares regression: $min_\beta\sum_t (mathbf{x_t-\beta^T z_t})^2

**Note: the errors may be correlated over time!!!**

**Note2: this is also valid for any non-linear models**

### ACF as a diagnostic tool

*Which external variables should we use? What should be the order of the model?*

We can use ACF (and partial ACF) to determine the fit. The end result should look like white noise.

Example. $X_t = T_t + Y_t$

- $T_t$ = 50+3t (linear trend)
- $Y_t$ = 0.8Y_{t-1} + W_t$ (AR(1) model)

The plot below shows the result after fitting the linear trend only. The ACF shows the typical signal of an AR model!

![](pics/acf_diag1.png)

### Fitting autoregressive models

For a given autoregressive order $p$, the AR(p) model has $p+1$ parameters that need to be estimated from data: $\phi_1,\phi2,...,\phi_p,\sigma_W^2$.

We can estimate these parameters using the method of moments approach. The first step in estimation is to compute the autocovariance

$$
\hat{\gamma}_X(0),\hat{\gamma}_X(1),...,\hat{\gamma}_X(p).
$$

The second step is to find the $p+1$ equations that relate these moments to the unknown parameters above

$$
\hat{\gamma}_X(0),\hat{\gamma}_X(1),...,\hat{\gamma}_X(p) = \Gamma\left(\phi_1,\phi2,...,\phi_p,\sigma_W^2\right)
$$

we will then have $p+1$ equations with $p+1$ unknowns, which in general will have a unique solution

$$
(\phi_1,\phi2,...,\phi_p,\sigma_W^2) = \Gamma^{-1}\left(\hat{\gamma}_X(0),\hat{\gamma}_X(1),...,\hat{\gamma}_X(p)\right)
$$

The equations $\Gamma$ are known as the **Yule-Walker** equations. They can be expressed in the following way:

$$
\gamma_X(0) = \phi_1\gamma_X(1) + \phi_2\gamma_X(2) + ... + \phi_p\gamma_X(p) + \sigma_W^2
$$

$$
\gamma_X(1) = \phi_1\gamma_X(0) + \phi_2\gamma_X(1) + ... + \phi_p\gamma_X(p-1)
$$

$$
\vdots
$$

$$
\gamma_X(p) = \phi_1\gamma_X(p-1) + \phi_2\gamma_X(p-2) + ... + \phi_p\gamma_X(0)
$$

In matrix form we have

$$ 
\mathbf{\phi = \Gamma_p^-1\gamma_p}.
$$

### AR(p) model order determination

The following figure shows the ACF of the white noise model (left), the MA model (center) and the AR model (right).

![](pics/acf_comparison2.png)

We know that in the case of the MA(q) models, the ACF reveals the order $q$. Basically it is only necessary to count the number of time steps where the ACF is above the noise level.

For AR(p) models however this is not so clear, as the function decays exponentially fast. We need another form of diagnostic.

#### Partial autocorrelation

The best method to determine the order is to use the partial autocorrelation function.

Let X_0,...,X_n be a stationary time series. The autocorrelation function at lag $h$ is defined as

$$
\rho_x(h) = Corr(X_h,X_0) = \mathbb{E}[(X_h-\mathbb{E}[X_0])(X_0-\mathbb{E}[X_0])]/Var(X_0).
$$

The partial autocorrelation between $X_h$ and $X_0$ is the correlation between $X_h$ and $X_0$ with the correlation due to the intermediate terms of the series $X_1,...,X_{h-1}$ removed. 

Formally we can write that the partial autocorrelation of time series $X_t$ at lag $h$ is

$$
\alpha_X(h) = Corr(X_h-\hat{X}_h^{lin_{h-1}},X_0-\hat{X}_0^{lin_{h-1}}),
$$

where $\hat{X}_h^{lin_{h-1}}$ is the linear regression (projection) of $X_h$ on $X_1,...,X_{h-1}$ and $\hat{X}_0^{lin_{h-1}}$ is the linear regression of $X_0$ on $X_1,...,X_{h-1}$. 

The following shows a comparison between the ACF (left) and the PACF for AR(2). We can observe that in the PACF the two orders are clearly displayed. 

![](pics/pacf1.png)

Let's take a look at a previous example, where we have a time series composed of two series: 

$$
X_t = T_t + Y_t
$$

where

$$
T_t = 50+3t \implies \text{ time series with a linear trend}
$$

and

$$
Y_t = 0.8Y_{t-1} + W_t \implies \text{ AR(1) series}
$$

The following figure shows the time series (top plot) as well as the ACF (bottom left plot) and the PACF (bottom right plot) of the residuals after fitting a linear model only. From the ACF we can conclude that we have a AR time series (exponential decay, oscillation) and in the PACF plot we only observe one peak above noise. Therefore, we should fit a AR(1) model.

#### Akaike information criterion (AIC)

There are also other ways to determine which model is more apropriate among a few model candidates. Here we will explore two other more generic approaches. The first one is the Akaike Information Criterion or AIC. It is calculated as

$$
AIC = 2k-2ln(L),
$$

where $k$ is the number of parameters in the model, $n$ is the number of the observations in the dataset, and $L$ is the likelihood value of a given dataset.

Models with smaller AIC values are preferred to the models with larger AIC values, as the smaller values are associated with a smaller number of model parameters (i.e. less complexity) and a better fit to the data (larger likelihood value). It is the interplay between these values that will determine the best model.

Another commonly used information criterion is the Bayesian information criterion (BIC) and can be calculated as

$$
BIC = kln(n)-2ln(L).
$$

The equation parameters are the same as in AIC, but a lower weight is given to the number of parameters.

#### Cross validation

Cross validation can be used in special cases for AR models (Bergmeier, Hyndman & Koo 2015). 

*Which technique to use? An information criterion (IC) or CV?*

- IC is more prefereable if the speed is priority
- IC is more preferable if the sample size is small
- CV is more preferable if model selection results are different with different ICs

**Note: minimizing AIC is asymptotically equivalent to leave-one-out CV (Stone 1977). See also Shao (1997).**

## Forecasting with AR(p) models

*Can we predict $X_{n+m}$ based on observed $x_n,x{n-1},...,x1$?*

**Yes!...but only for short horizons $m$!** For long horizons, the forecast converges to the mean of the time series.

1. We estimate the coefficients $\hat{\phi}_1,...,\hat{\phi}_p$
2. Estimate the steps ahead of the time series
   - 1 step ahead $\implies \hat{x}_{n+1|n} = \hat{\phi}_1x_n + \hat{\phi}_2 x_{n-1} + ... + \hat{\phi}_p x_{n-p+1}$
   - 2 steps ahead $\implies \hat{x}_{n+2|n} = \hat{\phi}_1x_{n+1} + \hat{\phi}_2 x_{n} + ... + \hat{\phi}_p x_{n-p+2}$
   - general $\implies \hat{x}_{n+m|n} = \hat{\phi}_1x_{n+m-1} + \hat{\phi}_2 x_{n+m-2} + ... + \hat{\phi}_p x_{n+m-p}$

where we use $x_t$ instead of $\hat{x}_t$ where available.

The following plot shows an example of a forecast and the time frame of the convergence towards the mean.

![](pics/ar_forecast.png)

## Fitting a time series: overview

1. Transform to make it stationary
   - log-transform
   - remove trends/seasonality
   - differentiate successively
2. Check for white noise (ACF)
3. If stationary: plot autocorrelation
   - If finite lag, fit MA (ACF gives order)
   - Otherwise fit AR

**Fitting AR(p)**

1. compute PACF to 
2. get order
3. estimate coefficients $\phi_k$ and noise variance $\sigma_W^2$ via Yule-Walker equations
4. Compute residuals, test for white noise

**Fitting MA(q)**

1. compute ACF to get order
2. estimate coefficients via maximum likelihood
3. compute residuals, test for white noise.

**Fitting ARMA(p,q)**

1. attempt to fit an AR model, compute residuals
2. attempt to fit an MA model to residuals (or original data)
3. fit ARMA(p,q) using p,q, determined in steps 1 and 2
4. compute residuals, test for white noise

**Review table**

![](pics/time_series_table.png)

## Linear process

The MA and the AR models can be related to each other with the concept **linear process**. Linear process models can be written as

$$
X_t = \sum_{j=-\infty}^\infty \psi_j W_{t-j}.
$$

For the process to be well defined, $\sum_j ||\psi_j|| < \infty$.

A linear process is called **causal** if $\psi_j=0$ whenever $j<0$. This is to say that the value of $X_t$ depends only on the past!

A linear process is weakly stationary, where

$$
\mathbb{E}[X_t] = 0
$$

and

$$
\gamma_X(t,t+h) = \sum_{i=-\infty}^\infty \psi_i\psi_{i+h}\sigma_w^2,
$$

which only depends on the length of the gap $h$.

This means that:

- MA(q) is a linear process and causal.
- AR(p) is stationary and causal if linear process converges (it converges if $|\phi_1|<1$)

## Local linear regression

- No time series and time series (TBD)

# Time series example: Stock price forecasting

Our research question here is the following: *How well can we predict Meta's closing stock price one day in advance and one month in advance?*

and

*Given the historical series of Meta's closing stock price, what is the best estimator of the stock price one day ahead or one month ahead? What is the test-sample mean squared error of each estimate?*

## Data set: stock prices

Our data is sourced from Yahoo finance, which maintains a python package to retrieve daily stock prices.

The following code loads Meta's stock prices since 2012-05-18. In the table below we can view a sample of Meta's stock series which includes the date, the daily price variation (opening session, highest value, lowest value and closing value). The last column shows the daily stock volume transactions.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import yfinance as yf
url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
print("All available NASDAQ listings: \n", companies.head())
fb_series = yf.download("META",start='2012-05-18',progress=False)
print("Sample of Meta stock series: \n", fb_series.head())
price = fb_series.reset_index()['Close'].values.reshape(-1,1)
date = fb_series.reset_index().Date.dt.date.values.reshape(-1,1)
plt.plot(date, price)
plt.title("Meta stock price over time")
plt.show()

All available NASDAQ listings: 
   Symbol                                       Company Name  \
0   AAIT  iShares MSCI All Country Asia Information Tech...   
1    AAL                      American Airlines Group, Inc.   
2   AAME                      Atlantic American Corporation   
3   AAOI                      Applied Optoelectronics, Inc.   
4   AAON                                         AAON, Inc.   

                                       Security Name Market Category  \
0  iShares MSCI All Country Asia Information Tech...               G   
1       American Airlines Group, Inc. - Common Stock               Q   
2       Atlantic American Corporation - Common Stock               G   
3       Applied Optoelectronics, Inc. - Common Stock               G   
4                          AAON, Inc. - Common Stock               Q   

  Test Issue Financial Status  Round Lot Size  
0          N                N           100.0  
1          N                N           100.0  
2          N                N           100.0  
3          N                N           100.0  
4          N                N           100.0  
Sample of Meta stock series: 
                  Open       High        Low      Close  Adj Close     Volume
Date                                                                        
2012-05-18  42.049999  45.000000  38.000000  38.230000  38.230000  573576400
2012-05-21  36.529999  36.660000  33.000000  34.029999  34.029999  168192700
2012-05-22  32.610001  33.590000  30.940001  31.000000  31.000000  101786600
2012-05-23  31.370001  32.500000  31.360001  32.000000  32.000000   73600000
2012-05-24  32.950001  33.209999  31.770000  33.029999  33.029999   50237200

```

The plot generated with the code depicts Meta's stock price over time as shown below.

![](pics/stock1.png)

## Scaling the data

*Do we need to scale the data?* 

In this case yes, because as we can see in the plot the degree of the variance is not constant over time and this violates one of the assumptions of stationarity. We can correct this by taking a log transformation of the data. 

```python
# Applying log transformation
log_price = np.log(price)
plt.plot(date, log_price)
plt.title("log of FB stock price over time")
plt.show()
```

As shown in the plot below, the variance now looks more constant.

![](pics/stock2.png)

## Information criteria

As seen before we will use the AIC and the BIC in the form of the functions depicted in the code. 

```python
# Example here about how to find; what my assumption is behind this 
from scipy.stats import norm 
def evaluate_AIC(k, residuals):
  """
  Finds the AIC given the number of parameters estimated and 
  the residuals of the model. Assumes residuals are distributed 
  Gaussian with unknown variance. 
  """
  standard_deviation = np.std(residuals)
  log_likelihood = norm.logpdf(residuals, 0, scale=standard_deviation)
  return 2 * k - 2 * np.sum(log_likelihood)
def evaluate_BIC(k, residuals):
  """
  Finds the AIC given the number of parameters estimated and 
  the residuals of the model. Assumes residuals are distributed 
  Gaussian with unknown variance. 
  """
  standard_deviation = np.std(residuals)
  log_likelihood = norm.logpdf(residuals, 0, scale=standard_deviation)
  return k * np.log(len(residuals)) - 2 * np.sum(log_likelihood)
```

## Linear model

Let's start with a linear model. The code outputs the MSE,AIC and BIC as shown below.

```python
from sklearn import linear_model
clf = linear_model.LinearRegression()
index = fb_series.reset_index().index.values.reshape(-1,1)

clf.fit(index, log_price)
print(clf.coef_) # To print the coefficient estimate of the series. 
linear_prediction = clf.predict(index)
plt.plot(date, log_price, label='log stock price (original data)')
plt.plot(date, linear_prediction, 'r', label='fitted line')
plt.legend()
plt.show()
linear_residuals = log_price - linear_prediction
plt.plot(date, linear_residuals, 'o')
plt.show();
print("MSE with linear fit:", np.mean((linear_residuals)**2))
print("AIC:", evaluate_AIC(1, linear_residuals))
print("BIC:", evaluate_BIC(1, linear_residuals))

MSE with linear fit: 0.12649596448928904
AIC: 2268.3173142051555
BIC: 2274.304159106317
```

The code also produces two plots: the log stock price versus time plot and its residuals as shown below.

![](pics/stock3.png)

From the residuals we can observe the need of a higher order model.

![](pics/stock4.png)

## Quadratic model

Let's try a model of order 2. 

```python
## After linear fit, it seems like a higher order model is needed
from sklearn import linear_model
clf = linear_model.LinearRegression()
index = fb_series.reset_index().index.values.reshape(-1,1)

new_x = np.hstack((index, index **2))
clf.fit(new_x, log_price)
print(clf.coef_) # To print the coefficient estimate of the series. 
quad_prediction = clf.predict(new_x)
plt.plot(date, log_price, label='log stock price (original data)')
plt.plot(date, quad_prediction, 'r', label='fitted line')
plt.legend()
plt.show()
quad_residuals = log_price - quad_prediction
plt.plot(date, quad_residuals, 'o')
plt.show();
print("MSE with quadratic fit:", np.mean((quad_residuals)**2))
print("AIC:", evaluate_AIC(2, quad_residuals))
print("BIC:", evaluate_BIC(2, quad_residuals))

MSE with quadratic fit: 0.05321548541531681
AIC: -277.04566883374355
BIC: -265.07197903142077
```

There is an huge improvement in the AIC and BIC criteria. The fitted plot as well as the residuals are shown below.

![](pics/stock5.png)

![](pics/stock6.png)

Thus, the quadratic fit is a satisfactory model for the general trend. 

## Examining periodicities with ACF/PACF analysis

Let's now move one and analyze the remaining signal with ACF/PACF methods, as shown in the following code.

```python
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(quad_residuals, lags=30)
plt.show()
sm.graphics.tsa.plot_pacf(quad_residuals, lags=30)
plt.show()
```

The ACf plot shows a exponentially decaying signal (remember, we're analysing a log transformed signal, so an exponentially decaying signal will be transformed into a negative linear trend)

![](pics/stock7.png)

The PACF plot shows a strong first term, which is evidence of an AR(1) component.

![](pics/stock8.png)


It's important to model these terms, because AR/MA can change their error structure. It can also change how we make forecasts, and our confidence bands on those forecasts.

## Simulated data

To review the AR/MA models, we're going to an exercise with simulated data first. 

### ARMA Models

Recall the general form of an AR model of order F

$$
X_t = \phi_1X_{t-1} + \phi_2X{t-2} + ... + \phi T_{t-p} + W_k,
$$

and the form of an MA model

$$
X_t = W_k + \theta_1W_{t-1} + \theta_2W_{t-2} + ... + \theta_qW_{t-q},
$$

where $W_t$ is the white noise that is uncorrelated with any lagged or future values.

### Simulation

First, we need to simulate the dataset with random noise and AR/MA terms.

The condition that the characteristic polynomial must have roots outside the unit circle is used here. For an order-$p$ time series $X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} \dots + \phi_p X_{t - p}$, there is a **characteristic polynomial** $\Phi(X_t) := 1 - \sum_{i=1}^p \phi_i X_{t-i}$. 

Note that we can equivalently express this polynomial as $\Phi(X_t) = \prod_{i=1}^p (1 - \alpha_i L) X_t$ where $L$ is the lag operator (i.e. $L(X_t) = X_{t-1}$.
By fixing all $\alpha_i = R$, for some constant $R \in (-1,1)$, we guarantee the time series is stationary. 

See [this link](https://math.unm.edu/~ghuerta/tseries/week4_1.pdf) for more details.

The following code shows the generating function for the simulation.

```py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from scipy.special import comb
import statsmodels.api as sm
def sim_dataset(AR, MA, t=100, trend = None, polynomial_root = -0.5, MA_weight = 0.5):
  # Note: AR=AM=0 will output white noise
  """
  Simulates a dataset given AR or MA order. 
  Selects AR terms so that the polynomial root is the given value;
  as long as the root is within (-1, 1), the series will be stationary. 
  """
  if trend is None: 
    trend = lambda x: 0
  arparams = np.array([comb(AR, i)*(polynomial_root)**(i) for i in range(1, AR + 1)])
  maparams = np.array([MA_weight] * MA)
  ar = np.r_[1, arparams] # add zero-lag 
  ma = np.r_[1, maparams] # add zero-lag
  arma_process = sm.tsa.ArmaProcess(ar, ma)
  print("ARMA process is stationary: ", arma_process.isstationary)
  print("ARMA process is invertible: ", arma_process.isinvertible)
  y = arma_process.generate_sample(t)
  y = np.array([_y + trend(j) for j, _y in enumerate(y)])
  return y
```

#### White noise

Let's start by generating white noise with our function

```py
np.random.seed(1234) #to observe the same output
white_noise = sim_dataset(0, 0, 100) #AR=0,AM=0 --> white noise
plt.plot(white_noise)
plt.show()
sm.graphics.tsa.plot_acf(white_noise, lags=15)
plt.show()
sm.graphics.tsa.plot_pacf(white_noise, lags=15)
plt.show()
```

The previous code generates 3 plots. The first plot shows randomly generated white noise.

![](image.png)

The second and third plots show the ACF and the PACF respectively. We can observe that there appears to be only white noise.

![](image-1.png)

![](image-2.png)

### Adding an AR component

Let's now add an AR component to the data. We should expect to observe an exponentially decaying pattern in ACF plot and significant terms showing up in the PACF plot.

```py
np.random.seed(1234)
AR_series = sim_dataset(2, 0, 250, polynomial_root = -0.8)
plt.plot(AR_series)
plt.show()
sm.graphics.tsa.plot_acf(AR_series, lags=15)
plt.show()
sm.graphics.tsa.plot_pacf(AR_series, lags=15)
plt.show()
```

The first generated plot shows the data. The ACF/PACF plots are depited in the second and third figures respectively.

![](image-3.png)

The ACF plot shows the exponential decay. The PACF plot shows the existince of two significant peaks other that the one at $lag=0$. Therefore, this time series must be an AR of order 2.

### Adding an MA component

Now we will test adding a MA component of order 3 to the simulation.

```py
np.random.seed(1234)
MA_series = sim_dataset(0, 3, 250, MA_weight=0.9)
plt.plot(MA_series)
plt.show()
sm.graphics.tsa.plot_acf(MA_series, lags=15)
plt.show()
sm.graphics.tsa.plot_pacf(MA_series, lags=15)
plt.show()
```

The first generated plot is the time series. The second plot is the ACF, where we observe significant terms up to the order of the ACF. Inm the PACF we observe exponentially decaying terms.

### ARMA model

Here we will observe exponentially decaying terms in both ACF and PACF. The usual plots were generated with the code below

```py
np.random.seed(1234)
ARMA_series = sim_dataset(1, 2, 250, polynomial_root = -0.5)
plt.plot(ARMA_series)
plt.show()
sm.graphics.tsa.plot_acf(ARMA_series, lags=15)
plt.show()
sm.graphics.tsa.plot_pacf(ARMA_series, lags=15)
plt.show()
```

![](image-4.png)

The ACF plot shows a 2 or 3 order MA

![](image-5.png)

The PACF plot we can observe a strong signal other than the one at $lag=0$ which indicates an AR of order 1 or or 3.

![](image-6.png)

When we have ARMA signals it can be difficult to estimate its parameters. Luckily we can use python packages to solve this problems quickly and efficiently.

## Fiting ARIMA MODELS

From `statsmodels.tsa.arima.model` we can import `ARIMA` which will use the `SARIMAX` algorithm to obtain the parameters for AR and MA.

In this case we will fit an autoregressive time series of order 2 only as shown in the code below. The code will first output a summary of the fitting and then MSI,AIC and BIC.

```py
from statsmodels.tsa.arima.model import ARIMA
AR_order = 2
ar_higher = ARIMA(AR_series, order=(AR_order, 0, 0)).fit() #AR_series was defined before
print(ar_higher.summary())
ar_higher_predictions = ar_higher.predict()
ar_higher_residuals = AR_series - ar_higher_predictions
ar_higher_residuals = ar_higher_residuals # Fitting AR 1 model means removing one observation
plt.plot(AR_series, label='original data')
plt.plot(ar_higher_predictions, 'r', label='fitted line')
plt.legend()
plt.show()
plt.plot(ar_higher_residuals, 'o')
plt.show()
print("MSE with AR(1) model:", np.mean(ar_higher_residuals**2))
print("AIC with AR(1) model:", evaluate_AIC(AR_order + 1, ar_higher_residuals))
print("BIC with AR(1) model:", evaluate_BIC(AR_order + 1, ar_higher_residuals))

                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  250
Model:                 ARIMA(2, 0, 0)   Log Likelihood                -351.516
Date:                Mon, 23 May 2022   AIC                            711.032
Time:                        20:26:33   BIC                            725.118
Sample:                             0   HQIC                           716.701
                                - 250                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.2190      1.508      0.808      0.419      -1.737       4.175
ar.L1          1.5431      0.058     26.696      0.000       1.430       1.656
ar.L2         -0.5853      0.057    -10.249      0.000      -0.697      -0.473
sigma2         0.9599      0.080     11.935      0.000       0.802       1.118
===================================================================================
Ljung-Box (L1) (Q):                   0.03   Jarque-Bera (JB):                 3.77
Prob(Q):                              0.85   Prob(JB):                         0.15
Heteroskedasticity (H):               0.94   Skew:                            -0.24
Prob(H) (two-sided):                  0.79   Kurtosis:                         3.37
===================================================================================
MSE with AR(1) model: 0.963273633944107
AIC with AR(1) model: 706.114826572702
BIC with AR(1) model: 716.6792093262887
```

We observe a difference between the AIC,BIC of the model and the calculated AIC/BIC and this may be due to a slight difference in the number of terms that are added for the calculation. The values `ar.L1`,`ar.L2`, and `sigma2` are the $\phi_1$, $\phi_2$ and $W$ respectively. Also the p-values are very small.

The code above also plots the original data and the fitted line as well as its residuals. From the first plot we observe a very good agreement with the data, also shown in the p-values of the coefficients.

![](image-7.png)

The residuals look like white noise, but we still have high magnitudes relative to the values of the time series! 

![](image-8.png)

Let's look more closely at the time series.

```py
plt.plot(AR_series[:20], label='original data')
plt.plot(ar_higher_predictions[:20], 'r', label='fitted line')
plt.legend()
plt.show()
```

![](image-9.png)

In this zoom in view of the first 20 values we observe that, in fact, there is always one step off from the original data.

This is a caveat of this program. To circunvent this we will step up the predictions as shown in the code below.

```python
ar_higher_residuals2 = AR_series[:-1]-ar_higher_predictions[1:]
plt.plot(ar_higher_residuals2, 'o')
plt.show()
print("MSE with AR(1) model:", np.mean(ar_higher_residuals2**2))
print("AIC with AR(1) model:", evaluate_AIC(AR_order + 1, ar_higher_residuals2))
print("BIC with AR(1) model:", evaluate_BIC(AR_order + 1, ar_higher_residuals2))
```

As observed in the plot and in the error and IC values we now have a much better fit.

![](image-10.png)


In our next step we will compute a grid of models in order to find the best parameters: MA Order and AR Order.

First we define a general grid search function for the `ARIMA` algorithm as shown in the code below 

```python
def grid_search_ARIMA(data, AR_range, MA_range, verbose=False):
  min_aic = np.inf 
  min_bic = np.inf
  min_aic_index = None
  min_bic_index = None 
  aic_matrix = np.zeros((len(AR_range), len(MA_range)))
  bic_matrix = np.zeros((len(AR_range), len(MA_range)))
  for AR_order in AR_range:
    for MA_order in MA_range:
      arma = ARIMA(data, order=(AR_order, 0, MA_order)).fit()
      aic_matrix[AR_order, MA_order] = arma.aic
      bic_matrix[AR_order, MA_order] = arma.bic
      if arma.aic < min_aic:
        min_aic = arma.aic
        min_aic_index = (AR_order, 0, MA_order)
      if arma.bic < min_bic:
        min_bic = arma.bic
        min_bic_index = (AR_order, 0, MA_order)
  if verbose:
    print("Minimizing AIC order: ", min_aic_index)
    print("Minimizing BIC order: ", min_bic_index )
    print("matrix of AIC", aic_matrix)
    print("Matrix of BIC", bic_matrix)
  return min_aic_index, min_bic_index, aic_matrix, bic_matrix
```

We will first apply it for the AR(2) series.

```python
min_aic_index, min_bic_index, _, _ = grid_search_ARIMA(AR_series, range(4), range(4), verbose=True)
if min_aic_index == min_bic_index:
  arma = ARIMA(AR_series, order=min_bic_index).fit()
  print(arma.summary())
  arma_predictions = arma.predict()
  arma_residuals = AR_series - arma_predictions
  arma_residuals = arma_residuals # Fitting AR 1 model means removing one observation
  plt.plot(AR_series, label='AR time series')
  plt.plot(arma_predictions, 'r', label='fitted line')
  plt.legend()
  plt.show()
  plt.plot(arma_residuals, 'o')
  plt.show()
  print("Automatic selection finds model with AR {0}, MA {2}".format(*min_aic_index))
  print("MSE with selected model:", np.mean(arma_residuals**2))
else:
  print("AIC, BIC do not agree.")

SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  250
Model:                 ARIMA(2, 0, 0)   Log Likelihood                -351.516
Date:                Wed, 31 Jan 2024   AIC                            711.032
Time:                        10:40:27   BIC                            725.118
Sample:                             0   HQIC                           716.701
                                - 250                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.2190      1.508      0.808      0.419      -1.737       4.175
ar.L1          1.5431      0.058     26.696      0.000       1.430       1.656
ar.L2         -0.5853      0.057    -10.249      0.000      -0.697      -0.473
sigma2         0.9599      0.080     11.935      0.000       0.802       1.118
===================================================================================
Ljung-Box (L1) (Q):                   0.03   Jarque-Bera (JB):                 3.77
Prob(Q):                              0.85   Prob(JB):                         0.15
Heteroskedasticity (H):               0.94   Skew:                            -0.24
Prob(H) (two-sided):                  0.79   Kurtosis:                         3.37
===================================================================================
Automatic selection finds model with AR 2, MA 0
MSE with selected model: 0.9632736337780373
```

In the end we obtain the same results as above, but the objective here was to observe if the program could find the AR(2) model as the best and it did. I will not show the plots here as they are the same as the one above.

We will now input our MA(3) data and observe the results.

```python
min_aic_index, min_bic_index, _, _ = grid_search_ARIMA(MA_series, range(4), range(4), verbose=True)
if min_aic_index == min_bic_index:
  arma = ARIMA(MA_series, order=min_bic_index).fit()
  print(arma.summary())
  arma_predictions = arma.predict()
  arma_residuals = MA_series - arma_predictions
  arma_residuals = arma_residuals
  plt.plot(MA_series, label='MA model')
  plt.plot(arma_predictions, 'r', label='fitted line')
  plt.legend()
  plt.show()
  plt.plot(arma_residuals, 'o')
  plt.show()
  print("Automatic selection finds model with AR {0}, MA {2}".format(*min_aic_index))
  print("MSE with selected model:", np.mean(arma_residuals**2))
else:
  print("AIC, BIC do not agree.")

SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  250
Model:                 ARIMA(0, 0, 3)   Log Likelihood                -347.614
Date:                Wed, 31 Jan 2024   AIC                            705.227
Time:                        11:07:41   BIC                            722.834
Sample:                             0   HQIC                           712.314
                                - 250                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.1543      0.252      0.613      0.540      -0.339       0.647
ma.L1          0.9240      0.034     27.025      0.000       0.857       0.991
ma.L2          0.9620      0.129      7.440      0.000       0.709       1.215
ma.L3          0.9548      0.122      7.854      0.000       0.717       1.193
sigma2         0.9081      0.138      6.593      0.000       0.638       1.178
===================================================================================
Ljung-Box (L1) (Q):                   1.46   Jarque-Bera (JB):                 6.56
Prob(Q):                              0.23   Prob(JB):                         0.04
Heteroskedasticity (H):               0.93   Skew:                            -0.31
Prob(H) (two-sided):                  0.75   Kurtosis:                         3.49
===================================================================================```

Automatic selection finds model with AR 0, MA 3
MSE with selected model: 0.9390641964205751
```

Again, the algorithm finds MA(3)) as the best fit to the data. The generated plots show the data, the fitted curve, and the residuals.

![](image-11.png)

![](image-12.png)

Let's now see the result of our ARMA(1,2) series.

```python
min_aic_index, min_bic_index, _, _ = grid_search_ARIMA(ARMA_series, range(4), range(4), verbose=True)
if min_aic_index == min_bic_index:
  arma = ARIMA(ARMA_series, order=min_bic_index).fit()
  print(arma.summary())
  arma_predictions = arma.predict()
  arma_residuals = ARMA_series - arma_predictions
  arma_residuals = arma_residuals
  plt.plot(ARMA_series, label='ARMA series ')
  plt.plot(arma_predictions, 'r', label='fitted line')
  plt.legend()
  plt.show()
  plt.plot(arma_residuals, 'o')
  plt.show()
  print("Automatic selection finds model with AR {0}, MA {2}".format(*min_aic_index))
  print("MSE with selected model:", np.mean(arma_residuals**2))
else:
  print("AIC, BIC do not agree.")

Automatic selection finds model with AR 1, MA 2
MSE with selected model: 0.957173618267881
```

As expected the best model found is a ARMA(1,2). The figures below show the data+model and the residuals respectively.

![](image-13.png)

![](image-14.png)


**Note: don't forget about the time-step correction between model and data to increase the accuracy**

### What if AIC/BIC don't agree?

TBD

## Forming forecasts

Now that we can find models, we can form forecasts. 

With estimates of each $\theta$ and $\phi$ term, we can predict one step ahead. 

$$
\hat{X}_{t + 1} = \hat{\phi}_1 X_{t} + \hat{\phi}_2 X_{t-1} \dots + \hat{\phi}_p X_{t - p + 1} + \hat{\theta}_1 W_{t} + \hat{\theta}_2 W_{t-1} \dots + \hat{\theta}_q W_{t-q + 1} 
$$

To form long-range forecasts, we can just plug in our estimate $\hat{X}_{t + 1}$ in place of $X_{t + 1}$ at every future value, and so on for other future estimates. Let's see what happens when we do this. The forecast will be done with `arma.get_forecast(50)` for 50 time points. We will be using our previouse AR(1,2) series.

```python
arma = ARIMA(ARMA_series, order=min_bic_index).fit()
print(arma.summary())
arma_predictions = arma.predict()
fcast = arma.get_forecast(50).summary_frame()

fig, ax = plt.subplots(figsize=(15, 5))

arma_predictions = arma.predict()
ax.plot(np.arange(len(ARMA_series)), ARMA_series, label='original data')
predicted_values = arma_predictions.reshape(-1,1)
ax.plot(np.arange(len(ARMA_series)), predicted_values, 'r', label='fitted ARMA model')
forecast_means = fcast['mean'].values.reshape(-1,1)
forecast_x_vals = np.arange(len(ARMA_series), len(ARMA_series) + 50)
ax.plot(forecast_x_vals, forecast_means, 'k--', label='mean forecast')
ax.fill_between(forecast_x_vals, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.legend();

 SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  250
Model:                 ARIMA(1, 0, 2)   Log Likelihood                -349.736
Date:                Wed, 31 Jan 2024   AIC                            709.472
Time:                        11:39:08   BIC                            727.080
Sample:                             0   HQIC                           716.559
                                - 250                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.1964      0.224      0.878      0.380      -0.242       0.635
ar.L1          0.4084      0.088      4.652      0.000       0.236       0.581
ma.L1          0.5164      0.075      6.917      0.000       0.370       0.663
ma.L2          0.5762      0.068      8.420      0.000       0.442       0.710
sigma2         0.9545      0.083     11.568      0.000       0.793       1.116
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                 3.23
Prob(Q):                              0.92   Prob(JB):                         0.20
Heteroskedasticity (H):               0.96   Skew:                            -0.23
Prob(H) (two-sided):                  0.85   Kurtosis:                         3.32
===================================================================================
```

In this case we obtain a mean forecast. Yes, it sucks.

![](image-15.png)

As we've seen before, long-term predictions converge rapidly to the mean. As we don't have observations, the CI in the forecast regions rely on the variance of the whole series.

## Return to the price forecasting example.

We'll now finally return to the practical problem of price forecasting. We've applied a quadratic trend to the log stock price and obtained the following Figures. 

The first figure shows the log data versus time and the quadratic fit.

![](pics/stock5.png)

The second figure shows the residuals.

![](pics/stock6.png)

The third and fourth figures show the ACF and the PACF respectively. 

![](pics/stock7.png)

![](pics/stock8.png)

From here we can observe that the ACF/PACF plots suggest an AR(1) model: there are two significant terms in the PACF and a slowly decaying ACF. 

Let's see what our grid search finds.