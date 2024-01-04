# Mammography study - a discrete variable example

Here I will demonstrate how we can analyze data from observational studies using a mammography study as an example.

The key questions here are the following: 

* *Does mammography speeds up cancer detection?*

* *How do we set up an experiment in order to minimize the problem of confounding?*

The study consists in the screening of women's breasts by X-rays as shown in the Table below.

![Alt text](pics/mammography_table.png)

Intuitively, we are inclined to compare between those who took the treatment with the ones how refused it. However this is an **observational comparison**!

Instead we need to compare the whole treatment group against the whole control group.

We need to do an **intent-to-treat analysis**.

From the table we will assume that the **outcome variable** will be death by breast cancer. This variable will depend on the treatment variable, that will be the people offered mammography.

Why? Because we cannot **force** people to do the mammography!

## RCT

In this experimental design we need to consider the following properties:

* Patient selection: Some populations are more likely to develop breast cancer than others depending on their prior health background, so the interpretation of any result we obtain will depend on how we have defined our selection procedure. In general, how we select the treatment and control groups will influence the population for which the conclusion is valid.

* Control group: We need to compare the outcome variable for those who have received the treatment with a baseline (i.e, the control group). Here, the control group (who were not offered a mammography) must be a comparable set of people to the treatment group (who have been offered a mammography).

* Features of the patients: One way to make accurate comparison and interpretation of results is to ensure that the treatment group is representative across factors such as health status, age, and ethnicity, and is similar along these dimensions as the control group. This way, we can attribute any differences in outcome to the treatment rather than differences in covariates. In a general study, it is upon the researchers' discretion to determine which features to be careful about.

An experimental design where treatments are assigned at random and that satisfies the three points described above is called a **randomized controlled trial (RCT)**. The controls can be an observational group or treated with a placebo.

## Double blind

In any experiment that involves human subjects, factors related to human behavior may influence the outcome, obscuring treatment effects. For example, if patients in a drug trial are made aware that they actually received the new treatment pill, their behavior may change in a number of ways, such as by being more or less careful with their health-related choices. Such changes are very difficult to model, so we seek to minimize their effect as much as possible.

The standard way to resolve this is through a double-blind study, also called a blinded experiment. Here, human subjects are prevented from knowing whether they are in the treatment or control groups. At the same time, whoever is in charge of the experiment and anyone else who could interact with the patient are also prevented from directly knowing whether a patient is in the treatment or the control group. This is to prevent a variety of cognitive biases such as observer bias or confirmation bias that could influence the experiment.

In some cases, it is impossible to ensure that a study is completely double-blind. In the mammography study, for example, patients will definitely know whether they received a mammography. If we modify the treatment instead to whether a patient is offered mammography (which they could decline), then we neither have nor want double-blindness.

## Hypothesis testing

From the table we can observe that

* death rate from breast cancer in control group = $\frac{63}{31k}$ = 0.0020
* death rate from breast cancer in treatment group = $\frac{39}{31k}$ = 0.0013

***Key question:*** *Is the difference in death rates between treatment and control sufficient to establish that mammography reduces the risk of death from breast cancer?*

We need to perform an **hypothesis test**.

Hypothesis testing steps:

1. Determine a model. In our case Binomial (modeling as the outcome of a number *n* of heads/tails) or a Poisson model (modeling as number of arrivals/events).

2. Determine a mutually exclusive null hypothesis and alternative.

* $H_0$: $\pi = 0.002$ or $\lambda = 63$

* $H_1$: $\pi = 0.0013$ or $\lambda = 39$

3. Determine a test statistic (quantity that can differentiate between $H_0$ and $H_1$ and whose assumptions under $H_0$ are true)

T = #Deaths under $H_0$:
    
* T ~ Binomial(31k,0.002) 
    
or 
    
* T ~Poisson(63)

4. Determine a significance level ($\alpha$) i.e. the probability of rejecting $H_0$ when $H_0$ is true: e.g. $\alpha \le 0.05$.

The following Figure shows the pmf of a Binomial and a Poissonian distributions. The significance level is depicted as the dashed blue line. We can observe that the distributions are very similar. Therefore, for large n, it is preferred to use the Poissonian approximation.

In this case, if the p-value of the test statistic is lower than 0.05 we can say that we reject the null at the 0.05 confidence level. In practice, however, we should be very careful about the importance of p-values and should use other statistical tools as well as we'll see later on.

![](pics/pmf.png)

*Code:*

```python
import numpy as np
from scipy.stats import poisson,binom
import matplotlib.pyplot as plt

n = 31000 #control sample size
death = 63 # number of cancer deaths in control
rate = death/n #death ratio in control
x = np.arange(0,125) #parameter space
#binomial calculation exercise
pmf_binomial = binom.pmf(x,n,rate)

#poisson approximation
pmf_poisson = poisson.pmf(x,death)

alpha = x[np.where(np.cumsum(pmf_binomial) <= 0.05)][-1] #alpha = 0.05

plt.plot(x,pmf_poisson,'.',label='Binomial pmf')
plt.plot(x,pmf_poisson,'or',alpha=0.5,label='Poisson pmf')
plt.plot((63,63),(0,0.05),'r--',alpha=0.1,label='$E[H_0]$')
plt.plot(39,pmf_binomial[39],'b*',markersize=12,label='$E[H_1]$')
plt.plot((alpha,alpha),(0,pmf_binomial[alpha]),'b-',label="$\\alpha=0.05\ threshold$")
plt.xlabel('Cancer deaths'),plt.ylabel('pmf')
plt.xlim(20,100),plt.ylim(0,0.06)

plt.legend()
```

The p-value is calculated by summing all pmf values up to pmf(x=39). Therefore, the p-value is a probability. In this case we obtain a value of 0.0008 which is much smaller than 0.05. **Thus, we reject the null**.

We can now define type I error, type II error and power from the following table.

![](pics/table_power.png)

Type I error is a false positive and is bounded by $\alpha$ (meaning type I error  $\le \alpha$), Type II error is a false negative, and the power can be written as

$Power = 1-Type\ II\ error$

**Note:** there is a trade-off between type I error and type II error.

**Note:** the power of a 1-sided test is usually higher than the power of a 2-sided test. Thus you should always use 1-sided tests when evaluating deviations that go in one direction only.

The following plot shows a graphical depiction of the power of the test as well as the upper bound of the Type I error represented by $\alpha$. **The plot clearly shows the interplay between type I and type II errors: 

![](pics/power.png)

## Fisher exact test

What if we don't know the frequency of deaths of the control? We can do a **Fisher exact test** which is based on the hypergeometric distribution.

Null hypothesis: $\pi$ control = $\pi$ treatment

Alternative hypothesis: $\pi$ control > $\pi$ treatment

***Key question:*** *Knowing that 102 subjects died and that number of treatments / controls each is 31k what is the probability that deaths are so unevenly distributed?*

The $\textit{p-value}$ is then the sum of probabilities of obtaining a value of T that is more extreme than 39, in the direction of the alternate hypothesis:

$P_{H_0}(T \le 39) = \sum_{t=0}^{39}{{31k \choose t}{31k \choose 102-t}{62k \choose 102}}$

$\textit{p-value} = 0.011 < 0.05$

From here, based on the significance level $\alpha$, we can either

* reject the null if $p \le \alpha$

or

* fail to reject the null if $p > \alpha$.

Advantages:

- Does not assume knowledge about the true probability of dying due to breast cancer in the control population

Shortcomings:

- Assumes knowledge of the margins (i.e., row and column sums)
- **Alternative is Bernard’s test** (estimates the margins)
- Both tests are difficult to perform on large tables for computational reasons

# Sleeping drug study - a continuous variable example

Let's now move on to a different study where we have a **continuous variable** as our variable of interst.

In the clinical trial setup we are testing a new sleeping aid drug which in principle should help users suffering from insomnia to increase their sleeping time.

*Which should be the best approach to this problem?*

We could adapt the previous framework for the mammography study. However, the power of this hypothesis might not be very high, considering that the sample size is small: people have a wide range of sleep lengths and may be difficult to discern anything due to this natural variability. An alternative to the standard RTC test is a **paired test design**.

## Paired test design

In the paired test design one takes multiple samples from the same individual at different times, a group corresponding to the control situation and the other group corresponding to the treatment situation. This allow us to estimate the effect of the drug at the individual level. 

Therefore we will measure the difference between the observed values in the treatment and in the control situations,

$Y_i = X_{i,treatment}-X_{i_control}.

The null hypothesis is the one where the expected value of the measurement $E[Y_i] = 0$.

**Note: the paired test design removes the need for randomization. However the individual does not know if he is in the treatment or the in the control group. This can be done by having two seperate observation periods: one for a placebo administration and the other for the drug treatment.**

## Modeling choce for the sleeping drug study

The following table shows the data collected for our clinical trial.

![](pics/sleeping_drug.png)

*What should be our modeling choice?*

In this case we can think in two possibilities *a priori*:

* The number of hour slept in a day has an upper bound, as shown in the table which implies that the difference $Y$ is also bounded. This points in favour of the **uniform distribution** as this model has bounded support, while the Gaussian distribution has infinite support.

* We can look at the empirical distribution of the observations. The number of hours slept by an adult is known to be centered around 8 hours, and outliers are rare, so this favours the Gaussian distribution model.

In this case, it is prefereble to choose the Gaussian distribution as it closely matches the empirical distribution of the sleeping population. We can further argue that the number of hours slept is a cumulative effect of a large number of biological and lifestile variables. As most of these variables are unrelated to each other, the cumulative effect can be approximated by a normal distribution.

## The central limit theorem (CLT) and the z-test statistic

From the CLT, we know that when $n$ is large, the distribution of the random variables of a sample $X_1...X_n$ is approximately normal with mean $\mu$ and variance $\sigma^2$. Therefore,

$\overline{X} \sim N(\mu,\frac{\sigma^2}{n})$

From here we can defina a test statistic, the **z-test statistic** as

$z = \frac{\overline{X}-\mu}{\sigma/\sqrt{n}} \sim N(0,1)$.

This test statistic does not depend on the parameters $\mu$ or $\sigma$. It is, in fact, the standard normal. Thus we can simply use the CDF of this function to calculate the $\textit{p-value}$ of the test statistic.

In our case we want to answer the following question:

*Does the drug increase hours of sleep enough to matter?*

To answer this question we go through the following steps:

1. Choose a model (Gaussian) and use the proper random variables ($Y_i = X_drug-X_placebo).
2. State the hypothesis (in this case is a one-sided test):
    - H_0: $\mu = 0$
    - H_1: $\mu > 0$.
3. Calculate the test statistic.

We have, however, a problem. To calculate z we need to known the true value of the variance $\sigma$. Since we do not know the population variance, only the sample variance we cannot use this test.

## T-test

The solution to this problem is to use a t-test instead. The t-test uses the sample variance, and enable us to write

$T = \frac{\overline{X}-\mu}{\hat{\sigma}/\sqrt{n}} \sim N(0,1)$.

## Application to the sleeping drug experiment

To calculate the T-test to our experiment we first calculate the difference of the hours of sleep between drug and placebo as shown in the table below.

![](pics/diferences.png)

Then, we know that under the null the t-test follows a t distribution of 9 degrees of freedom,

$\frac{X}{\hat{\sigma}/sqrt{n}} \sim t_9$.

Using for instance the t-test program in the scipy python library we obtain

```python
t_test = sp.stats.ttest_1samp(x,0,alternative='greater') #note: alternative='greater' because by default the t-test output is a two-sided test.
print ('The t-test statistic =',t_test[0], 'with a p-value =',t_test[1],'and df = 9.')
The t-test statistic = 3.1835383022188735 with a p-value = 0.005560692749284678 and df = 9.
```

In this case we can also reject the null at the 5% level (or even at the 1% level).

The pdf of a $t_9$ distribution is shown below and compared to a standard normal distribution.

![](pics/z_and_t.png)

Code:
```python
df = 9 #number of degrees of freedom
x = np.linspace(-5,5,1000) #support
td = sp.stats.t.pdf(x,df)
tn = sp.stats.norm.pdf(x)

indx = np.max(np.where(x<t_test[0])) #location of the pdf value of the t-score

plt.plot(x,td,'r',label='T-distribution (df=9)')
plt.plot(x,tn,'b',label='Z distribution')
plt.plot(t_test[0],td[indx],'*k',label='T score',markersize=12)

plt.legend()
```
A great feature of the $t_n$ distribution is that it introduces uncertainty due to the estimation of the population variance: As $n$ gets smaller and smaller the tails of the T distribution get larger and larger as shon in the following Figure.

![](pics/t_distributions.png)

**Note: as a rule of thumb, when the sample size $n \ge 30$ the t-distribution is already very similar to the normal distribution. At this threshold the normal approximation becomes (more) valid.**


















