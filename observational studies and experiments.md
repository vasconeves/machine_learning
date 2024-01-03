# Mammography study

Here I will demonstrate how we can analyze data from observational studies using a mammography study as an example.

The key questions here are the following: 

* *Does mammography speeds up cancer detection?*

* *How do we set up an experiment in order to minimize the problem of confounding?*

This study consists in the screening of women's breasts by X-rays as shown in the Table below.

![Alt text](pics/mammography_table.png)

We are inclined to compare between those who took the treatment with the ones how refused it. However this is an **observational** comparison!

Instead we need to compare the whole treatment group against the whole control group.

We need to do an **intent-to-treat** analysis.

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

The standard way to resolve this is through a double-blind study , also called a blinded experiment . Here, human subjects are prevented from knowing whether they are in the treatment or control groups. At the same time, whoever is in charge of the experiment and anyone else who could interact with the patient are also prevented from directly knowing whether a patient is in the treatment or the control group. This is to prevent a variety of cognitive biases such as observer bias or confirmation bias that could influence the experiment.

In some cases, it is impossible to ensure that a study is completely double-blind. In the mammography study, for example, patients will definitely know whether they received a mammography. If we modify the treatment instead to whether a patient is offered mammography (which they could decline), then we neither have nor want double-blindness.

## Hypothesis testing

From the table we can observe that

* death rate from breast cancer in control group = $\frac{63}{31k}$ = 0.0020
* death rate from breast cancer in treatment group = $\frac{39}{31k}$ = 0.0013

Key question: *Is the difference in death rates between treatment and control sufficient to establish that mammography reduces the risk of death from breast cancer?*

We need to perform an **hypothesis test**.

Hypothesis testing steps:

1. Determine a model. In our case Bernoulli (modeling as 1/0 problem) or a Poisson model (modeling as number of events).

2. Determine a mutually exclusive null hypothesis and alternative.

* $H_0$: $\pi = 0.002$ or $\lambda = 63$

* $H_1$: $\pi = 0.0013$ or $\lambda = 39$

3. Determine a test statistic (quantity that can differentiate between $H_0$ and $H_1$ and whose assumptions under $H_0$ are true)

T = #Deaths under $H_0$:
    
T ~ Binomial(31k,0.002) 
    
or 
    
T ~Poisson(63)

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
plt.plot((63,63),(0,0.05),'r--',alpha=0.1,label='$E[X]$')
plt.plot((alpha,alpha),(0,pmf_binomial[alpha]),'b-',label="$\\alpha=0.05$")
plt.xlabel('Cancer deaths'),plt.ylabel('pmf')
plt.xlim(20,100),plt.ylim(0,0.06)

plt.legend()
```

The p-value is calculated by summing all pmf values up to pmf(x=39). Therefore, the p-value is a probability. In this case we obtain a value of 0.0008 which is much smaller than 0.05. **Thus, we reject the null**.

We can now define type I error, type II error and power from the following table.

![](pics/table_power.png)

Type I error is a false positive and is bounded by $\alpha$ (meaning type I error  $\le \alpha$), Type II error is a false negative, and the power can be written as

$Power = 1-Type II error$

**Note:** there is a trade-off between type I error and type II error.

**Note:** the power of a 1-sided test is usually higher than the power of a 2-sided test. Thus you should always use 1-sided tests when evaluating deviations that go in one direction only.

## Fisher exact test

What if we don't know the frequency of deaths of the control? We can do a **Fisher exact test** which is based on the hypergeometric distribution.

Null hypothesis: $\pi$ control = $\pi$ treatment

Alternative hypothesis: $\pi$ control > $\pi$ treatment

*Key question: Knowing that 102 subjects died and that number of treatments / controls each is 31k what is probability that deaths are so unevenly distributed?*

We can write the test as

$P_{H_0}(T=39) = (31k/39)(31k/63)/(62k/102)$

$p-value$ = 0.011 < 0.05

Advantages:

- Does not assume knowledge about the true probability of dying due to breast cancer in the control population

Shortcomings:

- Assumes knowledge of the margins (i.e., row and column sums)
- **Alternative is Bernard’s test** (estimates the margins)
- Both tests are difficult to perform on large tables for computational reasons












