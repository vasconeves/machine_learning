# Example. Molecular classification of cancer

Source of data: Golub et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring, Science, Vol. 286:531-537.

The data set golub consists of the expression levels of 3051 genes for 38 tumor mRNA samples. Each tumor mRNA sample comes from one patient (i.e. 38 patients total), and 27 of these tumor samples correspond to acute lymphoblastic leukemia (ALL) and the remaining 11 to acute myeloid leukemia (AML).

Here we will estimate the number of genes that can be used to differentiate the tumor types (meaning that their expression level differs between the two tumor types) using

- the uncorrected p-values,
- the Holm-Bonferroni correction, and
- the Benjamini-Hochberg correction

## Introduction

Let $\overline{X}_{ALL,i}$ be the mean of the expression levels for gene $i$ across ALL mRNA samples, and $\overline{X}_{AML,i}$ be the same but for the AML mRNA samples.

For both, we have that $N_{ALL}$ and $N_{AML}$ are the number of mRNA samples for the ALL and the AML tumors respectively.

Then, we can write the variance for the average of ALL, $\overline{X}_{ALL,i}$ as

$$
s^2_{\overline{X}_{ALL,i}} = \frac{s^2_{ALL,i}}{N_{ALL}},
$$

where $s^2_{ALL,i}$ is the sample variance for gene $i$.

Similarly we can write the variance for $\overline{X}_{AML,i} as

$$
s^2_{\overline{X}_{AML,i}} = \frac{s^2_{AML,i}}{N_{AML}}.
$$

We can now use the difference between the means $\Delta\overline{X}_i = \overline{X}_{ALL,i} - \overline{X}_{AML,i}$ to gauge the difference between expression levels for gene $i$. The variance of this metric is

$$
s^2_{\Delta\overline{X}_i} = s^2_{\overline{X}_{ALL,i}} + s^2_{\overline{X}_{AML,i}}.
$$

This allows us to use the well-known Welch unequal variances t-test statistic, which can be written as

$$
t_{Welch,i} = \frac{\overline{X}_{ALL,i} - \overline{X}_{AML,i}}{\sqrt{\frac{s^2_{ALL,i}}{N_{ALL}}+\frac{s^2_{AML,i}}{N_{AML}}}}.
$$

The distribution of this test statistic can be approximated by a t-distribution but with a modified number of degrees of freedom, which is approximately

$$
\nu_i \approx \frac{\left(\frac{s^2_{ALL,i}}{N_{ALL}}+\frac{s^2_{AML,i}}{N_{AML}}\right)^2}{\frac{1}{\nu_{ALL}}\left(\frac{s^2_{ALL,i}}{N_{ALL}}\right)^2 + \frac{1}{\nu_{AML}}\left(\frac{s^2_{AML,i}}{N_{AML}}\right)^2},
$$

where $\nu_{ALL} = N_{ALL}-1$ and $\nu_{AML} = N_{AML}-1$.

## Number of significant genes using uncorrected *p-values*

Using the following code we obtained the number of significantly associated genes (for $\alpha \le 0.05$) using uncorrected p-values.

Code:
```python
#Source of data: Golub et al. (1999). Molecular classification of cancer: 
#class discovery and class prediction by gene expression monitoring, 
#Science, Vol. 286:531-537
#First 27 tumor samples --> ALL
#Remaining 11 samples --> AML

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

golub_data = pd.read_csv('data_and_materials/golub_data/golub.csv',index_col=0)
golub_classnames = pd.read_csv('data_and_materials/golub_data/golub_cl.csv')

iall = golub_classnames['x'] == 0 #mask for ALL
iaml = golub_classnames['x'] == 1 #mask for AML
ALL_genes = golub_data.loc[:,iall.values] #only ALL samples
AML_genes = golub_data.loc[:,iaml.values] #only AML samples
ALL_N = ALL_genes.shape[1] # # of ALL samples
AML_N = AML_genes.shape[1] # # of AML samples
N_genes = golub_data.shape[1] # # of both

#Uncorrected p-values

# Calculate the Welch's t-test statistic
test_statistic = (ALL_genes.mean(axis=1) - AML_genes.mean(axis=1))/np.sqrt(ALL_genes.var(axis=1, ddof=1)/ALL_N + AML_genes.var(axis=1, ddof=1)/AML_N)
# Find the number of degrees of freedom of the statistic according to the approximation formula
test_dof = (ALL_genes.var(axis=1, ddof=1)/ALL_N + AML_genes.var(axis=1, ddof=1)/AML_N)**2 / ( (ALL_genes.var(axis=1, ddof=1)/ALL_N)**2/(ALL_N-1) + (AML_genes.var(axis=1, ddof=1)/AML_N)**2/(AML_N-1))
# Find the 2-sided p-values using survival function = 1-CDF
p_values = sp.stats.t.sf(np.abs(test_statistic), test_dof)*2
# Count how many of these p-values are below the significance threshold
(p_values <= 0.05).sum()
1078
```
The number of significant genes, according to the uncorrected p-values, is 1078.

*But what does this really mean? Do we really have 1078 significant genes?*

Out of the 3051 genes we have approximately 152 genes which are false positives! *How can we correct this?*

Our approach will depend on the level of control we need to have over the error rate or type I error. We'll show here two possible approaches: i) the Family-wise error rate via the Holms-Bonferroni correction and, ii) the false discovery rate via the Benjamini-Hochberg correction.

## Family-wise error rate (FWER)

FWER is usually used when we really need to be careful and control the error rate dut to possible serious consequences in any false discovery, such as the Pharmaceutical sector.

We can control the size of the FWER by choosing significance levels of the individual tests to vary with the size of the series of tests. This translates to correcting the *p-values* before comparing with a fixed significance level e.g. $\alpha = 0.05$.

### Holm-Bonferroni Correction

Suppose we have $m$ hypothesis. The application of the method consists in the following steps:

- Calculate the initial *p-values* for each hypothesis.
- Sort the initial *p-values* in increasing order.
- Start with the *p-value* with the lowest number. If

$$
p_{i} < \frac{\alpha}{m-(i-1)},
$$

then
  - reject $H_0^i$
- proceed to the next smallest *p-value* by 1, and again use the same rejection criterion above.
- As soon as hypothesis $H_0^k$ is not rejected, stop and do not reject any more of the $H_0$.

We calculate the Holm-Bonferroni correction through the following code.

```python
# Sort the p-values in ascending order
p_values_sorted = np.sort(p_values)
# These are the adjusted significance thresholds as an array.
# Each element is the threshold for the corresponding p-value in p_values_sorted
# Note that (np.arange(N_genes)+1)[::-1] gives an array with [N_genes, N_genes-1, N_genes-2, ..., 1]
# The [::-1] reverses the array.
holm_bonferroni_thresholds = 0.05/(np.arange(N_genes)+1)[::-1]
# First we compare the p-values to the associated thresholds. We then get an array
# where the p-values that exceed the threhold have a value of False.
holm_bonferroni_significant = p_values_sorted < holm_bonferroni_thresholds
# We want to find the first value of False in this array (first p-value that exceeds the threshold)
# so we invert it using logical_not.
holm_bonferroni_not_significant = np.logical_not(holm_bonferroni_significant)
# argwhere will return an array of indices for values of True in the supplied array.
# Taking the first element of this array gives the first value of True in holm_bonferroni_not_significant
# which is the same as the first value of False in holm_bonferroni_significant
holm_bonferroni_first_not_significant = np.argwhere(holm_bonferroni_not_significant)[0]
# We reject all hypothesis before the first p-value that exceeds the significance threshold.
# The number of these rejections is exactly equal to the index of the first value that
# exceeds the threshold.
num_holm_bonferroni_rejections = holm_bonferroni_first_not_significant
print(num_holm_bonferroni_rejections)
[103]
```
**Using the stringent Holm-Bonferroni correction we obtain only 103 significant genes!**

## False discovery rate (FDR)

In most cases, however, FWER is too strict and we loose too much statistical power. The most sensible course of action is then to control the expected proportion of false discoveries among all discoveries made. We can define 

$$
FDR = \mathbb{E}\left[ \frac{ \text{nº type 1 errors or false discoveries}}{\text{total number of discoveries}}\right].
$$

### The Benjamini-Hochberg correction

The Benjamini-Hochbert correction guarantees $FDR < \alpha$ for a series of **$m$ independent tests.**.

The method is as follows:

- Sort the $m$ *p-values* in increasing order.
- Find the maximum $k$ such that

$$
p_{k} \le \frac{k}{m}\alpha
$$

- Reject all of $H_0^1, H_0^2,...,H_0^k.$

The following code shows the application of this criterion.

```python
# These are the adjusted significance thresholds as an array.
benjamini_hochberg_thresholds = 0.05*(np.arange(N_genes)+1)/N_genes
# First we compare the p-values to the associated thresholds.
benjamini_hochberg_significant = p_values_sorted < benjamini_hochberg_thresholds
# We are intested in the last p-value which is significant.
# Remeber that argwhere returns an array of indicies for the True values, so
# we take the last element in order to get the index of the last p-value which
# is significant.
benjamini_hochberg_last_significant = np.argwhere(p_values_sorted < benjamini_hochberg_thresholds)[-1]
# We reject all hypotheses before the last significant p-value, AND we reject
# the hypothesis for the last significant p-value as well. So the number of rejected
# hypotheses is equal to the index of the last significant p-value PLUS one.
num_benjamini_hochberg_rejections = benjamini_hochberg_last_significant + 1
print(num_benjamini_hochberg_rejections)
[695]
```
**Using the less strict Benjamini-Hochberg correction we obtain a much more sensible number: 695 significant genes.**