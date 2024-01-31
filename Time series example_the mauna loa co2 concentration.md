# The Mauna Loa $CO_2$ concentration

## Introduction

In 1958, Charles David Keeling (1928-2005) from the Scripps Institution of Oceanography began recording carbon dioxide ($CO_2$) concentrations in the atmosphere at an observatory located at about 3,400 m altitude on the Mauna Loa Volcano on Hawaii Island. The location was chosen because it is not influenced by changing $CO_2$ levels due to the local vegetation and because prevailing wind patterns on this tropical island tend to  bring well-mixed air to the site. 

While the recordings are made near a volcano (which tends to produce $CO_2$), wind patterns tend to blow the volcanic $CO_2$ away from the recording site. Air samples are taken several times a day, and concentrations have been observed using the same measuring method for over 60 years. In addition, samples are stored in flasks and periodically reanalyzed for calibration purposes. The observational study is now run by Ralph Keeling, Charles's son. The result is a data set with very few interruptions and very few inhomogeneities. It has been called the “most important data set in modern climate research."

Let $C_i$ be the average $CO_2$ concentration in month $i$ ($i=,1,2...$). We want to look for a form of the type

$$
C_i = F(t_i) + P_i + R_i,
$$

where: 

- $F(t_i)$ accounts for the long-term trend 
- $t_i$ is the time at the middle of the $ith$ month, measured in fractions of years after Jan 15 1958. Specifically we take 

$$
t_i = \frac{i+0.5}{12}, i = 0,1,...,
$$

where $i$ = 0 corresponds to January 1958. We add 0.5 because the first measurement is halfway through the month.
- $P_i$ is periodic in $i$ with a fixed period, accounting for the seasonal pattern.
- $R_i$ is the remaining residual that accounts for all other influences.

## Data

The input data for this example can be found in the file `CO2.csv` under the folder `data_and_materials`. It provides the concentration of CO2 recorded at Mauna Loa for each month starting March 1958. Here, we will consider only the $CO_2$ concentration given in column 5, which is unadjusted. 

First we load the libraries and the data.

```python
import numpy as np
import math
import scipy as sp
import scipy.stats as st
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.sandbox.stats.multicomp as multi
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error

df = pd.read_csv('data_and_materials/CO2.csv',sep=',',skiprows=57,header=None)
df.columns=('yr','mn','datem','date','co2','season','fit','seasonf','co2_2','season_2')
```

## Pre-processing data


