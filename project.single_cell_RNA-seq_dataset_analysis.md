# Single-Cell RNA-seq analysis

In this project we will analyze a single-cell RNA-seq dataset from the mouse neocortex, a region of the brain that governs its higher functions, such as perception and cognition.

This dataset is a subset of a larger single-cell RNA-seq dataset compiled by the [Allen Institute](https://alleninstitute.org/).

The single-cell RNA-seq data comes in the form of a counts matrix, where

- each row corresponds to a cell

- each column corresponds to the normalized transcript compatibility count (TCC) of an equivalence class of short RNA sequences, rescaled to units of counts per million. You can think of the TCC entry at location $(i,j)$ of the data matrix as the level of expression of the $j-th$ gene in the $i-th$ cell.

## Visualization and clustering on a small subsample

We will start the analysis using the **p1** folder which contains a small, labeled subset of the data. In it we have the count matrix along with “ground truth" clustering labels , which were obtained by scientists using domain knowledge and statistical testing. This is for use in Problem 1.

We begin by loading the necessary libraries and loading the data.

```python
%matplotlib inline
%load_ext memory_profiler
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib as mpl
from yellowbrick.style import rcmod
from scipy.stats import spearmanr
from memory_profiler import profile

X = np.load('data/p1/X.npy') #Matrix nxp where n = 511 cells, p = 45768 genes
y = np.load('data/p1/y.npy') #ground truth labels
```

If we take a look at matrix X we observe that we have 511 rows and 45768 columns, corresponding to the different cells and genes respectively.

As the gene values can vary several orders of magnitude it is advisable to perform a log transformation on the data. In this case we use $log_2(x+1)$. The sum of one unit is done to avoid floating point infinity situations close to $log(0)$.

```python
X_log = np.log2(X + 1) #log transformation of the Data
```

We can calculate the PCA of the raw data and the log transformed data to show the reason for this transformation.

```python
pca = PCA().fit(X)
pca_log = PCA().fit(X_log)
```

First, we can calculate the percentage of variance explained by the first PCA for the raw and the log transformed data

```python
# Percentage of variance explained by first component
print("First component raw:", pca.explained_variance_ratio_[0])
print("First component log:", pca_log.explained_variance_ratio_[0])
First component raw: 0.4277967098357264
First component log: 0.13887564870826197
```

We can already observe that the explanatory power is much more concentrated in the first PC in the raw data than in the log transformed data.

Let's now plot the cumulative sum of the explained variance ratio. 

```python
# Cumulative variance explained plots
plt.plot(range(1, X.shape[0]+1), np.cumsum(pca.explained_variance_ratio_), color="red", label="raw")
plt.plot(range(1, X.shape[0]+1), np.cumsum(pca_log.explained_variance_ratio_), color="blue", label="log")
plt.legend()
plt.show()
```

![](pics/pca_cumsum.png)

The plot shows that while the information in the PCs of the raw data is concentrated in the first few PCAs, in the log transformed data it is more evenly distributed. 

**This is important because we're looking for genes that can differentiate between cells, even if they are not present inside many cells and or have low levels of expression. Therefore, the log transformed data allows to better discern low levels of expression as well as high levels of expression.**

The 85\% level variance cutoffs can be calculated with the following code

```python
# 85% variance cutoffs
print("Raw:", np.where(np.cumsum(pca.explained_variance_ratio_) >= .85)[0][0] + 1)
print("Log:", np.where(np.cumsum(pca_log.explained_variance_ratio_) >= .85)[0][0] + 1)
Raw: 36
Log: 339
```

**This means that while in the raw data 85\% of the explained variance is concentrated in the first 36 PCAs, in the log transformed data the same explained variance is much more distributed allowing a much more sensitive analysis across different expression magnitudes.**

### Data visualization

Let's now try and visualize the data. For instance, if we plot the data from the first two columns from the log transformed data what do we obtain? 

```python
# Nothing informative!
plt.scatter(X_log[:,0],X_log[:,1])
```

![](pics/rna_raw.png)

Not much! Let's try again, this time using PCA. 

```python
# Three visually distinct clusters. Could potentially argue for 4 or 5 as well, but these are not well-separated
z = pca_log.transform(X_log)
plt.scatter(z[:,0],z[:,1])
plt.xlabel('PC1'),plt.ylabel('PC2')
```

Now we can clearly distinguish at least 3 clusters and maybe two more. 

![](pics/rna_pca.png)

Let's now visualize the MDS and T-SNE plots.

```python
# We still see 3 distinct clusters.
mds=MDS(n_components=2).fit_transform(X_log)
plt.scatter(mds[:,0],mds[:,1])
```

![](pics/rna_mds.png)

The MDS plot shows 3 distinct clusters.

```python
# Emergence of at least 5 clusters. 
z_tsne = TSNE(n_components=2,perplexity=40).fit_transform(z[:,0:50])
plt.scatter(z_tsne[:,0],z_tsne[:,1])
```

![](pics/rna_tsne.png)

In the T-SNE plot it becames apparent that there are indeed 5 separate clusters. Of course we can still play around with the perplexity parameter.

### Cluster assignment


The next step will be to allocate the data points into the 5 different clusters. We will do that with the `KMeans` algorithm.

```python
# 5 clusters: PCA plot
kmeans = KMeans(5, tol=1e-6)
kmeans.fit(z[:,0:50])
plt.scatter(z[:,0],z[:,1], c=kmeans.labels_)
```
The code above runs the algorithm and generates a PCA plot color coded with each cluster assignment.

![](pics/rna_pca_cluster.png)

The cluster positions are now much clearer despite the two clusters at the bottom left corner being a bit mixed-up.

```python
# 5 clusters: MDA with K means plot
plt.scatter(mds[:,0],mds[:,1],c=kmeans.labels_)
```
![](pics/rna_mds_cluster.png)

The color-coded MDS plot is not so clear but is still a possible outcome. We must remember we are reducing many dimensions onto a plane.

```python
# T-SNE plot
plt.scatter(z_tsne[:,0],z_tsne[:,1], c=kmeans.labels_)
```

The T-SNE plot has the clearest distinction among the 5 clusters.

![](pics/rna_tsne_cluster.png)

### Cluster diagnostics

Let's now do a sanity check using the elbow method. *How many clusters should we find?*

```python
# Would select 3, 4, or 5 clusters
all_kmeans = [i for i in range(8)]
for i in range(8):
    cur_kmeans = KMeans(i+2)
    cur_kmeans.fit(z[:,0:50])
    print("Num clusters", i+2, "Inertia:", cur_kmeans.inertia_)
    all_kmeans[i] = cur_kmeans
plt.plot([i+2 for i in range(8)], [all_kmeans[i].inertia_ for i in range(8)])
```

![](pics/rna_elbow.png)

Observing the elbow plot, the number of clusters should be between 3 and 5 in my perspective.

Another test we can use is dendrogram analysis. In order to do this we must switch to R. 

