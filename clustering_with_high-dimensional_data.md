# Clustering with high-dimensional data

![](pics/intro.png)


Clustering aims to group data points together so that within the same group the observations are very similar while between different groups the observations are very different, according to some dissimilarity rule.

**Note: The data points do not need to come with any labels.** 

Often, clustering is used as a way to decide what and how many class labels are suitable for the data set. 

**For example, we may want to cluster gene expression data into groups that will characterize different cell types in a blood sample.**

Other examples:
  - Find customer groups to adjust advertisement
  - Find subtypes of diseases to fine-tune treatment

In a naïve approach we may try to assign data to clusters using *brute force* methods. However, we quickly find that it is impossible to search through all assignments as it scales with $k^N$, where $k$ is the number of clusters and $N$ the sample size.

Therefore we need smarter approaches to this problem.

## K-means

K means start with a pre-defined number of clusters $K$ and aims at minimizing the value of the **within group sum of squares (WGSS)**:

$$
WGSS = \sum_{k=1}^N\sum_{x^{{i}},x^{(j)} \in C_k} |x^{(i)}-x^{(j)}|^2,
$$

where the $k$ indexes the K different clusters, $C_k$ denotes the $k$-th cluster and $|x^{(i)}-x^{(j)}|$ is the euclidean distance between the data points $i$ and $j$.

The WGSS measures how dissimilar data points in the same cluster are. To find an algorithm to minimize WGSS, it helps to first rewrite WGSS in terms of the means $\mu_k = \sum_{x^{(i)} \in C_k} x^{(i)}/n_k$ of each cluster $C_k$.

$$
WGSS = \sum_{k=1}^N\sum_{x^{{i}},x^{(j)} \in C_k} |(x^{(i)}-\mu_k) - (x^{(j)}-\mu_k)|^2 = \cdots \\
= 2 \sum_{k=1}^K |(x^{(i)}-\mu_k)|^2.
$$

### K-means algorithm

1. Initialize the K means ${\mu_k}_k=1,...,K to random positions.

Then repeat 2. and 3. until convergence.

2. Cluster assignment. Cluster each point with the closest centroid $\mu_k$. Call the set of all points in the cluster $C_k$.
3. Centroids update. Update all centroids $\mu_k$ to be the average position of all points assigned to $C_k$ in the step above.

**Note: K-means does not guarantee convergence to the global minimum. Therefore it is necessary to conduct multiple runs starting with different random initializations. This is more effective when the number of clusters is small.**

### K-medoids

There are two limitations with K-means:
- Its results are sensitive to outliers.
- The cluster centroids are not necessarily data points

In order to overcome these issues, it is possible no modify the algorithm and use medoids instead of means. The medoid is just the data point of a cluster closest to its mean.

### Optimizing K

One heuristic method to determine the number of clusters K is the **elbow method**. 

Plot the loss function WGSS as a function of K and pick the K corresponding to the "elbow" of the plot. For instance, according to the plot below, the best choice for K is 3. Basically what is important here is to balance how much information each cluster provides is with the number of clusters: after some number $k$ the new information gained is very small.

![](pics/elbow.png)

## Gaussian mixture models

Clustering using Gaussian mixture model (GMM) generalizes $K$-mean in two ways:

- Cluster assignment is based on the probabilities of the data point being generated by the different clusters.

- The shape of the clusters can be elliptical rather than only spherical.

Consider the Gaussian mixture model of $K$ Gaussians. We can write the probability of obtaining the observation $\mathbf{X}$ as

$$
P(\mathbf{X}) = \sum_{k=1}^K p_k P(\mathbf{X}|cluster\ k)
$$

where

$$
p_k = P(cluster\ k)
$$

is the missing proportion of each cluster, and

$$
\mathbf{X}|cluster\ k \sim \mathcal{N}(\mu_k,\Sigma_k). 
$$

$P(X|cluster\ k)$ is the probability of obtaining the observation $\mathbf{X}$ given that it is generated by the model for cluster $k$. This mixture has parameters $\theta = \{p1,...,p_k,\mu_1,...,\mu_K,\Sigma_1,..,\Sigma_K\}$. They correspond to the missing proportions, means and covariance matrices of each of the $K$ gaussians respectively.

Given $n$ data points, $x^{(1)},...,x^{(n)} \in \mathbb{R}^d$, our goal is to optimize $\theta$ in order to maximize the data log-likelihood, where

$$
l(x^{(1)},...,x^{(n)};\theta) = \log{\prod_{i=1}^n P(x^{(i)};\theta)} = \sum_{i=1}^n\log{\left[\sum_{k=1}^n p_k P(\mathbf{x}^{(i)}|\text{cluster k};k)\right]}.
$$

**However, there is no closed-form solution to the parameter set $\theta$.**

### The EM algorithm

The **Expectation-Maximization algorithm** (or EM algorithm) is an iterative algorithm that finds a locally optimal solution $\hat{\theta}$ to the GMM likelihood maximization problem.

It involves two steps. 

1. The **E step** involves finding the posterior probability $p(k|i) = P(\text{cluster k}|x^{(i)};\theta)$ that point $x^{(i)}$ was generated by cluster $k$, for every $i=1,...,n$ and $k=1,...,K$. This step assumes the knowledge of the parameter step $\theta$. The posterior is calculated using the Bayes' rule,

$$
p(k|i) = P(\text{cluster k}|x^{(i)};\theta) = \frac{p_k P(x^{(i)}|\text{cluster k};\theta)}{P(x^{(i)};\theta)} = \frac{p_k\mathcal{N}(x^{(i)};\mu^{(k)},\Sigma_k)}{\sum_{j=1}^K\mathcal{N}(x^{(i)};\mu^{(j)},\Sigma_j)}.
$$

2. The **M step** maximizes the **expected log likelihood** function $\tilde{\ell}(x^{(1)},...,x^{(n)};\theta)$, which is a lower bound of the log-likelihood. The algorithm is therefore going to push the data likelihood upwards.

$$
\tilde{\ell}(x^{(1)},...,x^{(n)};\theta) = \sum_{i=1}^n\left[ \sum_{k=1}^K p(k|i)\log{\left( \frac{P(x^{(i)},\text{cluster k};\theta)}{p(k|i)}\right)}\right] = \tilde{\ell}(x^{(1)},...,x^{(n)};\theta) 
$$
$$
= \sum_{i=1}^n\left[ \sum_{k=1}^K p(k|i)\log{\left(\frac{p_k\mathcal{N}(x^{(i)};\mu^{(k)},\Sigma_k)}{p(k|i)}\right)}\right]
$$

This expected log-likelihood function is a lower bound on the actual log-likelihood, 

$$
{\ell}(x^{(1)},...,x^{(n)};\theta) = \sum_{i=1}^n\log{\left[ \sum_{k=1}^K  P(x^{(i)},\text{cluster k};\theta)\right]}.
$$

due to Jensen's inequality.

In the special case where the covariance matrix is $\Sigma_k = \sigma_k^2 \mathbf{I}$, the parameters that maximiz the expected log-likelihood function are as follows:

$$
\hat{\mu}^k = \frac{\sum_{i=1}^n x^{(i)}p(k|i)}{\sum_{i=1}^n p(k|i)}
$$
$$
\hat{p}_k = \frac{1}{n}\sum_{i=1}^n p(k|i),
$$
$$
\hat{\sigma}_k^2 = \frac{\sum_{i=1}^n p(k|i)|x^{(i)}-\hat{\mu}^k|^2}{\sum_{i=1}^n p(k|i)}.
$$

The E and M steps are repeated iteratively until there is no noticeable change in the actual likelihood computed after M step using the newly estimated parameters or if the parameters do not vary by much.

#### Initialization

For the initialization it is possible to 

- do a random initialization of the parameter set $\theta$.
- employ a k-means to find the initial cluster centers of the $K$ clusters and use the global variance of the dataset as the initial variance of all the $K$ clusters.  In the latter case, the mixture weights can be initialized to the proportion of data points in the clusters as found by the k-means algorithm.

### Optimizing K

In the context of GMM, the optimization of the number of clusters can be found by, for instance, maximizing the **Bayesian Information Criterion** (BIC), where

$$
BIC = \text{log-likelihood} - \frac{\log{n}}{2} \text{\# of parameters}.
$$

## Hierarchical clustering

Hierarchical clustering does not start with a fixed chosen number of clusters, but builds a hierarchy of clusters with different levels corresponding to different numbers of clusters.

Advantages of hierarchical clustering:

- solves clustering for all possible numbers of clusters $1,2,...,n$ all at once.
- can choose desired number of clusters *a posteriori*.

We can use a bottom-up or top-down approach:

- Agglomerative clustering (Bottom-Up)

or 

- Divisive clustering (Top-down)

**Here we will detail the bottom-up approach.**

### Agglomerative Clustering

Agglomerative clustering starts with 1 data point per cluster and, at each stage, merges pairs of clusters that are the closest together, according to a dissimilarity measure.

![](pics/dissimilarity.png)

The merging can be graphically be depicted by a tree, also known as **dendogram**. The bottom most level has $n$ clusters (of 1 observation each), and the merging occurs as the levels go up. The number of clusters decreases, and the top-most level only has 1 cluster comprising all data. 

#### Dissimilarity between clusters

In order to choose which pair of clusters to merge at each stage, we need to define a dissimilarity measure between clusters, and the dissimilarity measure between clusters is often based on dissimilarity between points. A few commonly used distances between individual points are:

- $l^2$ norm, i.e. the usual Euclidean distance

$$
d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) = \sqrt{\left(x_1^{(i)}-x_1^{(j)}\right)^2 + \left(x_2^{(i)}-x_2^{(j)}\right)^2 + ... + \left(x_p^{(i)}-x_1^{(p)}\right)^2}, \mathbf{x}^{(i)} \in \mathbb{R}^p
$$

- $l^1$ norm (also known as Manhattan distance)

$$
d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) = |x_1^{(i)} - x_1^{(j)}| + |x_2^{(i)} - x_2^{(j)}| + ... + |x_p^{(i)} - x_p^{(j)}|, \mathbf{x}^{(i)} \in \mathbb{R}^p
$$

- $\lambda^\infty$ norm (i.e. the maximum distance among all coordinates)

$$
d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) = \max_{k=1,...,p} |x_k^{(i)} - x_k^{(j)}|, \mathbf{x}^{(i)} \in \mathbb{R}^p.
$$

Now we can define dissimilarity measures between clusters:

- Minimum distance between points in two clusters, known as **single linkage**

$$
d(C_1,C_2) = \min_{x^{(i)} \in C_1,x^{(j)} \in C_2} d(x^{(i)},x^{(j)}).
$$

- Maximum distance between points in two clusters, known as **complete linkage**

$$
d(C_1,C_2) = \max_{x^{(i)} \in C_1,x^{(j)} \in C_2} d(x^{(i)},x^{(j)}).
$$

- Average distance between points in two clusters, known as **average linkage**

$$
d(C_1,C_2) = \frac{1}{n_1n_2}\sum_{x^{(i)}\in C_1}\sum_{x^{(j)}\in C_2}d(x^{(i)},x^{(j)}).
$$

*How do the resulting clusters look like? Which is which?*

![](pics/linkage_examples.png)
*The colors show the different clusters.*

### Optimizing K

- There are no strict rules.
- Rule of thumb: find the larges vertical "drop" in the tree.

![](pics/dendrogram_example.png)
*In this case we assume the largest drop is where K=3. It is always useful to check the end result on the left.*

## DBSCAN

**Density-based spatial clustering of applications with noise** (DBSCAN) aims to cluster together points that are close to each another in a dense region, and leave out points that are in low density regions.

Let two points be **connected** if they are within a distance $\epsilon$ of one another. a **core point** is a point that is connected to at least $k$ other points, where $k$ is a measure of **core strength**.

- Two points are placed in the same cluster if and only if there is a connecting path between them **consisting of only core points**, except possibly at the ends of the path. 

Let's observe the following figure.

![](pics/dbscan.png)

In this diagram the blue points are core points with core strenth $k=4$, since they are connected to at least four other points. Two clusters are formed. In each cluster, each non-core (black) point is connected to a core point (blue). The points not connected to any core points are **outliers** and are part of no cluster in DBSCAN.

## Silhouette plot

A silhouette plot helps us measuring the quality of cluster assignments. 

The **silhouette score** of a data point $\mathbf{x}^{(i)}$ is defined as

$$
S(x^{(i)}) = \frac{b(x{(i)}-a(x^{(i)}))}{\max{(b(x^{(i)}),a(x^{(i)}))}}
$$

where 

$$
a(x^{(i)}) = \frac{1}{n_i-1}\sum_{x^{(j)}\in C_i,j\ne i} d(x^{(i)},x^{(j)})
$$

is the average "within group" distance or dissimilarity from $x^{(i)} and

$$
b(x^{(i)}) = \min_{C_k\ne C_i}\frac{1}{n_k}\sum_{x^{(j)}\in C_k} d(x^{(i)}x^{(j)})
$$

is the average distance or dissimilarity from $x^{(i)} to the closest other cluster.
