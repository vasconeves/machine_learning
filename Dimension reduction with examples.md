# Dimension reduction with examples

## Introduction

Dimensional reduction is the transformation of high-dimensional data into a low dimension representation. During this process, some information is lost but the main features are (hopefully!) preserved.

These transformations are very important because processing and analyzing high-dimensional data can be intractable. Dimension reduction is thus very useful in dealing with large numbers of observations and variables and is widely used in many fields.

Here we'll approach three different techniques: i) Principal component analysis (PCA), ii) Multidimensional Scaling (MDS), and iii) Stochastic Neighbor Embedding (SNE).

PCA tries to project the original high-dimensional data into lower dimensions by capturing the most prominent variance in the data.

MDS is a technique for reducing data dimensions while attempting to preserve the relative distance between high-dimensional data points.

SNE is a non-linear technique to “cluster" data points by trying to keep similar data points close to each other.

PCA and classical MDS share similar computations: they both use the spectral decomposition of symmetric matrices, but on different input matrices.

## Principal component Analysis (PCA)

PCA is often used to find low dimensional representations of data that **maximizes the spread (variance) of the projected data.**

- The first principal component (PC1) is the direction of the largest variance of the data.
- The second principal component (PC2) is perpendicular to the first principal component and is the direction of the largest variance of the data among all directions that are perpendicular to the first principal component.
- The third principal component (PC3) is perpendicular to both first and second principal components and is in the direction of the largest variance among all directions that are perpendicular to both the first and second principal components. 
  
**This can continue until we obtain as many principal components as the dimension of the original space in which the data is given, i.e. an orthogonal basis of the data space consisting of principal components.**







