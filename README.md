# Machine learning

This repository showcases what I've learned in statistics and ML in the past few years.

TOC. THIS IS WORK IN PROGRESS! The first version will be finished by the end of January 2024! Please be patient! :) Multi-daily updates are expected up to the end of Feb. ;)

## Provisional TOC

- **Examples from medical science.**
  - [Observational and Experimental Studies in Clinical Trials: An Introduction to Statistics](observational_and_experimental_studies_in_CT.md).
    - Mammography study - a discrete variable example.
      - RCT, Double blind, hypothesis testing, fisher exact test.
    - Sleeping drug study - a continuous variable example.
      - Paired test design, Modeling choices, the CLT and z-test statistic, the t-test.
    - Other basic statistical concepts.
      - testing the normality assumption, the Wilcoxon signed-rank test, confidence intervals, the likelihood ratio test.
    - Multiple hypothesis testing.
      - Family-wise error rate, Bonferroni correction, Holm-Bonferroni correction, False discovery rate, Benjamini-Hochberg correction, commonly accepted practice.
  - [The Salk Vaccine Field Trial](example.the_salk_vaccine_field_trial.md). Here is shown how should the interpretation of simple, qualitative results be done in the context of clinical trials.
  - [Molecular classification of cancer](example.molecular_classification_of_cancer.md). In this example it is estimated  the number of genes that can be used to differentiate tumor types using different approaches, highlighting the difference between using uncorrected p-values and employing p-value corrections, such as the Holm-Bonferroni and the Benjamini-Hochberd methods.
  - [Project: Single-cell RNAseq dataset analysis](project.single_cell_RNA-seq_dataset_analysis.md). **Work in progress**
- **Examples from Science**
  - [Gamma ray observation analysis - a likelihood ratio test application](example.gamma_ray_observation_analysis.md)
- **Methodology**
  - [Gradient descent optimization example](optimization_example_gradient_descent.md)
    - Notation and convexity
    - Multidimensional convexity and local optimization
    - Quadratic minimization and gradient descent
    - Newton's method of minimization
    - Gradient descent algorithm
    - Step sizes and quadratic bounds
    - Stochastic gradient descent
    - Gradient descent example with synthetic data
  - [Correlation and Least Squares Regression: a general view](correlation_and_least_squares_regression_a_general_view.md)
    - Example from Astronomy (Hubble "constant")
    - Correcting simple non-linear relationships
      - Solar system example as a cautionary tale (Kepler's law)
    - Multiple linear regression (the **good stuff**)
      - An example of multiple linear regression - exoplanet mass data
  - [Methods of classification on high-dimensional data](methods_of_classification_on_high-dimensional_data.md)
    - Bayes rule for classification
    - Quadratic discriminant analysis (QDA)
    - Linear discriminant analysis (LDA)
    - Reduced-rank LDA (a.k.a. Fisher's LDA)
    - Logistic regression
    - Support Vector Machines (SVM)
    - Wrap-up
    - Quality of classification
- **Dimension Reduction, Data Visualization and Clustering**
  - [Dimension reduction with examples](dimension_reduction_with_examples.md)
    - Principal Component Analysis (PCA)
    - Multidimensional Scaling (MDS)
    - T-Distributed Stochastic Neighbor Embedding (T-SNE)
    - Dimension reduction example: digit recognition
      - Embedding techniques comparison
      - Discussion
  - [Clustering with high-dimensional data: theory](clustering_with_high-dimensional_data.md)
    - K-means
      - K-medoids
      - Optimizing K: the elbow method
    - Clustering using Gaussian mixture models 
      - The EM algorithm
      - Optimizing K via the Bayesian information criterion
    - Hierarchical clustering
      - Agglomerative clustering
      - Optimizing K
    - DBSCAN
    - Silhouette plot
  - [Clustering with high-dimensional data: examples](example.clustering_with_high-dimensional_data.md)
    - Gaussian Mixture Model generation
    - Data visualization: PCA, Elbow plot, MDS, T-SNE
    - Data visualization comparison
    - Clustering method examples
      - K-means applied to PCA, MDS and T-SNE plots
      - Diagnostics for clustering methods
        - Sum of squares criterion
        - Silhouette plots
        - Hierarchical clustering
    - A real world example: RA Fisher's Wine Dataset
      - Logistic regression in the context of the Wine Dataset
      - Feature selection using logistic regression

