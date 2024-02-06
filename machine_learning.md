# Machine learning

## Linear classifiers and generalizations

### Introduction

Machine Learning (ML) as a discipline aims to design, understand, and apply computer programs that learn from experience (data) for the purpose of modelling, prediction and control. 

Here we will start with **prediction**.

*What can we predict?* Ex: market forecast, weather, next word in a sentence, pedestrian behavior etc.

We can also try to predict unknown properties. Ex: properties of materials, an object in an imagem, what English sentence translates in another language, whether a product review carries + or - sentiment etc.

However, all these problems have a common difficulty: it is very hard to write down a solution in terms of rules or code directly. It is far easier to lay out examples of correct behavior. In other words, it is much easier to provide large numbers of examples and design an algorithm that can learn from these examples.

In supervised learning, we hypothesize a collection of functions (or mappings) with **parameters** from the examples (e.g. images) to the targets (e.g. objects in images). The ML algorithm then automates the process of finding the parameter of the function that fits best with the example-target pair.

Let's use an example of a movie recommender system. Let's suppose we have five features (yes/no or 1/0) of interest in a movie. We can encode them, for each movie, in a feature vector as

$$
X_i = [x_1,x_2,x_3,x_4,x_5],
$$

where $x_i \in \{0,1\}$.

Our goal is to predict whether someone will like a movie or not. For a subsample of our data we know this value. It is also a {1,0} vector, each value corresponding to one movie (e.g. like/dislike).

Therefore, we can define this sentiment as $y_i$ where $y_i \in {0,1}. We call also call it the label.

We can then use the subsample where we have the label information and define a training set as

$$
S_n = {(X_i,Y_i)},
$$

where $X^i$ are the feature vectors and $y^i$ are the targets (or labels), as well as a test set where we don't have the labels and we will evaluate the algorithm.

Training data can be graphically depicted on a (hyper)plane. **Classifiers** are mappings that take feature vectors as input and produce labels as output. The **hypothesis space** is the set of possible classifiers. In this case we have +1 or -1 labels and we can display them as a projection in a plane such as the figure below.

![](ml1.jpg)

A common kind of classifier is the **linear classifier** which linearly divides space into two as shown, for instance, in the figure below.

![](ml2.png)

The line dividing the two regions is called the **decision boundary**.

We can define this type of classifier as

$$
h: x \rightarrow \{-1,1\}.
$$

There are many ways to perform this division. But before looking into that let's first define the **training error** which is the error that the classifier commits when performing its task. Thus,

$$
\epsilon_n(h) = \frac{1}{n}\sum_{i=1}^n[h_i(x) \ne y_i].
$$

An illustration of such error is shown in the figure below.

![](image-1.png)

**Note: the classifier region needs to contain all regions of the test region as shown in the figure below. We need to generalize the best as possible but as the complexity arises, the more difficult it is to generalize.**

![](image-2.png)

Let's now move to the separation issue. We'll start with the **linear separation**. 

*How to parametrize the classifiers?* We can do this i) through origin or ii) through anywhere.

i) Through origin: 

$$
h(x;\mathbf{\theta}) = sign(\mathbf{\theta . x}),
$$

*...because the dot product will be negative if the vector x with angle $\alpha$ related to $\theta$ will have $\alpha$ negative. The same happens if positive values.*

The illustration below shows this parametrization in the 2-D plane.

![](image-3.png)


ii) Through anywhere: 


$$
h(x;\mathbf{\theta}) = sign(\mathbf{\theta . x} + \theta_0),
$$

where $\theta \in R^d$, $\theta_0 \in R$. The illustration below shows this parametrization in the 2-D plane. The only difference from i) to ii) is only the offset $\theta_0$ of the plane.

![](image-4.png)

In other words, given $theta$ and $theta_0$, a linear classifier $h:X\rightarrow \{-1,0,1\}$ is a function that outputs 1 if $\mathbf{\theta.x} + \theta_0 \gt 0$, 0 if it is zero and -1 if $\mathbf{\theta.x} + \theta_0 \lt 0$.

Definition: The training examples $S_n$ are **linearly separable** if there exists a parameter vector $\hat{\theta}$ and offset parameter $\hat{\theta_0}$ such that $y_i(\hat{\theta}.x_i+\hat{\theta_0}) \gt 0\text{ } \forall \text{ } i=1,...,n$.



