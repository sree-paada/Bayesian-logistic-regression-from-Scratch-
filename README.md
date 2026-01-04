# Bayesian-logistic-regression-from-Scratch-
Chronic Kidney Disease Prediction Using Bayesian Logistic Regression

Bayesian Logistic Regression from Scratch 

 

This is a  Python and NumPy version of the Bayesian Logistic Regression. In contrast to conventional Logistic Regression, which only gives point estimates of the weights (MLE or MAP) the Bayesian approach can give a probability distribution of the weights, which enables us to use this in making predictions to quantify the uncertainty 

 

Overview 

The project demonstrates the core concepts of Bayesian inference applied to binary classification. By implementing the model from scratch, we move away from "black-box" libraries to understand the underlying mathematics of: 

Likelihood Functions (Bernoulli/Sigmoid) 

Prior Distributions (Gaussian) 

Posterior Estimation (via Metropolis-Hastings MCMC or Laplace Approximation) 

Predictive Distributions (Integrating over the posterior) 

 

Mathematical Background 

1. The Model 

We model the probability of a binary outcome $y \in \{0, 1\}$ given features $x$ and weights $w$: 

$$P(y=1|x, w) = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$ 

2. The Bayesian Approach 

Instead of finding a single "best" $w$, we calculate the posterior distribution: 

$$P(w|X, y) = \frac{P(y|X, w) P(w)}{P(y|X)}$$ 

Where: 

Prior $P(w)$: Usually a Gaussian distribution $\mathcal{N}(0, \alpha^{-1}I)$. 

Likelihood $P(y|X, w)$: Product of Bernoulli trials. 

Posterior: Often intractable, requiring sampling methods like MCMC or approximations. 

 
