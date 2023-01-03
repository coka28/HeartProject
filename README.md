# HeartProject
A repo for a pattern recognition exercise

The task for which this repository was created is creating a binary classifier to predict heart disease in patients given 10 features. The dataset is stored as heart.csv.

Description of the approach:

  I)    The dataset
  
      The dataset consists of 918 patients, of whom 508 are labeled as having a heart disease, the rest are labeled as healthy (binary classes).
      The feature vectors contain 5 cardinal features (age, resting blood pressure/rbp, cholesterol, max heart rate and oldpeak). Rbp contains one NaN, cholesterol contains 172 NaNs.
      There are 3 binary features (sex, fasting blood sugar/fbs, ean), 2 ternary features (sts score, ecg) and one quarternary feature (chest pain type).
     
  II)   Preprocessing the data
  
      Missing values/NaNs have been imputed using the mean value of that features's column.
      Cardinal features have been standardized (z-transformed) and then transformed using a logistic function. k-nary features with k>2 have been represented by distributed binary encoding (00 = 0, 01 = 1, 10 = 2, 11 = 3 for instance for a quaternary feature). The result is a new Matrix of feature row vectors with 14 features, not counting the class label.
      
  III)  The general plan
  
      From the outset, the plan was to create a continuous representation of the partially discrete-valued feature vectors, in order to be able to use continuous distributions and metrics to analyze the data. To achieve this, an autoencoder had to be designed. [1] give a description, including MatLab code in the supplementary material, of how to train such an autoencoder using restricted boltzmann machines [2, 3, 4]. Usually deep autoencoders are hard to train using stochastic gradient descent, the standard method for artificial neural networks, due to vanishing gradients in layers further away from the output layer where the error signal is created. Restricted boltzmann machines (RBMs) are bipartite graphical models that represent an energy-based probability function over the space of visible (input) and hidden nodes' states. Given some input signal which is clamped to the input neurons, they assume a steady state after a (possible very large) nr or Gibbs-sampling steps, in which visible neurons send a weighted activation signal to hidden neurons, that then are activated by the sum of input signals (using a logistic function 1 / (1 + exp(-x))). The "amount" of activation resembles a probability p in [0, 1] and the hidden neurons get a new state, sampled from a bernoilli distribution with parameter p. The hidden neurons then send their state as a weighted signal back to the visible neurons, which may in turn be activated and possible (but not necessarily) sampled. Going back and forth like this brings the RBM closer to it's equilibrium state of "lowest energy", which dependds on symmetric weights between visible and hidden layer, as well as a per-neuron bias term. To train the RBM, the goal is to reduce the "difference" between the distribution of training data, and the inherent distribution of the RBM. This can be formulated as minimizing the KL divergence between the two distributions, but the actual equilibrium concentration of the RBM is not easily computationally accessible, since it would take an infeasably large amount of sampling steps to reach it for each parameter update. The gradient of the max likelihood w.r.t. the parameters of the model can however be approximated by sampling for just a few iterations (usually just 1, this is called CD-1 for contrastive divergence, the algorithm proposed by Hinton, which is a breakthrough when trying to train RBMs). RBMs that are trained in this way can be stacked on top of each other to form the layers that later become the hidden and code layers of an autoencoder, that is subsequently fine-tuned using standard stochastic gradient descent on a loss function. Training a net with millions of parameters in this way takes less than an hour and has a high probability of success.
      
  IV)    Using autoencoders to generate compact, continuous representations of the data point enables for instance the use of clustering algorithms. One such algorithm is Bayesian hierarchical clustering [5], which has the following nice properties:
  
        1. It is a probabilistic model and does rely less on heuristics than other clustering algorithms
        2. It does not assume a fixed number of clusters (much like other hierarchical clustering algorithms)
        3. It is a bayesian model, meaning it is less sensitive to choice of hyperparameters than many other methods
        4. Given an appropriate model of the distributions, during clustering, parameters for individual cluster-generating pdfs can be inferred (in this case using max log-likelihood), yielding for example a gaussian mixture model, which can then be used to perform soft clustering, modeling the uncertainty of assigning a point to a cluster
        
The implementation which can be found here relies on an inverse-wishart-prior for mean and covariance of MV gaussian estimation, it can be found at the archived git-repository [6].


Discussion of the results:

The approach relies mainly on unsupervised methods (RBMs and autoencoders) and using these tools to encode the data vectors from the data set, one can plot the data and recognize that without providing labels for the data points, a significant difference between healthy and diseased patients can be observed in many cases. Although this is very interesting, it seems that relying too heavily on unsupervised methods has reduced the potential quality of the classifier, since at no point did an algorithm try to optimize the quality of class predictions.
        
Literature:

[1] Hinton, G. E. (2006). Reducing the Dimensionality of Data with Neural Networks. Science, 313(5786), 504–507. doi:10.1126/science.1127647 

[2] An Introduction to Restricted Boltzmann Machines. Asja Fischer and Christian Igel. https://link.springer.com/content/pdf/10.1007/978-3-642-33275-3_2.pdf

[3] A Practical Guide to Training Restricted Boltzmann Machines, Version 1. Geoffrey Hinton., 2010

[4] Learning Deep Generative Models. Ruslan Salakhutdinov, 2015. doi:10.1146/annurev-statistics-010814-020120

[5] Bayesian Hierarchical Clustering. Katherine A. Heller, Zoubin Ghahramani. https://www2.stat.duke.edu/~kheller/bhcnew.pdf

[6] https://github.com/caponetto/bhc
