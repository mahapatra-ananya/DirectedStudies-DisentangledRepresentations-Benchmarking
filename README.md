# CPSC 448 Directed Studies Project 

Supervised by Dr. Jiarui Ding at University of British Columbia, Vancouver

Disentangled representation learning in single cell omics aims to separate condition-specific and
condition-invariant factors of variation in single cell RNA sequencing (scRNA-seq) data, facilitating
interpretable and biologically meaningful latent representations. DISCoVeR, a recently proposed variational
autoencoder framework designed for this purpose, was investigated in this project. The original
scRNA-seq DISCoVeR experiment was reproduced to to evaluate DISCoVeR, and the study was extended
by training the model on two additional datasets to assess generalizability. The results indicate
that while DISCoVeR performs as expected on the original dataset, it exhibits poor generalizability
across other scRNA-seq experiments. In contrast, Conditional Subspace VAE, the framework on which
DISCoVeR is built, demonstrates robust performance and better generalizability across datasets, suggesting
its suitability for broader applications in learning single cell disentangled representations.

The aim of this project was threefold:
(a) To build conceptual knowledge and understanding by extensively reviewing the literature on concepts such as probabilistic
graphical models, variational autoencoders (VAEs), and disentangled representation learning in single cell omics, in order to
investigate the performance of a new VAE framework named DISCoVeR for the purpose of disentangling condition-specific and
condition-invariant latent variables in scRNA-seq data.
(b) To reproduce the scRNA-seq experiment conducted by Slavutsky et al. (2025) to evaluate DISCoVeR’s performance, and
additionally contribute mutual information scores to aid evaluation.
(c) To extend the experiments by training the DISCoVeR model on two additional datasets and evaluating its generalizability
by comparing its performance to other variational autoencoder frameworks.
