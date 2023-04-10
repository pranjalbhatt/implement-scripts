# Neural Networks and more

This is a collection of several practice scripts for implementation of object detection, neural network compression, distillation, pruning and large models.   
Details as follows:


## Rank Approximations and compression (JAX based)

Low Rank Approximations are used for few cases:

1.   To compress a Neural Network 
2.   To denoise a signal
3.   To impute missing values/find a structure to fill empty data

Singular Value Decomposition (SVD) exists for every rectangular matrix. Here in '', gentle implementation of low rank SVD  is scripted to compress a deep neural network. 
