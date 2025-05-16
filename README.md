# CUDA-MLP-from-scratch

This is my second MLP implementation, this time using CUDA.
The goal is to speed up training using GPU parallelism.

Current Status: Not fully working yet, debugging kernel issues and issues with big Datasets like MNIST.

🔧 What Works:
- Forward pass 
- Basic backprop
- Weights updates

TODO:
- Fix why it doesn't work on big datasets
- Fix faulty loss tracking
- Implement better CPU support 

If you're reading this and know CUDA, feel free to help!
