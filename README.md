# CUDA-MLP-from-scratch

This is my second MLP implementation, this time using CUDA.
The goal is to speed up training using GPU parallelism.

Current Status: Not fully working yet (on big datasets), debugging kernel issues and issues with big Datasets like MNIST.
and the project files that ive uploaded are completly raw, i'll add comments soon too. 

## What Works:
- Forward pass  
- Basic backprop
- Weights updates
  
ive tried to bebug GPU backend a lot but still i couldnt pin point the issue

## TODO:
- Fix why it doesn't work on big datasets
- Fix faulty loss tracking
- Implement better CPU support 

If you're reading this and know CUDA, feel free to help!.
