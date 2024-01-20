# Resonate-and-fire neurons
This project consists on an adaptation of the Resonate-and-fire (RAF) neuron model created by [Eugene M. Izhikevich](https://www.izhikevich.org/publications/resfire.htm), designed to be used with [snnTorch](https://snntorch.readthedocs.io/en/latest/) and [PyTorch](https://pytorch.org/) libraries. 
The aim is to test the capabilities of a RAF neural network on a simple task like classification of the MNIST dataset using backpropagation. 

## Content
In the 'examples' folder, you will find simple implementations on the MNIST dataset. Additionally, the code for the RAF neuron model (raf.py) is located inside the 'neuron' folder.

## Results MNIST
A simple 3-layer fully-connected RAF neural network of dimensions 784-1000-10, 'frequency=30' and 'beta=0.99' with a rate encoded input along 25 time steps, obtained a maximum test accuracy of 94.66% after 25 epochs.

![Shallow RAF neural network](/assets/images/MNIST_shallow.jpg)


The same network but with an additional hidden layer (784-1000-1000-10) gives us a maximum test accuracy of 97.57% after 25 epochs.

![Deeper RAF neural network](/assets/images/MNIST_deep.jpg)


On the other hand, a convolutional RAF neural network like the one in the Examples folder obtained a maximum test accuracy of 98.4% after 25 epochs.

![Convolutional RAF neural network](/assets/images/MNIST_conv.jpg)

