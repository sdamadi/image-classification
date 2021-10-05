 
# Optimization method
 
In this document we will explain different methods that can be used to minimize the loss of image classification. All of these methods are first order optimization method since they only use the information of the gradient.
 
## Optimization algorithms
 
### Stochastic gradient descent 
### [Adam](https://arxiv.org/pdf/1412.6980.pdf)
 
 
## What modules are used for optimization?
 
### [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py)
 
To determine an optimizer algorithm we create an instance of Stochastic first order oracle, i.e., `SFO`. This class creates the proper optimizer based on the arguments that are passed to it.
 
 
[`utils/first_order_oracle.py`](https://github.com/sdamadi/image-classification/blob/main/utils/first_order_oracle.py)
 
[First order oracle module](https://github.com/sdamadi/image-classification/blob/main/utils/first_order_oracle.py) will create an optimizer.
 
