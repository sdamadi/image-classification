 
# Initialization 
 
In this document we will explain how models are initialized.
 
## Initialization methods
 
Currently three following initializations can be used:
 
- Kaiming
- Xavier 
 
## What modules are used for creating the network?
 
### [`main.py`](https://github.com/sdamadi/image-classification/blob/main/main.py)
 
In the [main module](https://github.com/sdamadi/image-classification/blob/main/main.py) we call `network_init` function to initialize the network. The argument `init_policy` determines which initialization should be applied. 
 
### [`utils/init.py`](https://github.com/sdamadi/image-classification/blob/main/utils/init.py)
 
This module includes two functions where `weights_init` initializes each layer and `network_init` uses `weights_init` to initialize all layers. 