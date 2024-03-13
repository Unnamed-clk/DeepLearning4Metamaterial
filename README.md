# DeepLearning4Metamaterial
A simple deep-learning model for metamaterial design.

## "BallCluster_Neuro" 
BallCluster_Neuro.py contains a FNN neural network for forward prediction.

This neural network contains 4 full connect hidden layers named as Dense0-4 and a flat layer as input layer, a reshape layer as output layer.

## "BallCluster_Oruen" 
BallCluster_Oruen.py contains a CNN neural network for reverse prediction.

This neural network contains 4 sets of convolution and max-pooling layers, and a full connect layer as output layer.

## "BallCluster_getdata" 
BallCluster_getdata.py mainly contains a method for extracting S-parameter form the ".s2p" or ".s4p" files. Parameters that can be extracted include frequency, magnitude/DB, phase, structural parameters(if contains).

This file also contains normalisation functions to be used in the deep-learning model.

## "Inversion"
Inversion.py mainly contains a function for S-parameter inversion extracting equivalent electromagnetic parameters.
This function is written according to the theory of PHYSICAL REVIEW E 70, 016608 (2004).




