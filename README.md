# Game-of-Life-with-Neural-Networks
Conway's game of life is a cellular autometon, which produces complex behaviours from a few simple rules.
This Reposetory compares the performence of different neural network architectures on the task of predicting the next state in Conway's game of life.

## Architectures

### 1. CNN
The CNN architecture consists out of 2 Layers with 8 and 1 feature maps. A Relu activation function in between and a Sigmoid activation function after the second layer were used.

### 2. Residual CNN
The residual CNN architecture consists out of 2 Layers with 8 and 1 feature maps. A Relu activation function in between and a Tanh activation function after the second layer were used.
The output of the network is added to the previous state to get the next one.

### 3. MLP
The MPL architecture consists out of 2 Fully connected with 256 neurons. A Relu activation function in between and a Sigmoid activation function after the second layer were used.

### 4. MLP residual
The MPL architecture consists out of 2 Fully connected with 256 neurons. A Relu activation function in between and a Tanh activation function after the second layer were used.
The output of the network is added to the previous state to get the next one.



## Results


![image](https://user-images.githubusercontent.com/72468505/128404168-fe40cad0-476a-435d-90b0-9abd93cea8a7.png)
