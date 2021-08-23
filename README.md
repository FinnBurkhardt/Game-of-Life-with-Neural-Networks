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
The MPL architecture consists out of 2 fully connected with 256 neurons. A Relu activation function in between and a Sigmoid activation function after the second layer were used.

### 4. MLP residual
The MPL architecture consists out of 2 fully connected with 256 neurons. A Relu activation function in between and a Tanh activation function after the second layer were used.
The output of the network is added to the previous state to get the next one.

## Approache
The size of one frame are 16x16 pixels where the value 1 represents a living cell and 0 represents a dead cell.
For the training set 5000 series of 30 frames were used. For the validation set 500 series of 30 frames were used.
The first state of each series was initialized with a 50% chance for each cell to be alive/dead.



## Results

Comperison of Models during training.
The loss for each model is the average over 10 full training runs.
![image](https://user-images.githubusercontent.com/72468505/128404168-fe40cad0-476a-435d-90b0-9abd93cea8a7.png)

![Alt Text]![MLP](https://user-images.githubusercontent.com/72468505/130492091-3f117123-b17f-417e-b936-438cdeabc3b7.gif))
