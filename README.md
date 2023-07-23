# Game-of-Life-with-Neural-Networks
Conway's game of life is a cellular automaton, which produces complex behaviors from a few simple rules.
This Repository compares the performance of different neural network architectures on the task of predicting the next state in Conway's game of life.

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

## Approach
The size of one frame are 16x16 pixels where the value 1 represents a living cell and 0 represents a dead cell.
For the training set 5000 series of 30 frames were used. For the validation set 500 series of 30 frames were used.
The first state of each series was initialized with a 50% chance for each cell to be alive/dead.
To evaluate the predictive power of the neural networks the mean squared error(MSE) between the next state and the predicted next state is calculated.



## Results

Comparison of Models during training.
The loss for each model is the average over 10 full training runs.
![image](https://user-images.githubusercontent.com/72468505/128404168-fe40cad0-476a-435d-90b0-9abd93cea8a7.png)

## CNN
![CNN](https://user-images.githubusercontent.com/72468505/130492201-b5c1079b-792c-43b7-925b-37bb8ea0dd49.gif)

## CNN_residual
![CNN_residual](https://user-images.githubusercontent.com/72468505/130492264-0aca39e9-ef76-42d8-ba02-139bb8667570.gif)

## MLP
![MLP](https://user-images.githubusercontent.com/72468505/130492091-3f117123-b17f-417e-b936-438cdeabc3b7.gif)

## MLP_residual
![MLP_residual](https://user-images.githubusercontent.com/72468505/131228345-31090da1-84e4-4398-9f86-64e4894e0c38.gif)

# Usage


1. Install dependencies

2. Create Data
```python
python3 createData.py
```
3. Select wanted model in train.py
4. 
5. Run train.py
```python
python3 train.py
```

# Dependencies
1. python 3.6
2. pytorch 1.8.0
3. numpy
4. matplotlib
