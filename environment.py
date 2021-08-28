import numpy as np
import matplotlib.pyplot as plt


class Environement():
	#Creates Conway's game of life
	def __init__(self,window_size=16):
		self.window_size = window_size
		self.state = np.random.randint(low=0,high=2,size=(self.window_size,self.window_size))

	def calcNextState(self):
		newState = np.zeros((self.window_size,self.window_size))	#inizializes new state
		for r,c in np.ndindex(self.state.shape): 					#interates over cells
			num_alive = np.sum(self.state[r-1:r+2, c-1:c+2]) - self.state[r, c] #counts living neighbours

			if self.state[r, c] == 1 and num_alive < 2 or num_alive > 3: #kills cell if number of neighbours is not 2 or 3.
				pass
			elif (self.state[r, c] == 1 and 2 <= num_alive <= 3) or (self.state[r, c] == 0 and num_alive == 3): #kills cell if number of neighbours is not 2 or 3.
				newState[r, c] = 1
				
		self.state = newState
					
	def plotState(self):
		#showes current state as plt
		plt.imshow(self.state,cmap='gray',interpolation='none')
		plt.show(block=False)
		plt.pause(0.001)

	def getState(self):
		return self.state	



if __name__ == '__main__':
	E = Environement(128)

	while True:
		E.plotState()
		E.calcNextState()
