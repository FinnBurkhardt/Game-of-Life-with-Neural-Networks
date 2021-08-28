from environment import  Environement
from tqdm import tqdm
import os
import numpy as np


def main():
	E = Environement()


	DATADIR = '/home/finn/Desktop/CGOL/data/val'

	for j in tqdm(range(500)):
		E = Environement(16)
		FOLDER = '/'+str(j).zfill(3)+'/'#create name of folder
		#create folder if not already there
		if not os.path.exists(DATADIR+FOLDER):
			os.makedirs(DATADIR+FOLDER)

		#calc 30 states and safe them as .npy
		for i in tqdm(range(30)):
			E.calcNextState()
			S=E.getState().astype(np.uint8)*255
			np.save(DATADIR+FOLDER+str(i).zfill(3)+'.npy', S)
			

if __name__ == "__main__":
	main()