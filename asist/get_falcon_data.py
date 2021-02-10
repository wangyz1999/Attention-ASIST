import pickle
import numpy as np
from numpy import linalg as LA
from numpy.linalg import matrix_rank
import math

import matplotlib.pyplot as plt

def jl_transform(dataset_in,objective_dim,type_transform="basic"):
	"""
	This function takes the dataset_in and returns the reduced dataset. The 
	output dimension is objective_dim.
	dataset_in -- original dataset, list of Numpy ndarray
	objective_dim -- objective dimension of the reduction
	type_transform -- type of the transformation matrix used. 
	If "basic" (default), each component of the transformation matrix 
	is taken at random in N(0,1).
	If "discrete", each component of the transformation matrix 
	is taken at random in {-1,1}.
	If "circulant", he first row of the transformation matrix 
	is taken at random in N(0,1), and each row is obtainedfrom the 
	previous one by a one-left shift.
	If "toeplitz", the first row and column of the transformation 
	matrix is taken at random in N(0,1), and each diagonal has a 
	constant value taken from these first vector.
	"""
	if type_transform.lower() == "basic":
		jlt=(1/math.sqrt(objective_dim))*np.random.normal(0,1,size=(objective_dim,
		len(dataset_in[0])))
	elif type_transform.lower() == "discrete":
		jlt=(1/math.sqrt(objective_dim))*np.random.choice([-1,1],
		size=(objective_dim,len(dataset_in[0])))
	elif type_transform.lower() == "circulant":
		from scipy.linalg import circulant
		first_row=np.random.normal(0,1,size=(1,len(dataset_in[0])))
		jlt=((1/math.sqrt(objective_dim))*circulant(first_row))[:objective_dim]
	elif type_transform.lower() == "toeplitz":
		from scipy.linalg import toeplitz
		first_row=np.random.normal(0,1,size=(1,len(dataset_in[0])))
		first_column=np.random.normal(0,1,size=(1,objective_dim))
		jlt=((1/math.sqrt(objective_dim))*toeplitz(first_column,first_row))
	else:
		print('Wrong transformation type')
		return None

	trans_dataset=[]
	[trans_dataset.append(np.dot(jlt,np.transpose(dataset_in[i]))) 
		for i in range(len(dataset_in))]
	return trans_dataset

if __name__ == '__main__':

    with open('falcon_hard_distance_matrix_noTriage.pkl', 'rb') as f:
        distance_matrix = pickle.load(f)

    # original distance matrix
    D = distance_matrix

    # method to translate distance matrix to coordinate system
    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    M = np.zeros((len(D), len(D)))
    for i in range(len(D)):
        for j in range(len(D)):
            M[i][j] = (D[0][j] ** 2 + D[i][0] ** 2 - D[i][j] ** 2) / 2
    # print(matrix_rank(M))
    w, v = LA.eig(M)
    X = v * np.sqrt(w)

    # remove zero and NaN columns
    X = X[:,~np.all(np.isnan(X), axis=0)]
    X = X[:,~np.all(X == 0, axis = 0)]

    # reduce dimension of X to 2
    lower = jl_transform(X, 2)
    print(lower)