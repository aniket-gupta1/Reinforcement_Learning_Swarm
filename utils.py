import numpy as np
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.animation as animation
from shapely.geometry.polygon import Polygon

def weight_matrix(d_matrix, N, N1, distance_bw_UAVs):
	dij = np.zeros((N + 1, N + 1))
	for i in range(N):
		for j in range(N):
			dij[i][j] = d_matrix[i][j]
			#print(dij[i][j], "\n")

	for i in range(N):
		dij[i][N] = d_matrix[N1][i]

	dij = dij * distance_bw_UAVs

	return dij

def Load_files():
	# To load all the hyperparameters
	parameter_file = open('number_of_UAVs', 'rb')
	parameters = pickle.load(parameter_file)

	# Load the weight matrix file
	d_matrix_file = open('weight_matrix', 'rb')
	matrix = pickle.load(d_matrix_file)

	# Load the flight waypoints
	# wp_file = open("wp_list", 'rb')
	# WP_list = pickle.load(wp_file)
	WP_list = [[-7,8],[7.6,4.3],[3.2,9.5]]
	#WP_list = [[7.6,4.3],[3.2,9.5]]

	N = int(parameters[0])
	N_f = int(parameters[1])-1
	D_wm = int(parameters[2])

	Weight_matrix = weight_matrix(matrix, N, N_f, D_wm)

	return N, N_f, Weight_matrix, WP_list

def rescale_vector(v, v_max, v_min):
	v_mod = np.linalg.norm(v)
	try:
		v = (v/v_mod)*min(max(v_mod,v_min),v_max)
	except:
		return v

	return v