import copy
import time
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.animation as animation
from shapely.geometry.polygon import Polygon

# Formation reward changed to a negative function based on distance from mean center
class Swarm(object):
	"""
	"""
	def __init__(self, v_max = 2, v_min = 0, safe_distance = 0.5, render_var=False):
		self.N = 10

		self.wp_rad = 0.5
		self.counter = 0
		self.render_var = render_var
		self.v_max = v_max
		self.v_min = v_min
		self.max_steps = 500
		self.safe_distance = safe_distance
		self.timestep = 0.1
		self.FOV_x = 1.1
		self.FOV_y = 0.6

		self.done = False

		if self.render_var:
			self.show_plot()

		# Define reward constants
		self.goal_reward_const = -1

		# Define rewards
		self.goal_reward = 10
		self.target_reward = 1
		self.collision_penalty = -10

		self.boundary_points = [(15,15),(-5,15),(-5,-5),(15,-5)]
		self.start_location = np.array([[i+1,0] for i in range(self.N)]).astype('float64')
		self.end_location = np.array([[i+1,10] for i in range(self.N)]).astype('float64')		
		
		# Create random targets
		self.target_location = np.array([[4,4], [5,5], [5,7], [7,4], [1,7], [2,5.5], [2.5,8], [8,4], [7.5,6], [9,5]])
		
		# Iterators for storing the position of agents
		self.pos = self.start_location

	def show_plot(self):
		plt.show()

	def get_distance(self, point1, point2):
		return np.linalg.norm(point1-point2)

	def get_rectangle_points(self, c, l, b):
		points = []
		points.append([c[0]+l/2, c[1]+b/2])
		points.append([c[0]-l/2, c[1]+b/2])
		points.append([c[0]-l/2, c[1]-b/2])
		points.append([c[0]+l/2, c[1]-b/2])
		return points

	def check_in_FOV(self, pos):
		FOV_points = self.get_rectangle_points(pos, self.FOV_x, self.FOV_y)

		point = [Point(t[0], t[1]) for t in self.target_location]
		polygon = Polygon(FOV_points)
		var = np.sum([polygon.contains(p) for p in point])
		return var

	def restore_start_location(self):
		# Restore the original values of pos
		self.pos = copy.copy(self.start_location)

	def reset(self):
		self.restore_start_location()

		state = list()

		for pos1 in self.pos:
			state.append(pos1)

		state += list(self.end_location)

		state = list(np.ndarray.flatten(np.array(state)))

		return state
	
	def render(self):
		x,y = [pos[0] for pos in self.pos], [pos[1] for pos in self.pos]
		x_end, y_end = [end[0] for end in self.end_location], [end[1] for end in self.end_location]
		x_target, y_target = [target[0] for target in self.target_location], [target[1] for target in self.target_location]
		plt.clf()
		plt.axis([-5, 15, -5, 15])
		plt.scatter(x,y)
		plt.scatter(x_end, y_end, color = 'red')
		plt.scatter(x_target, y_target, color = 'black')
		plt.pause(0.001)

	def action_sample(self):
		return np.random.uniform(-self.v_max, self.v_max, (self.N*2))		

	def boundary_check(self):
		var = False

		point = [Point(pos[0], pos[1]) for pos in self.pos]
		polygon = Polygon(self.boundary_points)
		truth_mat = [not polygon.contains(p) for p in point]
		var = np.mean(truth_mat)

		if not var:
			return False
		else:
			return True

	def update_pos(self, v):
		self.pos += v*self.timestep
		#time.sleep(0.1)

	def step(self, v):
		self.counter+=1
		#print(f"Step: {self.counter}")
		state = list()											# distance list (input to the actor network)
		reward = 0												# intialize reward variable
		if self.done:
			self.done = False
			self.restore_start_location()

		# End episode if max number of steps are reached
		if self.counter%self.max_steps == 0:
			self.counter = 0
			self.done = True

		# Check if all agents are within environment boundaries
		var = self.boundary_check()
		if var:
			self.done = True
			self.counter=0

		# Reshape and limit the velocity vector
		v = np.reshape(v, (self.N,2))
		v = rescale_vector(v, self.v_max, self.v_min)

		# Update vehicle position using current velocity
		self.update_pos(v)
		
		# Goal position reward
		for i, pos1 in enumerate(self.pos):
			goal_distance = self.get_distance(pos1, self.end_location[i])

			detected_targets = self.check_in_FOV(pos1)

			reward += self.target_reward * detected_targets

			if goal_distance<=self.wp_rad:
				reward += self.goal_reward
			else:
				reward += self.goal_reward_const * abs(goal_distance-self.wp_rad)

			for j, pos2 in enumerate(self.pos):
				if i==j:
					continue

				dist = self.get_distance(pos1,pos2)

				if dist<self.safe_distance:
					reward += self.collision_penalty		

			state.append(pos1)

		state += list(self.end_location)
		state = list(np.ndarray.flatten(np.array(state)))

		#print(f"Step: {self.counter}| Reward:{reward}")
		return state, reward, self.done, "gibberish"

	def close(self):
		plt.close()

if __name__ == '__main__':
	a = Swarm()
	plt.show()