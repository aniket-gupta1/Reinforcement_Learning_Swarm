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
		self.N, self.N_f, self.Weight_matrix, self.WP_list = Load_files()

		self.wp_rad = 0.5
		self.counter = 0
		self.render_var = render_var
		self.v_max = v_max
		self.v_min = v_min
		self.max_steps = 400
		self.wp_update_var = 0
		self.safe_distance = safe_distance
		self.timestep = 0.1

		self.done = False

		if self.render_var:
			self.show_plot()

		# Define reward constants
		self.goal_reward_const = 1
		self.formation_reward_const = 1

		# Define rewards
		self.goal_reward = 10
		self.formation_reward = 1
		self.collision_penalty = -1

		self.const = 15
		self.boundary_points = [(self.const,self.const),(-self.const,self.const),(-self.const,-self.const),(self.const,-self.const)]
		self.start_location = np.array([[i,np.random.randint(3)] for i in range(self.N)]).astype('float64')
		
		# Iterators for storing the position of agents
		self.pos = self.start_location
		self.pos_old = self.start_location

	def show_plot(self):
		plt.show()

	def get_distance(self, point1, point2):
		return np.linalg.norm(point1-point2)

	def restore_start_location(self):
		# Restore the original values of pos
		self.WP_list = list(np.random.permutation([[-8,9],[-8,-9],[8,-9],[8,9]]))
		self.pos = copy.copy(self.start_location)
		self.pos_old = copy.copy(self.start_location)
		self.wp_update_var = 0

	def reset(self):
		self.restore_start_location()

		goal_pos = self.get_current_waypoint()
		state = list()

		for pos1 in self.pos:
			state.append(pos1)

		state.append(goal_pos)
		state = list(np.ndarray.flatten(np.array(state)))
		
		return state
	
	def render(self):
		# wap = self.get_current_waypoint()
		# x,y = [pos[0] for pos in self.pos]+[wap[0]], [pos[1] for pos in self.pos]+[wap[1]]
		# plt.clf()
		# plt.axis([-10, 10, -10, 10])
		# plt.scatter(x,y)
		# plt.pause(0.001)
		wap = self.get_current_waypoint()
		x,y = [pos[0] for pos in self.pos], [pos[1] for pos in self.pos]
		plt.clf()
		plt.axis([-self.const, self.const, -self.const, self.const])
		plt.scatter(x,y)
		plt.scatter(wap[0], wap[1], color='red')
		plt.pause(0.001)

	def action_sample(self):
		return np.random.uniform(-self.v_max, self.v_max, (self.N*2))		

	def boundary_check(self):
		var = False

		point = [Point(pos[0], pos[1]) for pos in self.pos]
		polygon = Polygon(self.boundary_points)

		var = np.mean([not polygon.contains(p) for p in point])

		if not var:
			return False
		else:
			return True

	def get_current_waypoint(self):
		return self.WP_list[self.wp_update_var]

	def update_pos(self, v):
		self.pos_old = copy.copy(self.pos)
		self.pos += v*self.timestep
		#time.sleep(0.1)

	def step(self, v):
		self.counter+=1
		#print(f"Step: {self.counter}")
		goal_pos = self.get_current_waypoint()					# Waypoint to be followed
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

		# Find the value of next_state and reward
		for (i, pos1), pos1_old in zip(enumerate(self.pos), self.pos_old):
			for (j, pos2), pos2_old in zip(enumerate(self.pos), self.pos_old):
				if i==j:
					continue

				dist = self.get_distance(pos1,pos2)

				reward += self.formation_reward_const/(self.N + abs(dist-self.Weight_matrix[i][j]))
				
				if dist<self.safe_distance:
					reward += self.collision_penalty

			state.append(pos1)
		state.append(goal_pos)
		state = list(np.ndarray.flatten(np.array(state)))

		# Goal position reward
		temp_var=0
		for (i, pos1), pos1_old in zip(enumerate(self.pos), self.pos_old):
			goal_distance = self.get_distance(pos1, goal_pos)

			if abs(goal_distance-self.Weight_matrix[i][self.N])<=self.wp_rad:
				temp_var+=1
			
			goal_distance_old = self.get_distance(pos1_old, goal_pos)
			reward += self.goal_reward_const/(self.N + abs(goal_distance-self.Weight_matrix[i][self.N]-self.wp_rad))
			#reward += self.goal_reward_const * (goal_distance_old - goal_distance)
			#reward += (self.goal_reward_const/self.N) * abs(goal_distance-self.Weight_matrix[i][self.N])
			
			if temp_var==self.N:
				print("GOAL")
				reward += 10
				
				self.wp_update_var+=1
				if self.wp_update_var == len(self.WP_list):
					self.done = True
					print("END GOAL")
					self.wp_update_var-=1

		#print(f"Step: {self.counter}| Reward:{reward}")
		return state, reward, self.done, "gibberish"

	def close(self):
		plt.close()

if __name__ == '__main__':
	a = Swarm()
	plt.show()