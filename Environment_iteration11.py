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
	def __init__(self, v_max = 2, v_min = 0, safe_distance = 1, render_var=False):
		self.N, self.N_f, self.Weight_matrix, self.WP_list = Load_files()

		self.wp_rad = 0.5
		self.counter = 0
		self.render_var = render_var
		self.v_max = v_max
		self.v_min = v_min
		self.max_steps = 1000
		self.wp_update_var = 0
		self.safe_distance = safe_distance
		self.timestep = 0.1

		self.done = False

		if self.render_var:
			self.show_plot()

		# Define reward constants
		self.goal_reward_const = -1
		self.formation_reward_const = -1

		# Define rewards
		self.goal_reward = 10
		self.formation_reward = 1
		self.collision_penalty = -100

		self.const = 30
		self.boundary_points = [(self.const,self.const),(-self.const,self.const),(-self.const,-self.const),(self.const,-self.const)]
		self.start_location = np.array([[i,np.random.randint(3)] for i in range(self.N)]).astype('float64')
		
		# Iterators for storing the position of agents
		self.pos = copy.copy(self.start_location)
		self.pos_old = self.start_location

		self.discard_list = []
		self.record_x = copy.copy(list(self.start_location[:,0]))
		self.record_y = copy.copy(list(self.start_location[:,1]))

	def show_plot(self):
		plt.show()

	def get_distance(self, point1, point2):
		return np.linalg.norm(point1-point2)

	def restore_start_location(self):
		# Restore the original values of pos
		#self.WP_list = list(np.random.permutation([[-8,9],[-8,-9],[8,-9],[8,9]]))
		temp_var = 19
		self.WP_list = list(np.random.permutation([[-temp_var,temp_var],[-temp_var,-temp_var]
												,[temp_var,-temp_var],[temp_var,temp_var]]))
		#self.WP_list = list([[50,50]])
		self.pos = copy.copy(self.start_location)
		self.pos_old = copy.copy(self.start_location)
		self.wp_update_var = 0
		self.discard_list.clear()

	def reset(self):
		self.restore_start_location()

		goal_pos = self.get_current_waypoint()
		state = list()

		for pos1 in self.pos:
			state.append(pos1)

		state.append(goal_pos)
		state = list(np.ndarray.flatten(np.array(state)))
		
		return state
	
	def render(self, ep):
		# wap = self.get_current_waypoint()
		# x,y = [pos[0] for pos in self.pos]+[wap[0]], [pos[1] for pos in self.pos]+[wap[1]]
		# plt.clf()
		# plt.axis([-10, 10, -10, 10])
		# plt.scatter(x,y)
		# plt.pause(0.001)
		wap = self.get_current_waypoint()
		x,y = [pos[0] for pos in self.pos], [pos[1] for pos in self.pos]
		plt.clf()
		#plt.axis([-self.const, self.const, -self.const, self.const])
		plt.axis([-self.const, self.const, -self.const, self.const])
		plt.scatter(x,y, color = "#036016")
		# plt.scatter(x[8], y[8], marker='s')
		# plt.scatter(x[6], y[6], marker='s')
		plt.scatter(self.record_x, self.record_y, marker = '.', color = "#069E2D", alpha = 0.4)
		plt.scatter(wap[0], wap[1], color='red', marker='*')
		plt.pause(0.001)

		# if ep%20==0:
		# 	if len(self.record_x)>30:
		# 		del self.record_x[0:9]
		# 		del self.record_y[0:9]
		# 	self.record_x+=x
		# 	self.record_y+=y

		if ep%10==0:
			if len(self.record_x)>self.N*5:
				for i in range(self.N):
					self.record_x.pop(i)
					self.record_y.pop(i)

			self.record_x+=x
			self.record_y+=y
		# #00aeff #3bc1ff #84d8ff #b3e7ff #d0f0ff

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
			print("Max-steps reached")
			self.counter = 0
			self.done = True

		# Check if all agents are within environment boundaries
		# var = self.boundary_check()
		# if var:
		# 	self.done = True
		# 	self.counter=0

		# Reshape and limit the velocity vector
		v = np.reshape(v, (self.N,2))
		v = rescale_vector(v, self.v_max, self.v_min)

		# Update vehicle position using current velocity
		self.update_pos(v)

		temp_var=0
		# Find the value of next_state and reward
		for (i, pos1), pos1_old in zip(enumerate(self.pos), self.pos_old):
			# Calculate formation reward
			for (j, pos2), pos2_old in zip(enumerate(self.pos), self.pos_old):
				if i==j:
					continue
				dist = self.get_distance(pos1,pos2)

				if abs(dist-self.Weight_matrix[i][j])<=0.2:
					reward += 0.1
				elif dist<self.safe_distance:
					reward += -100

			# Goal Reward
			goal_distance = self.get_distance(pos1, goal_pos)

			#goal_distance_old = self.get_distance(pos1_old, goal_pos)
			if abs(goal_distance-self.Weight_matrix[i][self.N])<=self.wp_rad and i not in self.discard_list:
				#temp_var+=1
				self.discard_list.append(i)
				#print("ADDED")
				reward += 50
			else:
				reward += -0.5

			#if temp_var==self.N:
			if len(self.discard_list)==self.N:
				print("GOAL", end = " ")

				self.discard_list.clear()
				self.wp_update_var+=1
				if self.wp_update_var == len(self.WP_list):
					self.done = True
					print("END GOAL", end = " ")
					self.wp_update_var-=1

			state.append(pos1)
		state.append(goal_pos)
		state = list(np.ndarray.flatten(np.array(state)))

		#print(f"Step: {self.counter}| Reward:{reward}")
		return state, reward, self.done, "gibberish"

	def close(self):
		plt.close()

if __name__ == '__main__':
	a = Swarm()
	plt.show()