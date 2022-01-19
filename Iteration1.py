import os
import sys
import gym
import time
import random
import logging
import itertools
import numpy as np
from utils import *
import argparse as ap
#from Environment import Swarm
from Environment_iteration6 import Swarm
import tensorflow as tf
from gym import wrappers
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers, optimizers, losses, models

time_var = time.localtime()
name = f'UAVSwarm_{time_var.tm_mon}_{time_var.tm_mday}_{time_var.tm_hour}_{time_var.tm_min}_{time_var.tm_sec}'
summary_writer = tf.summary.create_file_writer(logdir = f'Logs/{name}/')

def Model(input_shape, output_shape, input_activation, output_activation, hidden_layers):
	model = models.Sequential()

	model.add(layers.Dense(hidden_layers[0], input_shape = input_shape, activation = input_activation))
	model.add(layers.Dense(hidden_layers[1], activation = input_activation))

	model.add(layers.Dense(output_shape, activation = output_activation))

	return model

def copy_weights(Copy_from, Copy_to, constant):
	variables2 = Copy_from.trainable_variables
	variables1 = Copy_to.trainable_variables
	for v1, v2 in zip(variables1, variables2):
		v1.assign((1-constant)*v2.numpy() + constant*v1.numpy())

def return_func(rews, discount):
	n = len(rews)
	rtgs = np.zeros_like(rews, dtype = 'float32')
	for i in reversed(range(n)):
		rtgs[i] = rews[i] + (discount*rtgs[i+1] if i+1 < n else 0)
	return rtgs

class ReplayMemory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.push_count = 0

	def push(self, experience):
		if len(self.memory)<self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

class TD3():
	"""Class to train a policy network using the DDPG algorithm"""
	def __init__(self, env_name, env, Actor_net, Critic_net1, Critic_net2, act_dim, obs_dim ,lam = 0.97,decay = False,
				Actor_lr = 0.0001, gamma = 0.99, delta = 0.01, Critic_lr = 0.001, render = False, epoch_steps = 2000, 
				value_train_iterations = 5, memory_size = 1000, polyak_const = 0.995, minibatch_size = 100):
		
		self.env_name = env_name
		self.env = env
		self.gamma = gamma
		self.Actor = Actor_net
		self.Critic1 = Critic_net1
		self.Critic2 = Critic_net2
		self.act_dim = act_dim
		self.obs_dim = obs_dim
		
		self.Actor_optimizer = optimizers.Adam(lr = Actor_lr)
		self.Critic_optimizer1 = optimizers.Adam(lr = Critic_lr)
		self.Critic_optimizer2 = optimizers.Adam(lr = Critic_lr)

		self.render = render
		self.lam = lam
		self.value_train_iterations = value_train_iterations

		self.Experience = namedtuple('Experience', ['states','actions', 'rewards', 'next_states', 'dones'])
		self.memory_size = memory_size
		self.memory = ReplayMemory(self.memory_size)

		self.Target_Actor = models.clone_model(self.Actor)
		self.Target_Critic1 = models.clone_model(self.Critic1)
		self.Target_Critic2 = models.clone_model(self.Critic2)

		self.minibatch_size = minibatch_size
		self.polyak_const = polyak_const
		self.act_limit = 2#self.env.action_space.high[0]
		self.policy_delay = 2
		self.decay = decay
		self.epsilon = 0.5

		_, _, self.WM, _ = Load_files()
		print(type(self.WM))
		print(self.WM.shape)
		print(self.env.N)
		print(self.env.N_f)
		print(self.env.WP_list)

	def select_action(self, state):
		state = np.atleast_2d(state).astype('float32')
		action = self.Actor(state) * self.act_limit
		return action

	def get_distance(self, point1, point2):
		return np.linalg.norm(point1-point2)

	def select_action2(self, state):
		state = np.array(state)
		state = state.reshape((self.env.N + 1,2))
		
		v = []
		n = self.env.N
		goal_pos = state[n]
		for i in range(n):
			action = [0,0]

			pos1 = state[i]
			for j in range(n):
				if i==j:
					continue
				pos2 = state[j]
				action += (pos2-pos1)*(1-self.WM[i][j]/self.get_distance(pos1,pos2))

			action2 = (goal_pos-pos1)*(1-self.WM[i][n]/self.get_distance(goal_pos, pos1))
			action = rescale_vector(action, 2, 0)
			action2 = rescale_vector(action2, 2, 0)

			action += action2
			action = rescale_vector(action, 2, 0)

			v.append(action)
		v = np.ndarray.flatten(np.array(v))

		#print(v)

		#action = np.random.uniform(-2, 2, (3*2))
		return v

	def add_noise(self, action, step, decay = False, noise_scale = 0.1):
		"""Function to add OU noise to the action for exploration"""
		if decay:
			noise_scale *= 1/((step+1)**0.5)
		action += noise_scale * np.random.randn(self.act_dim)
		action = np.squeeze(np.clip(action, -self.act_limit, self.act_limit))
		return action

	def close(self):
		"""This function closes all the running environments""" 
		self.env.close()

	def render_episode(self, n=1):
		"""Renders n episodes using the current policy network"""
		for i in range(n):
			state = self.env.reset()
			total_reward = 0
			done = False
			while not done:
				self.env.render()
				action = self.select_action2(state)
				next_state, reward, done, _ = self.env.step(action)
				state = next_state
				total_reward += reward
			print(f"Total Reward accumulated: {total_reward}")

	def load_weights_test(self, path):
		# Load saved weights to test the algorithm
		self.Actor.load_weights(path)

	def load_weights_train(self, actor_path, c1_path, c2_path):
		self.Actor.load_weights(actor_path)
		self.Critic1.load_weights(c1_path)
		self.Critic2.load_weights(c2_path)

	def make_video(self, path, n=1):
		env = wrappers.Monitor(self.env, path, force=True)
		
		for _ in range(n):
			rewards = 0
			steps = 0
			done = False
			observation = env.reset()
			while not done:
				action = self.select_action(observation)
				observation, reward, done, _ = env.step(action)
				steps += 1
				rewards += reward
			print("Testing steps: {} rewards {}: ".format(steps, rewards))

	def update_critic(self, states, actions, rewards, next_states, dones, step):
		with tf.GradientTape() as critic_tape1, tf.GradientTape() as critic_tape2:
			# Represent the actions suggested by the target Actor network
			target_actions = self.Target_Actor(next_states)*self.act_limit
			# Adding noise to the target actions
			target_actions = self.add_noise(target_actions, step, self.decay)
			# Represent the target Q values for the target actions by the target Critic network
			target_Q_values1 = self.Target_Critic1(tf.concat([next_states, target_actions], axis = -1))
			target_Q_values2 = self.Target_Critic2(tf.concat([next_states, target_actions], axis = -1))
			#Find minimum of the target Q values
			target_Q_values = tf.minimum(target_Q_values1, target_Q_values2)
			# Bellman backup	
			backup = tf.stop_gradient(rewards + self.gamma*(1-dones)*target_Q_values)
			# Represent the Q values suggested by the Critic Network
			Q_values1 = self.Critic1(tf.concat([states, actions], axis = -1))
			Q_values2 = self.Critic2(tf.concat([states, actions], axis = -1))
			# Loss function to update the critic network
			critic_loss1 = tf.reduce_mean((backup - Q_values1)**2)
			critic_loss2 = tf.reduce_mean((backup - Q_values2)**2)
		
		critic_loss = critic_loss1 + critic_loss2

		critic_gradients1 = critic_tape1.gradient(critic_loss1, self.Critic1.trainable_variables)
		self.Critic_optimizer1.apply_gradients(zip(critic_gradients1, self.Critic1.trainable_variables))
		critic_gradients2 = critic_tape2.gradient(critic_loss2, self.Critic2.trainable_variables)
		self.Critic_optimizer2.apply_gradients(zip(critic_gradients2, self.Critic2.trainable_variables))

		return critic_loss

	def update_actor(self, states, actions, rewards, next_states, dones):
		with tf.GradientTape() as actor_tape:
			# Represents the true actions the Actor network would take
			# Since the actions stored above also contains actions selected due to random noise function N
			true_actions = self.Actor(states)*self.act_limit
			# Represent the true Q_values using the Critic network
			true_Q_values = self.Critic1(tf.concat([states, true_actions], axis = -1))
			# Loss function to update the Actor network
			actor_loss = -tf.reduce_mean(true_Q_values)

		actor_gradients = actor_tape.gradient(actor_loss, self.Actor.trainable_variables)
		self.Actor_optimizer.apply_gradients(zip(actor_gradients, self.Actor.trainable_variables))

		return actor_loss

	def update_target_networks(self):
		# Updating the target Actor and target Critic networks
		copy_weights(self.Actor, self.Target_Actor, self.polyak_const)
		copy_weights(self.Critic1, self.Target_Critic1, self.polyak_const)
		copy_weights(self.Critic2, self.Target_Critic2, self.polyak_const)

	def train_step(self, episode):
		if self.render and episode%2==0:
			self.render_episode()

		# Only for Env 3 iteration at 9.26 jun 30
		if episode>2000:
			self.epsilon=0.2
		elif episode>5000:
			self.epsilon=0.1
		elif episode>10000:
			self.epsilon=0.05

		done = False
		state = self.env.reset()
		total_reward = 0
		actor_losses = np.array([])
		critic_losses = np.array([])
		for t in itertools.count():

			if episode<10:
				#action = self.env.action_sample()
				action = self.select_action2(state)
			else:
				rand_prob = np.random.rand()
				if self.epsilon>rand_prob:
					action = self.env.action_sample()
					#action = self.select_action2(state)
				else:
					action = self.select_action(state)
					action = self.add_noise(action, t, self.decay)

			next_state, reward, done, _ = self.env.step(action)
			total_reward += reward
			self.memory.push(self.Experience(state, action, [reward], next_state, [done]))
			state = next_state

			if self.memory.can_provide_sample(self.minibatch_size):
				experiences = self.memory.sample(self.minibatch_size)
				batch = self.Experience(*zip(*experiences))

				states = np.asarray(batch[0]).astype('float32')
				actions = np.asarray(batch[1])
				rewards = np.asarray(batch[2])
				next_states = np.asarray(batch[3]).astype('float32')
				dones = np.asarray(batch[4])

				critic_loss = self.update_critic(states, actions, rewards, next_states, dones, t)

				if t%self.policy_delay == 0:
					actor_loss = self.update_actor(states, actions, rewards, next_states, dones)
					self.update_target_networks()
				else:
					actor_loss = 0
					
				# Good old book-keeping
				actor_losses = np.append(actor_losses, actor_loss)
				critic_losses = np.append(critic_losses, critic_loss)

			if done:
				break

		with summary_writer.as_default():
			tf.summary.scalar("reward", total_reward, step=episode)
			tf.summary.scalar("actor_loss", np.mean(actor_losses), step=episode)
			tf.summary.scalar("critic_loss", np.mean(critic_losses), step=episode)

		print(f"Ep:{episode} total_reward:{total_reward:0.2f} critic_loss:{np.mean(critic_losses):0.2f} actor_loss:{np.mean(actor_losses):0.2f}")	

	def train(self, episodes):
		print(f"Starting training, saving checkpoints and logs to: {name}")

		for episode in range(episodes):
			self.train_step(episode)

			if episode%10 == 0 and episode != 0:
				self.Actor.save_weights(f"Weights/{name}/Episode{episode}/actor.ckpt")
				self.Critic1.save_weights(f"Weights/{name}/Episode{episode}/critic1.ckpt")
				self.Critic2.save_weights(f"Weights/{name}/Episode{episode}/critic2.ckpt")

if __name__ == "__main__":
	# Parsing the arguments
	parser = ap.ArgumentParser()
	parser.add_argument("-tr", "--train", action="store_true", help = "To train the model")
	parser.add_argument("-ts", "--test", action="store_true", help = "To test the model")
	parser.add_argument("-l", "--load", action="store_true", help = "To test the model")
	parser.add_argument("-i", "--iterations", type=int, help = "To test the model")
	parser.add_argument("-ap", "--a_path", type=str, help = "To test the model")
	parser.add_argument("-c1p", "--c1_path", type=str, help = "To test the model")
	parser.add_argument("-c2p", "--c2_path", type=str, help = "To test the model")

	args = parser.parse_args()
	env_name = "UAVSwarm"
	env = Swarm()
	No_of_UAVs = env.N
	obs_dim = No_of_UAVs*2 + 2
	act_dim = No_of_UAVs*2

	# Policy Model
	Actor_net = Model((obs_dim,), act_dim, 'relu', 'tanh', [400,300])
	#print(Actor_net.summary())

	# Q-value Model
	Critic_net1 = Model((obs_dim+act_dim,), 1, 'relu', 'linear', [400,300])
	Critic_net2 = Model((obs_dim+act_dim,), 1, 'relu', 'linear', [400,300])
	#print(Critic_net.summary())

	agent = TD3(env_name, env, Actor_net, Critic_net1, Critic_net2, decay = False, render=False, act_dim = act_dim, obs_dim = obs_dim)

	if args.test:
		# path = args.a_path
		# agent.load_weights_test(path)
		#agent.make_video(path, int(sys.argv[5]))
		agent.render_episode(5)
	elif args.train:
		if args.load:
			a_path = args.a_path
			c1_path = args.c1_path
			c2_path = args.c2_path
			agent.load_weights_train(a_path, c1_path, c2_path)
		episodes = args.iterations
		agent.train(episodes)

	agent.close()