### MDP Value Iteration and Policy Iteration
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	V = np.zeros(nS, dtype=np.float)
	for i in range(max_iteration):
		V_last = V.copy() # store last V for compare
		for state in range(nS): # for each state
			action = policy[state] # pick the action
			# take the action
			tmp = 0
			for prob, nextstate, reward, terminal in P[state][action]:
				# calc new value V_new by the formula
				tmp += prob * (reward + gamma * V_last[nextstate]) 
			V[state] = tmp
		if np.all( np.abs(V_last - V) < tol ):
			break
	############################
	return V

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	for state in range(nS):
		best_policy_Q = -np.inf
		for action in range(nA):
			tmp = 0
			for prob, nextstate, reward, terminal in P[state][action]:
				tmp += prob * ( reward + gamma * value_from_policy[nextstate] )
			if tmp>best_policy_Q:
				best_policy_Q = tmp
				policy[state] = action
	############################
	return policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS, dtype=np.float)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	for i in range(max_iteration):
		policy_last = policy.copy()
		V = policy_evaluation(P, nS, nA, policy)
		policy = policy_improvement(P, nS, nA, V, policy)
		if np.all(policy == policy_last):
			break
	############################
	if(i == max_iteration-1):
		print('Policy Iteration Max Iter!')

	return V, policy

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS, dtype=np.float)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	for i in range(max_iteration):
		V_last = V.copy()
		for state in range(nS):
			max_Q = -np.inf
			for action in range(nA):
				tmp = 0
				for prob, nextstate, reward, terminal in P[state][action]:
					tmp += prob * ( reward + gamma * V_last[nextstate])
				if tmp>max_Q:
					max_Q = tmp
					policy[state] = action
			V[state] = max_Q
		if np.all(  np.abs(V-V_last) < tol ):
			break
	############################
	return V, policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = {
		'D4':gym.make("Deterministic-4x4-FrozenLake-v0"), 
		'D8':gym.make("Deterministic-8x8-FrozenLake-v0"),
		'S4':gym.make("Stochastic-4x4-FrozenLake-v0"),
	}
	print('===================')
	for name,env in env.items():
		s = time.time()
		value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
		e = time.time()
		t = (e-s)*1000
		print('vi in {} time {:>1.3f}ms'.format(name,t))
		s = time.time()
		policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
		e = time.time()
		t = (e-s)*1000
		print('pi in {} time {:>1.3f}ms'.format(name,t))

