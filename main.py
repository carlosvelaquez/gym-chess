import sys
import random
import numpy as np
import gym
import gym_chess

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

env = gym.make('ChessVsSelf-v0')
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer=Adam())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# policy = EpsGreedyQPolicy()
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
# ), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=100, target_model_update=1e-2)
dqn.compile(optimizer=Adam(), metrics=['accuracy', 'mse'])
dqn.fit(env, nb_steps=100000, visualize=True, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format("chess"), overwrite=True)
dqn.test(env, nb_episodes=10, visualize=True)
