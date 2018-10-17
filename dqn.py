import numpy as np
import gym

from env import ColorFlood

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from nn import resnet

ENV_NAME = 'ColorFlood'

# Get the environment and extract the number of actions.
env = ColorFlood(size=12)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
INPUT_SHAPE = (12, 12)
WINDOW_LENGTH = 8

input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE

model = resnet(input_shape, nb_actions)
model.summary()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()

# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    nb_steps_warmup=50000,
    gamma=.99,
    target_model_update=10000,
    train_interval=4,
    enable_dueling_network=True,
    dueling_type='avg',
    delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
for _ in range(500):
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights("model/resnet_ddqn_size12.h5f", overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=1000)