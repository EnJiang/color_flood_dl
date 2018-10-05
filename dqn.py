import numpy as np
import gym

from env import ColorFlood

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from nn import ResNet

ENV_NAME = 'ColorFlood'

# Get the environment and extract the number of actions.
env = ColorFlood()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
INPUT_SHAPE = (6, 6)
WINDOW_LENGTH = 4

input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE

model = ResNet.build(
    INPUT_SHAPE[0],
    INPUT_SHAPE[1],
    WINDOW_LENGTH,
    nb_actions,
    stages=[3, 4, 6],
    filters=[64, 128, 256, 512],
    dataset="ColorFlood")

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
    nb_steps_warmup=10000,
    gamma=.99,
    target_model_update=10000,
    train_interval=4,
    enable_dueling_network=True,
    delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
for _ in range(100):
    dqn.fit(env, nb_steps=330000, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=33, visualize=False, nb_max_episode_steps=1000)