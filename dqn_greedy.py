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

from itertools import product

from copy import deepcopy

ENV_NAME = 'ColorFlood'

env = ColorFlood(size=12)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

INPUT_SHAPE = (12, 12)
WINDOW_LENGTH = 4

input_shape = (WINDOW_LENGTH, ) + INPUT_SHAPE

model = resnet(input_shape, nb_actions)
model.summary()

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()

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

dqn.load_weights("model/resnet_ddqn_size12.h5f")

# fetch the model
model = dqn.model
model.trainable = False

def forward(obs):
    state = memory.get_recent_state(obs)
    state = np.array(state)
    state = np.expand_dims(state, 0)
    q_value = model.predict(state)
    q_value = np.reshape(q_value, (-1))
    return q_value

root_observation = env.reset()
q_value = forward(root_observation)


action_seq_list = product((0, 1, 2, 3, 4, 5), repeat=4)
def f_rand_14738465(seq):
    iter1 = iter(seq)
    iter2 = iter(seq)
    next(iter2)
    for a, b in zip(iter1, iter2):
        if a == b:
            return False
    return True
action_seq_list = filter(f_rand_14738465, action_seq_list)
action_seq_list = list(action_seq_list)

def run(env, seq):
    # print("test", seq)
    tmp_env = deepcopy(env)
    for i, a in enumerate(seq):
        obs, r, done, _ = tmp_env.step(a)
        if done:
            return i, tmp_env.game.target_area, np.max(forward(obs))
    return i, tmp_env.game.target_area, np.max(forward(obs))

a_list = []
while True:
    done = False
    env.reset()
    while True:
        best_area = -np.inf
        best_q = -np.inf
        best_seq = None
        best_middle_cut = np.inf
        for seq in action_seq_list:
            middle_cut, area, q = run(env, seq)
            if area < env.game.size**2: # not done:
                if q > best_q and best_area < env.game.size**2:
                    best_q = q
                    best_area = area
                    best_seq = seq
            else: # done
                if best_area == env.game.size ** 2: # last is also done
                    if middle_cut < best_middle_cut: # compare middle cut
                        best_middle_cut = middle_cut
                        best_seq = seq
                else: # last is not done
                    best_area = area
                    best_seq = seq
        # got best seq, run till done
        for a in best_seq:
            obs, r, done, _ = env.step(a)
            # print(env.game, done)
            if done:
                break

        # outer loop, if done, record
        if done:
            a_list.append(env.game.step)
            print(sum(a_list) / len(a_list), len(a_list))
            break
