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


# root_observation = env.reset()
# q_value = forward(root_observation)

action_seq_list = product((0, 1, 2, 3, 4, 5), repeat=3)

def f_rand_14738465(seq):
    iter1 = iter(seq)
    iter2 = iter(seq)
    next(iter2)
    return all(a != b for a, b in zip(iter1, iter2))


action_seq_list = filter(f_rand_14738465, action_seq_list)
action_seq_list = list(action_seq_list)


def run(env, seq, last_obs):
    tmp_env = deepcopy(env)

    memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)

    # first run the seq, notice might have a early stop
    for i, a in enumerate(seq):
        obs, r, done, _ = tmp_env.step(a)
        memory.append(last_obs, a, r, done, training=False)
        last_obs = obs
        if done:
            return tmp_env.game.step

    # run model until done
    while not done:
        q_value = forward(obs)
        a = np.argmax(q_value)
        obs, r, done, _ = tmp_env.step(a)
        memory.append(last_obs, a, r, done, training=False)
        last_obs = obs
        if tmp_env.game.step > 40:
            print("ERROR:", tmp_env.game.all_step)
            exit()

    return tmp_env.game.step, tmp_env.game.all_step

a_list = []
while True:
    root_obs = env.reset()

    best_path_len, path = min([run(env, one, root_obs) for one in action_seq_list], key=lambda x: x[0])

    memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)
    obs = root_obs
    last_obs = root_obs
    done = False
    # run model until done
    while not done:
        q_value = forward(obs)
        a = np.argmax(q_value)
        obs, r, done, _ = env.step(a)
        memory.append(last_obs, a, r, done, training=False)
        last_obs = obs
        if env.game.step > 40:
            print("ERROR:", env.game.all_step)
            exit()

    print(best_path_len, env.game.step, best_path_len <= env.game.step)
    print(path, env.game.all_step)


    # a_list.append(best_path_len)
    # print(
    #     sum(a_list) / len(a_list),
    #     len(a_list)
    # )
