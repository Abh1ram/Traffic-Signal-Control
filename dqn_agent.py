# Code for DQN
import pickle
import random

import numpy as np

from collections import deque
from pathlib import Path

from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam

EXP_TABLE_FILE = "./dqn_exp_table.p"
MODEL_FILE = "./my_model.h5"

class DQN_Agent:
    def __init__(self, rew_attr="q_len", Lnorm=1, discount_rate=0.9,
               learning=True, learning_rate=0.2, exploration_eps=0.3,
               exp_table_sz=5000):
        # set the reward parameters
        self.rew_attr = rew_attr
        self.Lnorm = Lnorm
        # set the learning parameters
        self.learning = learning
        self.alpha = learning_rate
        self.gamma = discount_rate
        self.eps = exploration_eps

        self.create_exp_table()
        self.exp_table_sz = exp_table_sz

        self.state_size = 3
        self.action_size = 2
        self.model = self.create_net()
        # updating the q table - SARS'
        self.time_slice = 0
        self.old_state = None
        self.action = None

    def create_exp_table(self):
        self.exp_table = deque()
        if Path(EXP_TABLE_FILE).is_file():
            print("Loading old exp table")
            self.exp_table = pickle.load(open(EXP_TABLE_FILE, "rb"))
        return

    def create_net(self):
        if Path(MODEL_FILE).is_file():
            print("Loading pretrained model")
            return load_model(MODEL_FILE)
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha, decay=0.01))
        return model

    def remember(self, state, action, reward, next_state):
        self.exp_table.append((state, action, reward, next_state))
        if len(self.exp_table) > self.exp_table_sz:
            self.exp_table.popleft()

    def act(self, state):
        if random.random() <= self.eps:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        self.old_state = state
        self.action = np.argmax(act_values[0])
        # print(act_values)
        return self.action

    def replay(self, batch_size):
        minibatch = random.sample(self.exp_table, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * \
                   np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def get_reward(self, state):
      temp_list = [sum(state[self.rew_attr][0:2]),
                   sum(state[self.rew_attr][2:4])]
      return -sum([elem ** self.Lnorm for elem in temp_list])

    def run(self, env_state):
        self.time_slice += 1
        new_state = np.array([env_state["cur_phase"],
                             sum(env_state["q_len"][0:2]),
                             sum(env_state["q_len"][2:4])])
        new_state = np.reshape(new_state, [1, self.state_size])

        if self.learning and self.time_slice > 10:
            reward = self.get_reward(env_state)
            self.remember(self.old_state, self.action, reward, new_state)
        return self.act(new_state)

    def save_state(self):
        if not self.learning:
            return
        # replay and learn whatever moves have been taken
        print("Performing Experience Replay")
        for i in range(100):
            self.replay(32)
        pickle.dump(self.exp_table, open(EXP_TABLE_FILE, "wb"))
        self.model.save(MODEL_FILE)