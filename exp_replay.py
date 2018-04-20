import pickle
import random

from collections import deque
from pathlib import Path

q_pickle_file = "./q_table.p"

exp_pickle_file = "./exp_table.p"


class QLearn_ExpReplay_Agent():
  def __init__(self, rew_attr="q_len", Lnorm=1, discount_rate=0.9,
               learning=True, learning_rate=0.2, exploration_eps=0.2,
               num_exp_learns=3, exp_table_sz=2000):
    # set the reward parameters
    self.rew_attr = rew_attr
    self.Lnorm = Lnorm
    # set the learning parameters
    self.learning = learning
    self.alpha = learning_rate
    self.gamma = discount_rate
    self.eps = exploration_eps

    self.create_q_table()
    self.create_exp_table()
    self.num_exp_learns = num_exp_learns
    self.exp_table_sz = exp_table_sz
    # updating the q table - SARS'
    self.time_slice = 0
    self.old_state = None
    self.action = None
    self.new_state = None
    self.reward = None

  def create_q_table(self):
    # Dictionary of states - for each state,
    # which is a dictionary - action(key) : q-value(value)
    self.q_table = {}
    if Path(q_pickle_file).is_file():
      self.q_table = pickle.load(open(q_pickle_file, "rb"))
    self.initial_val = 0

  def create_exp_table(self):
    self.exp_table = deque()
    if Path(exp_pickle_file).is_file():
      self.exp_table = pickle.load(open(exp_pickle_file, "rb"))

  def get_default_dict(self, phase_num):
    if phase_num in [1, 3]:
      # If amber only possible action - wait/stay
      return {0 : self.initial_val}
    return {0 : self.initial_val,
            1 : self.initial_val}

  def get_action(self, env_state_tup):
    # currently only one intersection
    # state - phase_number, elapsed_phase_time, queue_len_lr, queue_len_tb 
    self.old_state = env_state_tup
    default_dict = self.get_default_dict(env_state_tup[0])

    action_dict = self.q_table.get(env_state_tup, default_dict)
    # print("ENV STATE: ",env_state_tup, "Action dict: ", action_dict)
    max_val = max(action_dict.values())
    best_actions = set(filter(lambda key: action_dict[key] == max_val,
      action_dict.keys()))
    # Based on whether agent is learning or not and moves available,
    # explore or exploit
    # explore with probability eps
    if self.learning and random.random() < self.eps:
      remaining_actions = set(action_dict.keys()) - best_actions
      if len(remaining_actions) > 0:
        self.action = random.choice(list(remaining_actions))
        return self.action
    # exploit the best action
    self.action = random.choice(list(best_actions))
    return self.action

  def update_q_table(self, exp_tuple):
    old_state = exp_tuple[0]
    default_dict = self.get_default_dict(old_state[0])
    old_action_dict = self.q_table.get(old_state, default_dict)
    
    new_state = exp_tuple[3]
    new_action_dict = self.q_table.get(new_state, default_dict)
    max_val = max(new_action_dict.values())
    # Updation formula
    action = exp_tuple[1]
    reward = exp_tuple[2]
    old_action_dict[action] += self.alpha * (
      reward + self.gamma * max_val - old_action_dict[action])
    # set the new dict
    self.q_table[old_state] = old_action_dict

  def run(self, env_state):
    self.time_slice += 1
    # print("Agent:", self.time_slice)
    # agent's state is cur_phase and queue_lens along two directions
    env_state_tup = (env_state["cur_phase"],
                     sum(env_state["q_len"][0:2]),
                     sum(env_state["q_len"][2:4]))
    # print("Env state; ", env_state)
    # get reward and update Q table
    if self.time_slice > 10 and self.learning:
      self.reward = self.get_reward(env_state)
      self.new_state = env_state_tup
      self.update_exp_table()
    action = self.get_action(env_state_tup)
    # print("Action: ", action)
    return action

  def update_exp_table(self):
    self.exp_table.append((self.old_state, self.action,
                           self.reward, self.new_state))
    if len(self.exp_table) > self.exp_table_sz:
      self.exp_table.popleft()
    # relearn old experiences
    for i in range(self.num_exp_learns):
      exp_tuple = random.choice(self.exp_table)
      self.update_q_table(exp_tuple)

  def get_reward(self, env_state):
    temp_list = [sum(env_state[self.rew_attr][0:2]),
                 sum(env_state[self.rew_attr][2:4])]
    return -sum([elem ** self.Lnorm for elem in temp_list])

  def save_state(self):
    pickle.dump(self.q_table, open(q_pickle_file, "wb"))
    pickle.dump(self.exp_table, open(exp_pickle_file, "wb"))
    
    