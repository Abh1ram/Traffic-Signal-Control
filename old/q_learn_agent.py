import pickle
import random

from pathlib import Path
# from agent import Agent

pickle_file = "./q_table.p"
pickle_Path = Path(pickle_file)

class QLearn_Agent():
  def __init__(self, num_steps=100, learning=True, learning_rate=0.2,
               discount_rate=0.9, exploration_eps=0.2):
    # super(QLearn_Agent, self).__init__(traffic_env, num_steps)
    self.create_q_table()
    self.learning = learning
    self.alpha = learning_rate
    self.gamma = discount_rate
    self.eps = exploration_eps

    # updating the q table
    self.time_slice = 0
    self.old_state = None
    self.action = None
    self.new_state = None
    self.reward = None

  def create_q_table(self):
    self.q_table = {}
    if pickle_Path.is_file():
      print("Loaded.............................")
      self.q_table = pickle.load(open(pickle_file, "rb"))
    self.initial_val = 0
    # Dictionary of states - for each state, which is a dictionary - action(key) : q-value(value)

  def get_default_dict(self, phase_num):
    if phase_num in [1, 3]:
      # If amber only possible action - wait/stay
      return {0 : self.initial_val}
    return {0 : self.initial_val,
            1 : self.initial_val}

  def get_action(self, env_state):
    # currently only one intersection
    # state - phase_number, elapsed_phase_time, queue_len_lr, queue_len_tb 
    self.old_state = env_state
    default_dict = self.get_default_dict(env_state[0])

    action_dict = self.q_table.get(env_state, default_dict)
    max_val = max(action_dict.values())
    best_actions = set(filter(lambda key: action_dict[key] == max_val,
      action_dict.keys()))
    # print(self.learning, "Action: ", action_dict)
    if self.learning and random.random() < self.eps:
      remaining_actions = set(action_dict.keys()) - best_actions
      # explore with probability eps
      if len(remaining_actions) > 0:
        self.action = random.choice(list(remaining_actions))
        return self.action
    # exploit the best action
    self.action = random.choice(list(best_actions))
    return self.action

  def update_q_table(self):
    default_dict = self.get_default_dict(self.old_state[0])
    old_action_dict = self.q_table.get(self.old_state, default_dict)
    
    new_action_dict = self.q_table.get(self.new_state, default_dict)
    max_val = max(new_action_dict.values())
    # Updation formula
    # print("Old: ", old_action_dict)
    old_action_dict[self.action] += self.alpha * (
      self.reward + self.gamma * max_val - old_action_dict[self.action])
    # set the new dict
    self.q_table[self.old_state] = old_action_dict
    # print("New: ", old_action_dict)


  def run(self, env_state, reward=None):
    self.time_slice += 1
    # print("Agent:", self.time_slice)
    env_state = tuple(env_state)
    # print("Env state; ", env_state)
    # print("reward: ", reward)
    # If reward is obtained, update Q table
    if self.time_slice > 1 and reward != None and self.learning:
      # print("Reward: ", reward)
      self.reward = reward
      self.new_state = env_state
      self.update_q_table()
    action = self.get_action(env_state)
    # print("Action: ", action)
    return action

  def save_q_table(self):
    pickle.dump(self.q_table, open(pickle_file, "wb"))