# DEBUGGER
import pickle

from q_learn_agent import pickle_file

q_table = pickle.load(open(pickle_file, "rb"))
print(len(q_table))
