# DEBUGGER
import pickle

from q_learn_agent import pickle_file

q_table = pickle.load(open(pickle_file, "rb"))
print(len(q_table))
i = 0
for key in q_table.keys():
    print("key:", key, "val: ", q_table[key])
    i +=1
    if i > 20:
        break
