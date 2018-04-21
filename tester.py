# Code for testing
import os
import random
import time

import matplotlib.pyplot as plt

import q_learn_agent 
import range_q_learn_agent
import simple_agent
import traffic_env
from fuzzyagent import FuzzyAgent

from shutil import copyfile

NUM_STEPS = 3000
NUM_TESTS = 1
NUM_ITERS = 10

def generate_test_set(num_tests=NUM_TESTS, num_steps=NUM_STEPS):
    for i in range(num_tests):
        file_name = "data/cross.rou%s.xml"
        traffic_env.generate_routefile(num_steps, 42, file_name %i)

def run_tests(env, num_tests=NUM_TESTS):
    avg_stats = dict([(key, []) for key in traffic_env.TRAFFIC_ATTRS])
    for i in range(num_tests):
        file_name = "data/cross.rou%s.xml" %i
        dest = "data/cross.rou.xml"
        copyfile(file_name, dest)
        env.run()
        for key in traffic_env.TRAFFIC_ATTRS:
            avg_stats[key].append(sum(env.stats[key]) / env.step)
    # give average over the num of tests
    for key in traffic_env.TRAFFIC_ATTRS:
        avg_stats[key] = sum(avg_stats[key]) / num_tests
    return avg_stats

def test_hyper_param(hyper_params, num_steps=NUM_STEPS, period=10):
    # Remove old pickle file
    try:
        os.remove("./q_table.p")
    except OSError:
        pass
    # input()
    traffic_env.generate_routefile(num_steps, 42)

    avg_stats = dict([(key, []) for key in traffic_env.TRAFFIC_ATTRS])
    for i in range(NUM_ITERS):
        t1 = time.time()
        print("Learning_step: ", i)
        learning_rate = 20/(100+i)
        exp_prob = 10/(15+i)
        print("LEarning rate: ", learning_rate)
        # agent = q_learn_agent.QLearn_Agent(learning=True,
        #     learning_rate=learning_rate,
        #     **hyper_params) 
        agent = range_q_learn_agent.Range_QLearn_Agent(learning=True,
            learning_rate=learning_rate,
            **hyper_params) 
        # traffic_env.generate_route_file(num_steps)
        env = traffic_env.Environment(agent)
        env.run()
        
        if i%period == 0:
            # agent = q_learn_agent.QLearn_Agent(learning=False,
            #     **hyper_params)
            agent = range_q_learn_agent.Range_QLearn_Agent(learning=False,
                **hyper_params)
            
            env = traffic_env.Environment(agent)
            env.run()
            for key in traffic_env.TRAFFIC_ATTRS:
                avg_stats[key].append(sum(env.stats[key]) / env.step)
            print("Difference: ", env.step, num_steps)
        print("Time for loop %s : %f" %(i, time.time() - t1))
        # print(env.stats)
    for label in avg_stats.keys():
        plt.figure()
        plt.plot(avg_stats[label])
        plt.xlabel("Iter num")
        plt.ylabel(label)
        plt.savefig(label + str(hyper_params.values()) + str(NUM_ITERS) + ".png")

def simple_test(hyper_params={"switch_time" : 25}):
    fuzzy_agent = FuzzyAgent()
    agent = simple_agent.SimpleAgent(**hyper_params)
    env = traffic_env.Environment(fuzzy_agent)
    print(run_tests(env))


if __name__ == "__main__":
    hyper_params = {
                "rew_attr" : "q_len",
                "Lnorm" : 3,
               }
    # test_hyper_param(hyper_params, period=5)
    generate_test_set()
    simple_test()
