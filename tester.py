# Code for testing
import os
import random
import time

import matplotlib.pyplot as plt

import env_tr
import exp_replay
import q_learn_agent
import range_q_learn_agent
import simple_agent
import traffic_env

from shutil import copyfile

from dqn_agent import DQN_Agent

NUM_STEPS = 2000
NUM_TESTS = 1
NUM_ITERS = 103

def generate_test_set(num_tests=NUM_TESTS, num_steps=NUM_STEPS):
    for i in range(num_tests):
        file_name = "data/cross.rou%s.xml"
        env_tr.generate_routefile(num_steps, i, file_name %i)

def run_tests(env, num_tests=NUM_TESTS):
    avg_stats = dict([(key, []) for key in env_tr.TRAFFIC_ATTRS])
    for i in range(num_tests):
        file_name = "data/cross.rou%s.xml" %i
        dest = "data/cross.rou.xml"
        copyfile(file_name, dest)
        env.run()
        for key in env_tr.TRAFFIC_ATTRS:
            avg_stats[key].append(sum(env.stats[key]) / len(env.stats[key]))
    # give average over the num of tests
    for key in env_tr.TRAFFIC_ATTRS:
        avg_stats[key] = sum(avg_stats[key]) / num_tests
    return avg_stats

def test_hyper_param(hyper_params, num_steps=NUM_STEPS, period=10):
    # Remove old pickle file
    try:
        os.remove("./range_q_table.p")
        # os.remove("./dqn_exp_table.p")
    except OSError:
        pass
    # input()

    avg_stats = dict([(key, []) for key in env_tr.TRAFFIC_ATTRS])
    for i in range(NUM_ITERS):

        t1 = time.time()
        env_tr.generate_routefile(num_steps, None)
        
        print("Learning_step: ", i)
        learning_rate = 10/(80 + i)
        exp_prob = 10/(10+i//5)
        print("EPS PROB: ", exp_prob)
        print("LEarning rate: ", learning_rate)
        
        agent = range_q_learn_agent.Range_QLearn_Agent(learning=True,
            learning_rate=learning_rate, exploration_eps=exp_prob,
             **hyper_params)
        # env_tr.generate_route_file(num_steps)
        env = env_tr.Environment(agent)
        env.run()
        
        if (i+1)%period == 0:
            agent = range_q_learn_agent.Range_QLearn_Agent(learning=False,
                **hyper_params)
            env = env_tr.Environment(agent)
            
            stats = run_tests(env)
            for key in env_tr.TRAFFIC_ATTRS:
                # Store the best seen q table
                if len(avg_stats[key]) > 1 and min(avg_stats[key]) > stats[key]:
                    print("Copying best configuration")
                    copyfile("range_q_table.p", "best_q_table.p")
                avg_stats[key].append(stats[key])
            
            print("Cur avg stat: ", stats)
        print("Time for loop %s : %f" %(i, time.time() - t1))
        # print(env.stats)
    plot_avg_stats(avg_stats, "Iter num")

def simple_test(start=5, end=31, jump=5):
    rng = range(start, end, jump)
    avg_stats = dict([(key, []) for key in env_tr.TRAFFIC_ATTRS])
    for i in rng:
        print("SWITCH TIME: ", i)
        agent = simple_agent.SimpleAgent(switch_time=i)
        env = env_tr.Environment(agent)
        stats = run_tests(env)
        for key in stats:
            avg_stats[key].append(stats[key])

    print(avg_stats)
    plot_avg_stats(avg_stats, "switch time", xvals=list(rng))

def plot_avg_stats(avg_stats, xlabel, xvals=None):
    for label in avg_stats.keys():
        plt.figure()
        if not xvals:
            plt.plot(avg_stats[label])
        else:
            plt.plot(xvals, avg_stats[label], "ro")
        plt.xlabel(xlabel)
        plt.ylabel(label)
        plt.savefig(label + str(hyper_params.values()) + str(NUM_ITERS) + ".png")


if __name__ == "__main__":
    hyper_params = {
                "rew_attr" : "wait_time",
                "Lnorm" : 3,
               }
    generate_test_set()
    test_hyper_param(hyper_params, period=5)
    # simple_test()
