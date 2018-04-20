# Code for testing
import os
import random
import time

import matplotlib.pyplot as plt

import exp_replay
import q_learn_agent
import simple_agent
import traffic_env

from shutil import copyfile

NUM_STEPS = 2000
NUM_TESTS = 1
NUM_ITERS = 53

def generate_test_set(num_tests=NUM_TESTS, num_steps=NUM_STEPS):
    for i in range(num_tests):
        file_name = "data/cross.rou%s.xml"
        traffic_env.generate_routefile(num_steps, i, file_name %i)

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

def test_hyper_param(hyper_params, num_steps=NUM_STEPS, period=4):
    # Remove old pickle file
    try:
        os.remove("./q_table.p")
        os.remove("./exp_table.p")
    except OSError:
        pass
    # input()
    traffic_env.generate_routefile(num_steps, 42)

    avg_stats = dict([(key, []) for key in traffic_env.TRAFFIC_ATTRS])
    for i in range(NUM_ITERS):
        t1 = time.time()
        print("Learning_step: ", i)
        learning_rate = 10/(50 + i)
        exp_prob = 10/(15+i)
        relearns = (3 + i//20)
        print("EPS PROB: ", exp_prob)
        print("LEarning rate: ", learning_rate)
        agent = exp_replay.QLearn_ExpReplay_Agent(learning=True,
            learning_rate=learning_rate, num_exp_learns=relearns,
            exploration_eps=exp_prob, **hyper_params) 
        # traffic_env.generate_route_file(num_steps)
        env = traffic_env.Environment(agent)
        env.run()
        
        if i%period == 0:
            agent = q_learn_agent.QLearn_Agent(learning=False,
                **hyper_params)
            env = traffic_env.Environment(agent)
            env.run()
            for key in traffic_env.TRAFFIC_ATTRS:
                avg_stat = sum(env.stats[key]) / env.step
                # Store the best seen q table
                if len(avg_stats[key]) > 1 and min(avg_stats[key]) > avg_stat:
                    copyfile("q_table.p", "best_q_table.p")
                avg_stats[key].append(avg_stat)
                print("Cur avg stat: ", key, avg_stat)
        print("Time for loop %s : %f" %(i, time.time() - t1))
        # print(env.stats)
    plot_avg_stats(avg_stats, "Iter num")

def simple_test(hyper_params={"switch_time" : 25}):
    avg_stats = dict([(key, []) for key in traffic_env.TRAFFIC_ATTRS])
    for i in range(5,30,5):
        hyper_params["switch_time"] = i
        agent = simple_agent.SimpleAgent(**hyper_params)
        env = traffic_env.Environment(agent)
        stats = run_tests(env)
        for key in stats:
            avg_stats[key].append(stats[key])

    print(avg_stats)
    plot_avg_stats(avg_stats, "switch_time")

def plot_avg_stats(avg_stats, xlabel):
    for label in avg_stats.keys():
        plt.figure()
        plt.plot(avg_stats[label], "ro")
        plt.xlabel(xlabel)
        plt.ylabel(label)
        plt.savefig(label + str(hyper_params.values()) + str(NUM_ITERS) + ".png")


if __name__ == "__main__":
    hyper_params = {
                "rew_attr" : "wait_time",
                "Lnorm" : 3,
               }
    generate_test_set()

    # test_hyper_param(hyper_params)
    simple_test()
