import os
import random
import time

import matplotlib.pyplot as plt
from fuzzyagent import FuzzyAgent
import q_learn_agent 
import range_q_learn_agent
import simple_agent
import env_tr
import argparse
import pickle
from tester import generate_test_set, run_tests

def or_prob(x, y):
    """ Probability of x or y happening """
    return (1 - (1 - x) * (1 - y))


def gen_probs():
    """ generate prob of traffic in all directions """
    divs = [2.25] * 4
    pNS, pSN, pWE, pEW = [random.random() / divs[i] + 0.2 for i in range(4)]
    mar1, mar2 = or_prob(pNS, pSN), or_prob(pWE, pEW)
    return (pNS, pSN, pWE, pEW), (mar1, mar2)


def plot_stats(stats):
    prob_vals = [sum(stat['probs']) for stat in stats]
    q_vals = [stat['stats']['q_len'] for stat in stats]
    wait_vals = [stat['stats']['wait_time'] for stat in stats]
    figure, ax = plt.figure(), plt.gca()
    ax.set_xlabel("Sum of probs");
    # Plot the two graphs
    plt.plot(prob_vals, q_vals, 'g')
    plt.plot(prob_vals, wait_vals, 'b')
    ax.legend(["Average queue length", "Average waiting time"])
    plt.show()


def pickle_dump(stats, filename=None):
    ti = int(time.time())
    if not filename: filename = "pkl__" + str(ti) + ".p"
    pickle.dump(stats, open(filename, "wb"))
    return filename


NUM_TESTS = 1
NUM_STEPS = 1000
def test_fuzzy_agent(num_tests=NUM_TESTS, hyper_params={'verbose': False}):
    """ Test the fuzzy agent """
    stats = []
    filename = None
    for test in range(num_tests):
        print("***** Test: " + str(test))
        probs, mars = gen_probs()
        print("For probs:", probs)
        env = env_tr.Environment(FuzzyAgent(mars=mars, **hyper_params))
        generate_test_set(num_tests=1, num_steps=NUM_STEPS, probs=probs)
        stats.append({'probs': probs, 'stats':run_tests(env, num_tests=1)})
        if test % 50 == 0:
            # Save states
            filename = pickle_dump(stats, filename)
    if num_tests == 1:
        print(stats)
    else:
        pickle_dump(stats, filename)
        plot_stats(stats)


if __name__ == "__main__":
    test_fuzzy_agent()