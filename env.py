from __future__ import absolute_import
from __future__ import print_function

import optparse
import os
import random
import subprocess
import sys
import time

import matplotlib.pyplot as plt

from q_learn_agent import QLearn_Agent


# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

attributes = ("q_len", "wait_time")

class Environment:
    def __init__(self, learning=True, state_attr="q_len", Lnorm=1,
                 n_steps=1000):
        self.learning = learning
        self.state_attr = state_attr
        # 1 - L1, 2 - L2
        self.Lnorm = Lnorm
        self.n_steps = n_steps
        options = self.get_options()

        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        if options.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def generate_routefile(self):
        random.seed(42)  # make tests reproducible
        N = self.n_steps  # number of time steps
        # demand per second from different directions
        pWE = 1. / 10
        pEW = 1. / 11
        pNS = 1. / 30
        pSN = 1. / 15
        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="veh" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

            <route id="right" edges="51o 1i 2o 52i" />
            <route id="left" edges="52o 2i 1o 51i" />
            <route id="down" edges="54o 4i 3o 53i" />
            <route id="up" edges="53o 3i 4o 54i" />""", file=routes)
            lastVeh = 0
            vehNr = 0
            for i in range(N):
                if random.uniform(0,1) < pWE:
                    print('    <vehicle id="right_%i" type="veh" route="right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0,1) < pEW:
                    print('    <vehicle id="left_%i" type="veh" route="left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0,1) < pNS:
                    print('    <vehicle id="down_%i" type="veh" route="down" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0,1) < pSN:
                    print('    <vehicle id="up_%i" type="veh" route="up" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i

            print("</routes>", file=routes)

    def execute_loop(self):

        agent = QLearn_Agent(learning=self.learning)
        print("Learning: ", self.learning)
        """execute the TraCI control loop"""
        self.step = 0
        # we start with phase 2 where EW has green

        cur_phase = 2
        cur_phase_len = 0

        traci.trafficlight.setPhase("0", cur_phase)
        
        # list of (queue length, delay) for last step
        # [EW,NS]
        while traci.simulation.getMinExpectedNumber() > 0:
            # print("ENV STEP: ", step)
            traci.simulationStep()
            cur_phase_len += 1

            if(cur_phase != traci.trafficlight.getPhase("0")):
                cur_phase = traci.trafficlight.getPhase("0")
                cur_phase_len = 0

            actual_state = dict([(key, []) for key in attributes])
            for edgeId in ["1i", "2i", "3i", "4i"]:
                x = traci.edge.getLastStepHaltingNumber(edgeId)
                y = traci.edge.getWaitingTime(edgeId)
                actual_state["q_len"].append(x)
                actual_state["wait_time"].append(y)

            # Store the sum of attributes to calculate the avg value 
            for key in attributes:
                self.stats[key].append(sum(actual_state[key]))

            agent_state = [
                            sum(actual_state[self.state_attr][0:2]),
                            sum(actual_state[self.state_attr][2:4]),
                            cur_phase,
                            # cur_phase_len,
                        ]

            # print("Agent state: ", agent_state)
            # print("Actual state: ", actual_state)
            reward = self.get_reward(agent_state)
            # print("Reward: ", reward)
            # Get action from the agent
            action = agent.run(agent_state, reward=reward)

            if(action == 1):
                cur_phase = (cur_phase + 1)%4
                traci.trafficlight.setPhase("0",cur_phase)
                cur_phase_len = 0

            self.step += 1
        # input()
        agent.save_q_table()
        traci.close()
        sys.stdout.flush()

    def get_reward(self, agent_state):
        return -sum([elem ** self.Lnorm for elem in agent_state[0:2]])

    def run(self):

        # first, generate the route file for this simulation
        self.generate_routefile()

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([self.sumoBinary, "-c", "data/cross.sumocfg",
                                 "--tripinfo-output", "tripinfo.xml"])
        self.stats = dict([(key, []) for key in attributes])
        self.execute_loop()

def learn():
    for i in range(50):
        print("Loop: ", i)
        env = Environment(learning=True)
        env.run()

def eval():
    env = Environment(learning=False)
    env.run()

def test_hyper_param(Lnorm = 1):
    hyper_params = {
                    "state_attr" : "wait_time",
                    "Lnorm" : Lnorm,
                    "n_steps" : 2000
                   }
    # Remove old pickle file
    try:
        os.remove("./q_table.p")
    except OSError:
        pass
    # input()
    avg_stats = dict([(key, []) for key in attributes])
    for i in range(NUM_ITERS):
        t1 = time.time()
        print("Learning_step: ", i)
        env = Environment(learning=True, **hyper_params)
        env.run()
        if i%4 == 0:
            env = Environment(learning=False, **hyper_params)
            env.run()
            for key in attributes:
                avg_stats[key].append(sum(env.stats[key]) / env.n_steps)
        print("Time for loop %s : %f" %(i, time.time() - t1))
        # print(env.stats)
    for label in avg_stats.keys():
        plt.figure()
        plt.plot(avg_stats[label])
        plt.xlabel("Iter num")
        plt.ylabel(label)
        plt.savefig(label + str(hyper_params.values()) + ".png")

NUM_ITERS = 53
if __name__ == "__main__":
    # for i in [1]:
    #     test_hyper_param(i)
    # learn()
    eval()
    # make is sum of square of each queue- wll caause it oen the longer queue. but similar to greedy.
