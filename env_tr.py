
# Environment class

# from __future__ import print_function

import optparse
import os
import random
import subprocess
import sys
import time

from dqn_agent import DQN_Agent
from q_learn_agent import QLearn_Agent
from range_q_learn_agent import Range_QLearn_Agent

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

# Attributes in consideration
TRAFFIC_ATTRS = ("q_len", "wait_time")

# Generates vehicles using a uniform distribution
def generate_routefile(num_steps, seed=None, file_name="data/cross.rou.xml"):
    random.seed(seed)
    N = num_steps  # number of time steps
    # demand per second from different directions
    pWE = 1. / 3
    pEW = 1. / 4
    pNS = 1. / 20
    pSN = 1. / 15
    with open(file_name, "w") as routes:
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

#Interface between simulator and agent
class Environment:
    def __init__(self, agent):
        self.agent = agent
        # get cli options
        options = self.get_options()
        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        if options.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

    # Options for simulator
    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    # Execution of simulator
    def execute_loop(self):

        """execute the TraCI control loop"""
        self.step = 0
        # we start with phase 2 where EW is green
        cur_phase = 2
        cur_phase_len = 0

        traci.trafficlight.setPhase("0", cur_phase)
        
        while traci.simulation.getMinExpectedNumber() > 0:
            
            # executes one step in simulation
            traci.simulationStep()
            self.step += 1
            sim_phase = traci.trafficlight.getPhase("0")

            # Set the state of the environment at this step
            actual_state = dict([(key, []) for key in TRAFFIC_ATTRS])
            for edgeId in ["3i", "4i", "1i", "2i"]:
                x = traci.edge.getLastStepHaltingNumber(edgeId)
                y = traci.edge.getWaitingTime(edgeId)
                actual_state["q_len"].append(x)
                actual_state["wait_time"].append(y)

            # Store the sum of attributes to calculate the avg value 
            for key in TRAFFIC_ATTRS:
                self.stats[key].append(sum(actual_state[key]))
            # If amber light don't run the agent
            if sim_phase in [1, 3]:
                continue

            cur_phase_len += 1

            if(cur_phase != traci.trafficlight.getPhase("0")):
                cur_phase = traci.trafficlight.getPhase("0")
                cur_phase_len = 0

            actual_state["cur_phase"] = cur_phase
            actual_state["cur_phase_len"] = cur_phase_len
            
            # get action from agent
            action = self.agent.run(actual_state)

            if(action == 1):
                # set yellow light
                cur_phase = (cur_phase + 1)%4
                traci.trafficlight.setPhase("0",cur_phase)
                cur_phase_len = 0

        self.agent.save_state()
        traci.close()
        sys.stdout.flush()


    def run(self):
        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([self.sumoBinary, "-c", "data/cross.sumocfg",
                                 "--tripinfo-output", "tripinfo.xml"])
        self.stats = dict([(key, []) for key in TRAFFIC_ATTRS])
        self.execute_loop()

# helper function to learn
def learn():
    for i in range(100):
        print("Inside learning step: ", i)
        generate_routefile(2000)
        learning_rate = 10/(50 + i)
        eps_prob = 10/(10 + i)
        print("Loop: ", i)
        agent =DQN_Agent(learning=True, rew_attr="wait_time", Lnorm=3,
            # learning_rate=learning_rate,
            # exploration_eps=eps_prob
            )
        env = Environment(agent)
        env.run()

#helper function to evaluate
def eval():
    agent = DQN_Agent(learning=False, rew_attr="wait_time")
    env = Environment(agent)
    generate_routefile(2000)
    env.run()
    for key in TRAFFIC_ATTRS:
        print("STATS: ", sum(env.stats[key])/len(env.stats[key]))

if __name__ == "__main__":
    # for i in [1]:
    #     test_hyper_param(i)
    learn()
    # eval()
