
# BaseLine Agent class
class SimpleAgent:
    # switch_time: duration for a phase
    def __init__(self, switch_time):
        self.switch_time = switch_time

    # determines action for agent.
    def run(self, env_state):
        if env_state["cur_phase"] not in [1, 3]:
            if env_state["cur_phase_len"] >= self.switch_time:
                return 1
        return 0

    # does nothing since the agent does not need to save any information.
    def save_state(self):
        return 0
        