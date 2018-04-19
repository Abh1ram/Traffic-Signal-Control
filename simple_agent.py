# BaseLine Agent class

class SimpleAgent:
    def __init__(self, switch_time):
        self.switch_time = switch_time

    def run(self, env_state):
        if env_state["cur_phase"] not in [1, 3]:
            if env_state["cur_phase_len"] >= self.switch_time:
                return 1
        return 0
        