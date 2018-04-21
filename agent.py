from sets import *
from fuzzy import FuzzyOperators
from rules import RULES
import random
from functools import reduce


NOTOGGLE, TOGGLE = 0, 1
AMBER = [1, 3]


def run_rule(rule, t, a, q):
    """ Execute the rule, and return the extension calculated """
    eval_list = []
    for clause in rule:
        ti, arr, qu = clause
        val = FuzzyOperators.f_and([(ti, t), (arr, a), (qu, q)])
        eval_list.append(val)
    # clarity, ext = reduce(lambda a, b: a if a[0] > b[0] else b, eval_list)
    return max(eval_list)
    
    
# TODO Set this properly
MAR = 0.5
def gen_state(t, q):
    """ Generate new state (vehicle arrival state)
        for the next t seconds
    """
    a, ax = [(1 if random.random() <= MAR else 0) for _ in range(t)],\
            [(1 if random.random() <= MAR else 0) for _ in range(t)]
    b, bx = [], []
    s, sx = 0, q
    for i in range(t):
        s, sx = s + a[i], sx + ax[i]
        b.append(s); bx.append(sx)
    return (b, bx)
    
    
def calc_extension(rule, max_ext, q):
    b, bx = gen_state(max_ext, q)
    ext = []
    for t in range(1, max_ext + 1):
        ext.append((t, run_rule(rule, t, b[t - 1], bx[t - q])))
    ext_tup = reduce(lambda a, b: a if a[1] > b[1] else b, ext)
    return ext_tup[0] if ext_tup[1] > 0.5 else 0


class FuzzyAgent(object):
    MAX_EXT = 10
    def __init__(self, start=7, amber_dur=3.5):
        self.t = 0
        self.next_action, self.start = start, start
        self.ri = 0
        self.to_toggle = False
    
    def run(self, env_state):
        ret = NOTOGGLE
        if env_state[0] not in AMBER and self.t == self.next_action:
            ret = self._toggle() if self.to_toggle\
                    else self._take_action(env_state)
        self.t +=1
        return ret
            
    def _take_action(self, env_state):
        ext = self._calc_extension(env_state)
        self.next_action += ext
        self.ri += 1
        self.to_toggle = bool(ext < 8)
        if ext == 0 or self.ri == len(RULES):
            return self._toggle()
        return NOTOGGLE
    
    def _calc_extension(self, env_state):
        other_q_len = sum(env_state["q_len"][1:3])\
                if env_state['cur_phase'] >= 2 else sum(env_state["q_len"][2:4])
        return calc_extension(RULES[self.ri], self.MAX_EXT, other_q_len)
            
    def _toggle(self):
        self.t, self.ri, self.to_toggle = 0, 0, False
        self.next_action = self.start
        return TOGGLE
            