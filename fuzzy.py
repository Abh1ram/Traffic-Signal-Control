# Fuzzy logic controller
# We have parameters, and we need to update them, based on the fuzzy set rules
# FSet can represent a fuzzySet class
import types
import copy
from functools import reduce

class FuzzySet(object):
    def score(self, obj):
        return 0 # Empty set otherwise
    
    def max(self):
        return None
    
    def lt(self):
        return None
    
    def mt(self):
        return None
        

class DictSet(FuzzySet):
    def __init__(self, dic):
        self.dict = dic
    
    def score(self, elem):
        if elem not in self.dict:
            raise AttributeError("Invalid member: " + str(elem))
        return self.dict[elem]
    
    def max(self):
        mx_key, mx_val = None, 0
        for key, val in self.dict.items():
            if val > mx_val:
                mx_val, mx_val = key, val
        max_list = sorted([key for key in self.dict\
                if self.dict[key] == self.dict[mx_key]])
        mx_key = max_list[int(len(max_list) / 2)]
        return mx_key
    
    def mt(self):
        new_dict = {}
        mx_key = self.max()
        for key, val in self.dict:
            new_dict[key] = (1 - self.dict[key]) if key > mx_key else 0
        return DictSet(new_dict)

    def lt(self):
        new_dict = {}
        mx_key, _ = self.max()
        for key, val in self.dict:
            new_dict[key] = (1 - self.dict[key]) if key < mx_key else 0
        return DictSet(new_dict)
        
    def any(self):
        return DictSet({ key: 1 for key in self.dict })
    
    def none(self):
        return DictSet({ key: 0 for key in self.dict })
        

class ListSet(FuzzySet):
    def __init__(self, lis, s=0, e=-1):
        if e == -1:
            e = s + len(lis)
        self.list = lis
        self.s, self.e = s, e
    
    def score(self, elem):
        if elem >= self.e:
            raise AttributeError("Invalid member: " + str(elem))
        return self[elem]
    
    def max(self):
        max_i = self.s
        # Get index at which it's max
        for i in range(self.s, self.e):
            if self[i] > self[max_i]: max_i = i
        max_list = [i for i in range(self.s, self.e)\
                if self[i] == self[max_i]]
        max_i = max_list[int(len(max_list) / 2)]
        return max_i
    
    def mt(self):
        mx_key = self.max()
        new_list = [0] * (mx_key - self.s + 1) \
                + [1 - self[i] for i in range(mx_key + 1, self.e)]
        return ListSet(new_list)

    def lt(self):
        mx_key = self.max()
        new_list = [1 - self.list[i - self.s]\
                for i in range(self.s, mx_key)]\
                + [0] * (self.e - mx_key) 
        return ListSet(new_list)
    
    def fany(self):
        return ListSet([1] * range(len(self.list)), self.s, self.e)
    
    def none(self):
        return ListSet([0] * range(len(self.list)), self.s, self.e)
        
    def __getitem__(self, item):
        return self.list[item - self.s]
        

class FuncSet(FuzzySet): 
    def __init__(self, func):
        self.func = func
        
    def score(self, elem):
        return self.func(elem)
        
        
def make_fuzzy_set(obj):
    f_set = None
    if type(obj) is dict:
        f_set = DictSet(obj)
    elif callable(obj):
        f_set = FuncSet(obj)
    elif type(obj) is list:
        f_set = ListSet(obj)
    elif type(obj) is tuple:
        f_set = ListSet(*obj)
    else:
        raise ValueError("Invalid type of argument: " + type(obj))
    return f_set
    
    
class FuzzyOperators(object):
    @staticmethod
    def f_and(fsets_list):
        """ Or multiple sets """
        return min([fset.score(arg) for fset, arg in fsets_list])
    
    @staticmethod
    def f_or(fsets_list):
        """ And multiple sets """
        return max([fset.score(arg) for fset, arg in fsets_list])