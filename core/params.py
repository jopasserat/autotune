# Code excerpt taken from Hyperband experiments, L. Li 2016
# parameter represention module



import numpy
import random
import copy


class Param(object):
    '''
    define different properties and helper functions
    '''

    def __init__(self, name, min_val, max_val, init_val=None,
                 distrib='uniform', scale='log', logbase=numpy.e, interval=None):
        self.name = name
        self.init_val = init_val
        self.min_val = min_val
        self.max_val = max_val
        self.distrib = distrib  # uniform or normal
        self.scale = scale  # log or linear
        self.logbase = logbase
        self.param_type = 'continuous'
        self.interval = interval

    def __repr__(self):
        return "%s (%f,%f,%s)" % (self.name, self.min_val, self.max_val,
                                  self.scale)

    def get_param_range(self, num_vals, stochastic=False):
        if stochastic:
            if self.distrib == 'normal':
                # bad design but here min_val is mean and max_val is sigma
                val = numpy.random.normal(self.min_val, self.max_val, num_vals)
            else:
                val = numpy.random.rand(num_vals)*(self.max_val - self.min_val) + self.min_val
            if self.scale == "log":
                val = numpy.array([self.logbase ** v for v in val])
        else:
            if self.scale == "log":
                val = numpy.logspace(self.min_val, self.max_val, num_vals, base=self.logbase)
            else:
                val = numpy.linspace(self.min_val, self.max_val, num_vals)
        if self.interval:
            return (numpy.floor(val / self.interval) * self.interval).astype(int)

        return val

    def get_transformed_param(self, x):
        if self.distrib == 'normal':
            print('not implemented')
            return None
        else:
            val = x
            if self.scale == "log":
                val = self.logbase**x
            if self.interval:
                val = (numpy.floor(val / self.interval) * self.interval).astype(int)
        return val

    def get_min(self):
        return self.min_val

    def get_max(self):
        return self.max_val

    def get_type(self):
        if self.interval:
            return 'integer'
        return 'continuous'


class IntParam(Param):
    '''
    discrete (integer) parameters
    '''
    def __init__(self, name, min_val, max_val, init_val=None):
        super(IntParam, self).__init__(name, min_val, max_val, init_val=init_val)
        self.param_type = "integer"

    def get_param_range(self, num_vals, stochastic=False):
        # If num_vals greater than range of integer param then constrain to the range and if stochastic param results in
        # duplicates, only keep unique entries
        if stochastic:
            return numpy.unique(int( numpy.random.rand(num_vals)*(1 + self.max_val - self.min_val) + self.min_val ))

        return range(self.min_val, self.max_val+1, max(1, (self.max_val-self.min_val)/num_vals))


class CategoricalParam(object):
    '''
    categorical parameters
    '''
    def __init__(self, name, val_list, default):
        self.name = name
        self.val_list = val_list
        self.default = default
        self.init_val = default
        self.num_vals = len(self.val_list)
        self.param_type = 'categorical'

    def get_param_range(self, num_vals, stochastic=False):
        if stochastic:
            return random_combinations(self.val_list, num_vals)
        if num_vals >= self.num_vals:
            return self.val_list
        else:
            # return random subset, but include default value
            tmp = list(self.val_list)
            tmp.remove(self.default)
            random.shuffle(tmp)
            return [self.default] + tmp[0:num_vals-1]


def random_indices(high, size):
    return numpy.random.randint(high, size=size)


def random_combinations(val_list, num_vals, unique = True):
    rand_indices = random_indices(len(val_list), num_vals)
    if(unique):
        indices = numpy.unique(rand_indices)
    else:
        indices = rand_indices
    return [val_list[i] for i in indices]


class DenseCategoricalParam(object):
    '''
    Similar to CatogricalParam, but draws with replacement
    '''
    def __init__(self, name, val_list, default):
        self.name = name
        self.val_list = val_list
        self.default = default
        self.init_val = default
        self.num_vals = len(self.val_list)
        self.param_type = 'densecategorical'

    def get_param_range(self, num_vals, stochastic=False):
        if stochastic:
            return random_combinations(self.val_list, num_vals, unique=False)
        else:
            # return random subset, but include default value
            tmp = self.val_list.copy()
            tmp.remove(self.default)
            values = random_combinations(tmp, num_vals - 1, unique=False)
            return [self.default] + values


class PairParam(object):
    '''
    parameters combosed of two sub-parameters (keys, values)
    '''
    def __init__(self, name, get_param1_val, param1_key, current_arm, param2,
                 default):
        self.name = name
        self.current_arm = current_arm
        self.param1_key = param1_key
        self.get_param1_val = get_param1_val
        self.param2 = param2
        self.default = default
        self.init_val = default
        self.param_type = 'pair'

    def get_param_range(self, unused_num_vals, stochastic=False):

        # get from input param key, get a corresponding set of random values
        val_p1 = self.get_param1_val(self.current_arm, self.param1_key)
        vals_p2 = self.param2.get_param_range(val_p1, stochastic)

        # FIXME yet another trick so that generate_random_arm in problem_def.py
        # takes the whole list if any
        return (vals_p2, val_p1)


class ConditionalParam(object):
    '''
    draws a parameter with a constrained condition
    '''
    def __init__(self, cond_param, cond_val, param):
        self.name = param.name
        self.cond_param = cond_param
        self.cond_val = cond_val
        self.param = param
        self.param_type = 'conditional'

    def check_condition(self, hps):
        if self.cond_param not in hps:
            return None
        if hps[self.cond_param] == self.cond_val:
            return self.param
        return None


# Adaptive heuristic zooms into a local portion of the search space.
# Not recommended for actual use as there are no theoretical guarantees.
def zoom_space(params, center, pct=0.40):
    '''
    todo
    '''
    new_params = copy.deepcopy(params)
    for p in params.keys():
        range = params[p].max_val - params[p].min_val
        best_val = center[p]
        if params[p].scale == 'log':
            best_val = numpy.log(best_val)
        new_min = max(params[p].min_val, best_val-pct/2*range)
        new_max = new_min+(pct*range)
        new_params[p].min_val = new_min
        new_params[p].max_val = new_max

    return new_params
