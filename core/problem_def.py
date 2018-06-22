import pprint
import abc

from collections import OrderedDict

class Problem(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # make sure keys are retrieved in same order for pair parameters
        self.domain = OrderedDict()
        # FIXME awful trick to retrieve generated values from PairParams given current design
        self.current_arm = {}

    def generate_arms(self, n, hps=None):
        arms = []
        for i in range(n):
            arms.append(self.generate_random_arm(hps))
        return arms

    def generate_random_arm(self, hps=None):
        if not hps:
            hps = self.domain.keys()
        self.current_arm.clear()
        for hp in self.domain.keys():
            # if sample is required, draw from param range
            if hp in hps:
                val = self.domain[hp].get_param_range(1, stochastic=True)
                self.current_arm[hp] = val[0]
            # else set to default
            else:
                val = self.domain[hp].init_val
                assert val is not None, "No default value is set for param {}".format(hp)
                self.current_arm[hp] = val
        return self.current_arm.copy()

    def print_domain(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.domain)

    @abc.abstractmethod
    def eval_arm(self, arm, n_resources):
        pass

    @abc.abstractmethod
    def initialise_domain(self):
        pass
