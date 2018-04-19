import pickle
import argparse

from ..core.HyperbandOptimiser import HyperbandOptimiser
from ..core.RandomOptimiser import RandomOptimiser
from ..benchmarks.cifar_problem_2 import CifarProblem2

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-i', '--input_dir', type=str, help='input dir')
parser.add_argument('-o', '--output_dir', type=str, help='output dir')
args = parser.parse_args()

print(args.input_dir)
print(args.output_dir)

# Define problem instance
problem = CifarProblem2(args.input_dir, args.output_dir)
problem.print_domain()

# Define maximum units of resource assigned to each optimisation iteration
n_resources = 81

# Run hyperband
hyperband_opt = HyperbandOptimiser()
hyperband_opt.run_optimization(problem, max_iter=n_resources, verbosity=True)

# Constrain random optimisation to the same time budget
time_budget = hyperband_opt.checkpoints[-1]
print("Time budget = {}".format(time_budget))

random_opt = RandomOptimiser()
random_opt.run_optimization(problem, n_resources, max_time=time_budget, verbosity=True)

filename = args.output_dir + 'results.pkl'
with open(filename, 'wb') as f:
    pickle.dump([hyperband_opt, random_opt], f)