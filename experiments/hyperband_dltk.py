import pickle
import argparse

from core.HyperbandOptimiser import HyperbandOptimiser
from benchmarks.dltk_problem import DLTKProblem
import os

data_dir = '/home/jopasserat/postdoc/openmole/hyperparam-tuning/data'
output_dir = '/tmp/exp1'

parser = argparse.ArgumentParser(description='DLTK Synapse Training')
parser.add_argument('-i', '--input_dir', default=data_dir, type=str, help='input dir where the csv files are')
parser.add_argument('-o', '--output_dir', default=output_dir, type=str, help='output dir where to checkpoint models')
parser.add_argument('-res', '--n_resources', default=3, type=int, help='n_resources')
parser.add_argument('-s', '--seed', default=42, type=int, help='random seed for python and TensorFlow')
parser.add_argument('--cuda_devices', '-c', default='0')

args = parser.parse_args()

print(args.input_dir)
print(args.output_dir)

# GPU allocation options
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

# Define maximum units of resource assigned to each optimisation iteration
n_resources = args.n_resources

problem = DLTKProblem(args.input_dir, args.output_dir, args.seed)
problem.print_domain()

# Run hyperband
hyperband_opt = HyperbandOptimiser()
hyperband_opt.run_optimization(problem, max_iter=n_resources, verbosity=True)

print("Optimal arm: "
      "parameters = {}"
      "top losses = {}"
      "opt res = {}".format(hyperband_opt.arm_opt, hyperband_opt.Y_best, hyperband_opt.fx_opt)
)

# Constrain random optimisation to the same time budget
time_budget = hyperband_opt.checkpoints[-1]
print("Time budget = {}s".format(time_budget))

filename = os.path.join(args.output_dir, 'results.pkl')
with open(filename, 'wb') as f:
    pickle.dump([hyperband_opt], f)
