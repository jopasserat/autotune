from __future__ import division
# import torch
#
from benchmarks.cifar_problem import CifarProblem
# from data.svhn_data_loader import get_train_val_set, get_test_set
from core.params import *
from collections import OrderedDict


def get_param_vals(arm, param_key):
    return arm[param_key]

class DLTKProblem(CifarProblem):

    def __init__(self, data_dir, output_dir):
        super(DLTKProblem, self).__init__(data_dir, output_dir)

        # Set this to choose a subset of tunable hyperparams
        # self.hps = None
        self.hps = ['num_residual_units', 'learning_rate', 'nb_scales', 'filters', 'strides']

    # def initialise_data(self):
    #     # 40k train, 10k val, 10k test
    #     print('==> Preparing data..')
    #     train_data, val_data, train_sampler, val_sampler = get_train_val_set(data_dir=self.data_dir,
    #  valid_size=0.2)
    #     test_data = get_test_set(data_dir=self.data_dir)
    #
    #     self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, sampler=val_sampler,
    #   num_workers=2, pin_memory=False)
    #     self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True,
    #    num_workers=2, pin_memory=False)
    #     self.train_data = train_data
    #     self.train_sampler = train_sampler

    def initialise_domain(self):
        strides_values = DenseCategoricalParam("strides_values", [[1, 1, 1], [2, 2, 2]], [1, 1, 1])
        filters_values = DenseCategoricalParam("filters_values", [16, 64, 128, 256, 512], 16)

        params = OrderedDict([
            ("num_residual_units", IntParam("num_residual_units", 1, 8, 3)),
            ("learning_rate", Param("learning_rate", -6, 0, distrib='uniform', scale='log', logbase=10)),
            ("nb_scales", IntParam("nb_scales", 1, 8, 4)),
            # FIXME what is a proper default value??
            ("filters", PairParam("filters", get_param_vals, "nb_scales", self.current_arm, filters_values, 42)),
            ("strides", PairParam("strides", get_param_vals, "nb_scales", self.current_arm, strides_values, 42))
        ])

        return params

    def construct_model(self, arm):
        print("<<construct_model not implemented yet>>")

    def eval_arm(self, arm, n_resources):
        print("<<eval_arm not implemented yet>>")
