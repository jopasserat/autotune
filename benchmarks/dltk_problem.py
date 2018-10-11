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

    def __init__(self, data_dir, output_dir, train_csv = "train.csv", validation_csv = "val.csv"):

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_csv = train_csv
        self.validation_csv = validation_csv

        super(DLTKProblem, self).__init__(data_dir, output_dir)
        # Set this to choose a subset of tunable hyperparams
        # self.hps = None
        self.hps = ['num_residual_units', 'learning_rate', 'nb_scales',
                    'filters', 'strides']

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
        '''
        parse parameters with value and type
        '''
        strides_values = DenseCategoricalParam("strides_values",
                                               [[1, 1, 1], [2, 2, 2]], [1, 1, 1])
        filters_values = DenseCategoricalParam("filters_values",
                                               [16, 64, 128, 256, 512], 16)

        # makes sure to draw in order from parameters sequentially in the same
        # order as the insertion order
        params = OrderedDict([
            ("num_residual_units", IntParam("num_residual_units", 1, 8, 3)),
            ("learning_rate", Param("learning_rate", -6, 0, distrib='uniform',
                                    scale='log', logbase=10)),
            ("nb_scales", IntParam("nb_scales", 1, 8, 4)),
            # FIXME what is a proper default value??
            ("filters", PairParam("filters", get_param_vals, "nb_scales",
                                  self.current_arm, filters_values, 42)),
            ("strides", PairParam("strides", get_param_vals, "nb_scales",
                                  self.current_arm, strides_values, 42))
        ])

        return params

    # FIXME can we reuse same prototype?
    def save_checkpoint(self, path, epoch, model, reader, reader_example_shapes):

        export_dir = model.export_saved_model(
            export_dir_base=path,
            serving_input_receiver_fn=reader.serving_input_receiver_fn(reader_example_shapes))
        print('Model saved to {}.'.format(export_dir))

    def construct_model(self, arm):
        """
        Construct model and optimizer based on hyperparameters
        :param arm:
        :return:
        """
        arm["save_path"] = arm['dir'] + "/synapse_ct_seg"

        # force FCN and cross entropy for the moment
        static_params = {
            "net": "fcn",
            "loss": "ce",
            "opt": "momentum"
        }
        arm_params = dict((k, convert_if_numpy(arm[k])) for k in self.hps)
        # nb_scales is only needed to generate correct number of filters and strides, don't need to carry it any further
        arm_params.pop("nb_scales")

        config_dict = merge_two_dicts(static_params, arm_params)

        # save DLTK config to JSON file
        config_file = arm['dir'] + "/config.json"
        with open(config_file, "w") as config_file:
            json.dump(config_dict, config_file, indent=2)

        # now create model DLTK style
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        synapse_model = SynapseMultiAtlas()

        # Instantiate the neural network estimator
        model = tf.estimator.Estimator(
            model_fn=synapse_model.model_fn,
            model_dir=arm["save_path"],
            params=config_dict,
            config=tf.estimator.RunConfig(session_config=config))

        (reader, reader_example_shapes, _) = setup_reader()
        # FIXME looks like i'll have to train at least 1 step :(
        synapse_model.mock_train(self.utils, model)
        self.save_checkpoint(arm["save_path"], 0, model, reader, reader_example_shapes)

        return arm["save_path"]

    def eval_arm(self, arm, n_resources):
        print("<<eval_arm not implemented yet>>")
