from __future__ import division
# import torch
#
from benchmarks.cifar_problem import CifarProblem
# from data.svhn_data_loader import get_train_val_set, get_test_set
from core.params import *
from collections import OrderedDict


def get_param_vals(arm, param_key):
    return arm[param_key]

def setup_reader():
    NUM_CHANNELS = 1

    # Set up a data reader to handle the file i/o.
    reader_params = {
        'n_examples': 32,
        'example_size': [64, 64, 64],
        'extract_examples': True
    }

    reader_example_shapes = {
        'features': {'x': reader_params['example_size'] + [NUM_CHANNELS, ]},
        'labels': {'y': reader_params['example_size']}}

    reader = Reader(read_fn, {'features': {'x': tf.float32},
                              'labels': {'y': tf.int32}})

    return (reader, reader_example_shapes, reader_params)


class DLTKBoilerplate(object):
    """
    Just a value class (if that makes any sort of sense in Python) to aggregate all the utilities created for training.
    """
    def __init__(self, train_input_fn, train_qinit_hook, val_input_fn, val_qinit_hook, val_summary_hook, step_cnt_hook, train_filenames, val_filenames):
        self.train_input_fn = train_input_fn
        self.train_qinit_hook = train_qinit_hook
        self.val_input_fn = val_input_fn
        self.val_qinit_hook = val_qinit_hook
        self.val_summary_hook = val_summary_hook
        self.step_cnt_hook = step_cnt_hook
        self.train_filenames = train_filenames
        self.validation_filename = val_filenames

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

    def initialise_data(self):
        import pandas as pd
        import numpy as np
        import os

        BATCH_SIZE = 4
        SHUFFLE_CACHE_SIZE = 128

        # FIXME: move some place else
        np.random.seed(42)
        tf.set_random_seed(42)

        print('Setting up...')

        train_filenames = pd.read_csv(
            "{}/{}".format(self.data_dir, self.train_csv),
            dtype=object,
            keep_default_na=False,
            na_values=[]).values

        val_filenames = pd.read_csv(
            "{}/{}".format(self.data_dir, self.validation_csv),
            dtype=object,
            keep_default_na=False,
            na_values=[]).values

        (reader, reader_example_shapes, reader_params) = setup_reader()

        # Get input functions and queue initialisation hooks
        # for training and validation data
        train_input_fn, train_qinit_hook = reader.get_inputs(
            train_filenames,
            tf.estimator.ModeKeys.TRAIN,
            example_shapes=reader_example_shapes,
            batch_size=BATCH_SIZE,
            shuffle_cache_size=SHUFFLE_CACHE_SIZE,
            params=reader_params)

        val_input_fn, val_qinit_hook = reader.get_inputs(
            val_filenames,
            tf.estimator.ModeKeys.EVAL,
            example_shapes=reader_example_shapes,
            batch_size=BATCH_SIZE,
            shuffle_cache_size=min(SHUFFLE_CACHE_SIZE, EVAL_STEPS),
            params=reader_params)

        # Hooks for validation summaries
        val_summary_hook = tf.contrib.training.SummaryAtEndHook(
            os.path.join(self.output_dir, 'eval'))
        step_cnt_hook = tf.train.StepCounterHook(
            every_n_steps=EVAL_EVERY_N_STEPS, output_dir=self.output_dir)

        self.utils = DLTKBoilerplate(
            train_input_fn, train_qinit_hook, val_input_fn, val_qinit_hook, val_summary_hook, step_cnt_hook, train_filenames, val_filenames)

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
        # TODO load model from previous checkpoint and resume training
        # DLTK automatically resumes when finding a model in the specified path


class SynapseMultiAtlas(object):

    def __init__(self):
        self.NUM_CLASSES = 14


    # MODEL
    def model_fn(self, features, labels, mode, params):
        """Build architecture of network as an instance of tf.estimator.EstimatorSpec according
         to HPs from top-level optimiser.

        Args:
            features (TYPE): Description
            labels (TYPE): Description
            mode (TYPE): Description
            params (TYPE): Description

        Returns:
            TYPE: tf.estimator.EstimatorSpec
        """
        # 1. create a model and its outputs

        from dltk.core.metrics import dice
        from dltk.core.losses import sparse_balanced_crossentropy
        from dltk.networks.segmentation.unet import residual_unet_3d
        from dltk.networks.segmentation.unet import asymmetric_residual_unet_3d
        from dltk.networks.segmentation.fcn import residual_fcn_3d
        from dltk.core.activations import leaky_relu

        filters = params["filters"]
        strides = params["strides"]
        num_residual_units = params["num_residual_units"]
        loss_type = params["loss"]
        net = params["net"]

        def lrelu(x):
            return leaky_relu(x, 0.1)

        if net == 'fcn':
            net_output_ops = residual_fcn_3d(
                features['x'], self.NUM_CLASSES,
                num_res_units=num_residual_units,
                filters=filters,
                strides=strides,
                activation=lrelu,
                mode=mode)
        elif net == 'unet':
            net_output_ops = residual_unet_3d(
                features['x'], self.NUM_CLASSES,
                num_res_units=num_residual_units,
                filters=filters,
                strides=strides,
                activation=lrelu,
                mode=mode)
        elif net == 'asym_unet':
            net_output_ops = asymmetric_residual_unet_3d(
                features['x'],
                self.NUM_CLASSES,
                num_res_units=num_residual_units,
                filters=filters,
                strides=strides,
                activation=lrelu,
                mode=mode)

        # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=net_output_ops,
                export_outputs={'out': tf.estimator.export.PredictOutput(
                    net_output_ops)})

        # 2. set up a loss function
        if loss_type == 'ce':
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=net_output_ops['logits'], labels=labels['y'])
            loss = tf.reduce_mean(ce)
        elif loss_type == 'balce':
            loss = sparse_balanced_crossentropy(
                net_output_ops['logits'], labels['y'])

        # 3. define a training op and ops for updating
        # moving averages (i.e. for batch normalisation)
        global_step = tf.train.get_global_step()
        if params["opt"] == 'adam':
            optimiser = tf.train.AdamOptimizer(
                learning_rate=params["learning_rate"], epsilon=1e-5)
        elif params["opt"] == 'momentum':
            optimiser = tf.train.MomentumOptimizer(
                learning_rate=params["learning_rate"], momentum=0.9)
        elif params["opt"] == 'rmsprop':
            optimiser = tf.train.RMSPropOptimizer(
                learning_rate=params["learning_rate"], momentum=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimiser.minimize(loss, global_step=global_step)

        # 4.1 (optional) create custom image summaries for tensorboard
        my_image_summaries = {}
        my_image_summaries['feat_t1'] = tf.expand_dims(
            features['x'][:, 0, :, :, 0], 3)
        my_image_summaries['labels'] = tf.expand_dims(
            tf.cast(labels['y'], tf.float32)[:, 0, :, :], 3)
        my_image_summaries['predictions'] = tf.expand_dims(
            tf.cast(net_output_ops['y_'], tf.float32)[:, 0, :, :], 3)

        [tf.summary.image(name, image)
         for name, image in my_image_summaries.items()]

        # 4.2 (optional) create custom metric summaries for tensorboard
        dice_tensor = tf.py_func(
            dice, [net_output_ops['y_'], labels['y'],
                   tf.constant(self.NUM_CLASSES)], tf.float32)

        [tf.summary.scalar('dsc_l{}'.format(i), dice_tensor[i])
         for i in range(self.NUM_CLASSES)]

        # 5. Return EstimatorSpec object
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=net_output_ops,
            loss=loss, train_op=train_op,
            eval_metric_ops=None)

    def mock_train(self, utils, net):
        """
        Trigger minimal training in order to be able to export model.

        :param utils: Instance of DLTKBoilerplate
        :param net: Network to train, instance of tf.estimator.Estimator
        :return:
        """
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)

        (reader, reader_example_shapes, reader_params) = setup_reader()

        # Get input functions and queue initialisation hooks
        # for training and validation data
        train_input_fn, train_qinit_hook = reader.get_inputs(
            utils.train_filenames,
            tf.estimator.ModeKeys.TRAIN,
            example_shapes=reader_example_shapes,
            batch_size=1,
            shuffle_cache_size=1,
            params=reader_params)


        try:
            net.train(
                input_fn=utils.train_input_fn,
                steps=1)
        except:
            return
