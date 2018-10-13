# deepMedic Autotune
#
import argparse


# =============================================================================


def generate_net(params, modelname, output_dir='./'):
    ''' generate an initial model with specific params
    returns:
        config file of the initial model
    '''
    # generate config file with given params
    modelname = (str)(params)
    # create model

    pass

def train(model, train_data_dir, output_dir='./', gpu_dev=0):
    ''' train network
    returns:
        trained model
    '''
    # train model

    # saved model

    pass

def test(model, test_data_dir):
    ''' test model and return accuracy
    '''
    #

    pass





# =============================================================================

def initialise_domain(self):
    params = {
        # 'learning_rate': Param('learning_rate', -6, 0, distrib='uniform',
        #                        scale='log', logbase=10),
        # 'weight_decay':  Param('weight_decay', -6, -1, distrib='uniform',
        #                        scale='log', logbase=10),
        # 'momentum':      Param('momentum', 0.3, 0.9, distrib='uniform',
        #                        scale='linear'),
        'batchSizeTrain': Param('batchSizeTrain', 5, 30, distrib='uniform',
                               scale='linear', interval=1),
    }
    return params


def eval_arm(self, params, n_resources):
    model = generate_net(params)
    model.train(n_iter=n_resources)
    acc = model.test()
    return 1-acc
