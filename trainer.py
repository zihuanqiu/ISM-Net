import sys
import logging

import numpy as np
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
from tensorboardX import SummaryWriter


def train(args):
    for order_i in args['order_ls']:
        args['order_i'] = order_i
        _train(args)


def _train(args):
    logfilename = 'logs/{}/{}_order{}_{}_{}_{}_{}'.format(args['prefix'], args['model_name'], args['order_i'],
                                                           args['convnet_type'], args['dataset'], args['init_cls'],
                                                           args['increment'])
    savefilename = 'ckpts/{}/{}_order{}_{}_{}_{}_{}'.format(args['prefix'], args['model_name'], args['order_i'],
                                                           args['convnet_type'], args['dataset'], args['init_cls'],
                                                           args['increment'])
    tbfilename = 'tensorboard/{}/{}_order{}_{}_{}_{}_{}'.format(args['prefix'], args['model_name'], args['order_i'],
                                                           args['convnet_type'], args['dataset'], args['init_cls'],
                                                           args['increment'])

    if not os.path.exists('logs/{}'.format(args['prefix'])):
        os.makedirs('logs/{}'.format(args['prefix']))
    if not os.path.exists(savefilename):
        os.makedirs(savefilename)
    if not os.path.exists(tbfilename):
        os.makedirs(tbfilename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    tensorboard = SummaryWriter(tbfilename)

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['order_i'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        model.before_task(data_manager)
        logging.info('All params: %.2fM' % ((count_parameters(model._network.convnet)+count_parameters(model._network.fc))/1000000.0))

        if task < args['start_task']:
            state_dict = torch.load("{}/model_step{}.ckpt".format(savefilename, task))
            memory = torch.load("{}/mem_step{}.ckpt".format(savefilename, task))
            model._data_memory, model._targets_memory, model._class_means = memory["x"], memory["y"], memory["m"]
            model._network.load_state_dict(state_dict)
            model._network.to(model._device)
        else:
            model.incremental_train(data_manager)
        if task >= args['start_task']:
            model.save_checkpoint(savefilename)
        model.after_task()  # TODO:  move wa to here
        cnn_accy, nme_accy = model.eval_task()
        model._known_classes = model._total_classes

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))

            tensorboard.add_scalar('CNN_top1_curve', cnn_accy['top1'], task)
            tensorboard.add_scalar('CNN_top5_curve', cnn_accy['top5'], task)
            tensorboard.add_scalar('NME_top1_curve', nme_accy['top1'], task)
            tensorboard.add_scalar('NME_top5_curve', nme_accy['top5'], task)

            if task == data_manager.nb_tasks-1:
                logging.info('CNN Average top1: {}'.format(np.mean(cnn_curve['top1'])))
                logging.info('CNN Average top5: {}\n'.format(np.mean(cnn_curve['top5'])))
                logging.info('NME Average top1: {}'.format(np.mean(nme_curve['top1'])))
                logging.info('NME Average top5: {}\n'.format(np.mean(nme_curve['top5'])))

                tensorboard.add_scalar('CNN_Average_top1', np.mean(cnn_curve['top1']))
                tensorboard.add_scalar('CNN_Average_top5', np.mean(cnn_curve['top5']))
                tensorboard.add_scalar('NME_Average_top1', np.mean(nme_curve['top1']))
                tensorboard.add_scalar('NME_Average_top5', np.mean(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))

            tensorboard.add_scalar('CNN_top1_curve', cnn_accy['top1'], task)
            tensorboard.add_scalar('CNN_top5_curve', cnn_accy['top5'], task)

            if task == data_manager.nb_tasks-1:
                logging.info('CNN Average top1: {}'.format(np.mean(cnn_curve['top1'])))
                logging.info('CNN Average top5: {}\n'.format(np.mean(cnn_curve['top5'])))
                tensorboard.add_scalar('CNN_Average_top1', np.mean(cnn_curve['top1']))
                tensorboard.add_scalar('CNN_Average_top5', np.mean(cnn_curve['top5']))


def _set_device(args):
    device_type = args['gpu']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
