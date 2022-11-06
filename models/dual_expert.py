import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DualNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy


EPSILON = 1e-8

init_epoch = 170
init_lr = 0.1
init_milestones=[100, 120]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 170
lrate = 0.1
milestones = [100, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 0.0005
num_workers=8


class DualExpert(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = DualNet(args['convnet_type'], False)
        self.channel_per_class = args['channel_per_class']
        self.queue_length = args['queue_length']
        self.mix_ratio = args['mix_ratio']
        self._old_network = nn.ModuleList()

    def before_task(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task, self.channel_per_class)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

    def after_task(self):
        old_network = self._network.copy().freeze()
        if len(self._old_network) == 0:
            self._old_network.append(old_network)
        else:
            self._old_network.append(old_network)
        if len(self._old_network) > self.queue_length:
            self._old_network = self._old_network[-self.queue_length:]

        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes - self._known_classes)

    def incremental_train(self, data_manager):
        if self._cur_task == 0:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                     mode='train', appendent=None)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        else:
            train_dataset, memory_dataset = data_manager.get_dual_dataset(np.arange(self._known_classes, self._total_classes), appendent=self._get_memory())
            self.train_loader = (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                                 DataLoader(memory_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers))

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,lr=init_lr,weight_decay=init_weight_decay) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay)            
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            self._old_network.to(self._device)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss=F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            if epoch%5!=0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, init_epoch, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.
            correct, total = 0, 0
            for i, ((_, n_inputs, n_targets), (_, o_inputs, o_targets)) in enumerate(zip(train_loader[0], train_loader[1])):
                n_inputs, n_targets = n_inputs.to(self._device), n_targets.to(self._device)
                o_inputs, o_targets = o_inputs.to(self._device), o_targets.to(self._device)
                if isinstance(self.mix_ratio, list):
                    mix_ratio = torch.rand(n_inputs.size(0)).to(self._device)
                    mix_ratio = mix_ratio * (self.mix_ratio[1] - self.mix_ratio[0]) + self.mix_ratio[0]
                    mix_ratio = mix_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    mix_input = mix_ratio * n_inputs + (1 - mix_ratio) * o_inputs
                else:
                    mix_input = self.mix_ratio*n_inputs + (1-self.mix_ratio)*o_inputs

                # 第一轮优化
                inputs = torch.cat([n_inputs, o_inputs], dim=0)
                targets = torch.cat([n_targets, o_targets], dim=0)
                rp = torch.randperm(inputs.size(0))
                inputs, targets = inputs[rp], targets[rp]
                outputs = self._network(inputs)
                loss_reh = F.cross_entropy(outputs['logits'], targets)

                with torch.no_grad():
                    old_feat = []
                    old_feat.append(self._old_network[0].extract_vector(inputs))
                    for i in range(1, len(self._old_network)):
                        old_feat.append(self._old_network[i].feat_forward(inputs))
                    old_feat = torch.cat(old_feat, dim=1)

                loss_dis = torch.mean(torch.frobenius_norm(
                    outputs['memory_feat'] - old_feat, dim=-1)
                )

                aux_targets = targets.clone()
                aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0, aux_targets - self._known_classes + 1, 0)
                loss_aux = F.cross_entropy(outputs['aux_logits'], aux_targets)

                # 第二轮优化
                mix_outputs = self._network(mix_input)
                loss_cro = F.cross_entropy(mix_outputs['new_logits'], n_targets - self._known_classes)

                with torch.no_grad():
                    old_feat = []
                    old_feat.append(self._old_network[0].extract_vector(mix_input))
                    for i in range(1, len(self._old_network)):
                        old_feat.append(self._old_network[i].feat_forward(mix_input))
                    old_feat = torch.cat(old_feat, dim=1)

                loss_dis += torch.mean(torch.frobenius_norm(
                    mix_outputs['memory_feat'] - old_feat, dim=-1)
                )

                loss = loss_reh + loss_cro + loss_dis + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(outputs['logits'], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/i, train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs, losses/i,train_acc)
            prog_bar.set_description(info)
        logging.info(info)
