from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.components.utils.dl_util import EarlyStop
from mindware.components.utils.configspace_utils import check_for_bool

DIMS = [64, 128, 256, 512]


class MyDataset(Dataset):
    def __init__(self, data, label):
        assert len(data) == len(label)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class NeuralNetworkClassifier(BaseClassificationModel):

    def __init__(self, optimizer, batch_size, epoch_num, lr_decay,
                 sgd_learning_rate=None, sgd_momentum=None, weight_decay=None, nesterov=None,
                 adam_learning_rate=None, beta1=None,
                 random_state=None, device='cuda', **kwargs):
        BaseClassificationModel.__init__(self)

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr_decay = lr_decay

        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_momentum = sgd_momentum
        self.weight_decay = weight_decay
        if nesterov is not None:
            nesterov = check_for_bool(nesterov)
        self.nesterov = nesterov

        self.adam_learning_rate = adam_learning_rate
        self.beta1 = beta1

        self.random_state = random_state
        self.model = None
        self.best_model_stats = None
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.time_limit = None
        self.load_path = None

    def build_model(self, input_shape, output_shape):
        idx = 0
        while idx < len(DIMS) - 1 and input_shape >= DIMS[idx] * 3/4:
            idx += 1

        if idx == 0:
            DIMS.insert(0, 32)
            idx += 1

        layer_list = [layer for i in range(idx, 0, -1) for layer in (nn.Linear(DIMS[i], DIMS[i - 1]), nn.ReLU())]

        model = nn.Sequential(
            nn.Linear(input_shape, DIMS[idx]),
            nn.ReLU(), 
            *layer_list,
            nn.Linear(DIMS[0], output_shape),
            nn.Softmax(dim=1)
        )
        return model.to(self.device)

    # init model with kaiming initialization
    def _init_model(self, model):

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def fit(self, X, Y, X_val=None, Y_val=None):

        from sklearn.utils.multiclass import unique_labels
        self.classes_ = unique_labels(Y)
        if self.model is None:
            self.model = self.build_model(X.shape[1], len(np.unique(Y)))

        self._init_model(self.model)

        train_loader = DataLoader(dataset=MyDataset(X, Y), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=MyDataset(X_val, Y_val), batch_size=self.batch_size, shuffle=False)

        self.model.to(self.device)
        self._init_model(self.model)

        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.sgd_learning_rate,
                                        momentum=self.sgd_momentum,
                                        weight_decay=self.weight_decay,
                                        nesterov=self.nesterov)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.adam_learning_rate,
                                         weight_decay=self.weight_decay,
                                         betas=(self.beta1, 0.999))
        else:
            raise ValueError('Invalid optimizer: %s' % self.optimizer)

        scheduler = MultiStepLR(optimizer, milestones=[int(self.epoch_num * 0.5), int(self.epoch_num * 0.75)],
                                gamma=self.lr_decay)
        loss_func = nn.CrossEntropyLoss()
        early_stop = EarlyStop(patience=10, mode='min')

        for epoch in range(self.epoch_num):
            self.model.train()
            # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])
            epoch_avg_loss = 0
            val_avg_loss = 0
            num_train_samples = 0
            num_val_samples = 0
            for i, data in enumerate(train_loader):
                batch_x, batch_y = data
                num_train_samples += len(data)
                logits = self.model(batch_x.float().to(self.device))
                loss = loss_func(logits, batch_y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)

            epoch_avg_loss /= num_train_samples

            if epoch % 10 == 0:
                print('Epoch %d: Train loss %.4f.' % (epoch, epoch_avg_loss))

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        batch_x, batch_y = data
                        num_val_samples += len(data)
                        logits = self.model(batch_x.float().to(self.device))
                        val_loss = loss_func(logits, batch_y.to(self.device))
                        val_avg_loss += val_loss.to('cpu').detach() * len(batch_x)

                    val_avg_loss /= num_val_samples
                    if epoch % 10 == 0:
                        print('Epoch %d: Val loss %.4f.' % (epoch, val_avg_loss))

                    early_stop.update(val_avg_loss)
                    if early_stop.cur_patience == 0:
                        self.best_model_stats = self.model.state_dict()
                    if early_stop.if_early_stop:
                        print("Early stop!")
                        break

            scheduler.step()

        self.model.load_state_dict(self.best_model_stats)

        return self

    def predict(self, X):
        if self.model is None:
            raise NotImplementedError()
        proba = self.model(torch.Tensor(X).to(self.device)).detach().cpu().numpy()
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        if self.model is None:
            raise NotImplementedError()
        return self.model(torch.Tensor(X).to(self.device)).detach().cpu().numpy()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NN',
                'name': 'Neural Network Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac', **kwargs):
        cs = ConfigurationSpace()
        # optimizer = CategoricalHyperparameter('optimizer', ['SGD', 'Adam'], default_value='Adam')
        optimizer = CategoricalHyperparameter('optimizer', ['Adam'], default_value='Adam')

        # sgd_learning_rate = CategoricalHyperparameter(
        #     "sgd_learning_rate", [1e-3, 3e-3, 7e-3, 1e-2, 3e-2, 7e-2, 1e-1], default_value=1e-1)
        # sgd_momentum = UniformFloatHyperparameter(
        #     "sgd_momentum", lower=0.5, upper=0.99, default_value=0.9, log=False)
        weight_decay = CategoricalHyperparameter("weight_decay", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                                                 default_value=1e-4)
        # nesterov = CategoricalHyperparameter('nesterov', ['True', 'False'], default_value='True')

        adam_learning_rate = UniformFloatHyperparameter(
            "adam_learning_rate", lower=1e-4, upper=1e-2, default_value=2e-3, log=True)
        beta1 = UniformFloatHyperparameter(
            "beta1", lower=0.5, upper=0.999, default_value=0.9, log=False)

        # batch_size = CategoricalHyperparameter(
        #     "batch_size", [16, 32, 64, 128, 256], default_value=32)
        batch_size = CategoricalHyperparameter(
            "batch_size", [128, 256, 512, 1024], default_value=256)
        lr_decay = CategoricalHyperparameter("lr_decay", [1e-2, 5e-2, 1e-1, 2e-1], default_value=1e-1)
        epoch_num = UnParametrizedHyperparameter("epoch_num", 150)

        cs.add_hyperparameters(
            [optimizer, weight_decay,
            #  sgd_learning_rate, sgd_momentum, nesterov,
             adam_learning_rate, beta1,
             batch_size, epoch_num, lr_decay])

        # sgd_lr_depends_on_sgd = EqualsCondition(sgd_learning_rate, optimizer, "SGD")
        # sgd_momentum_depends_on_sgd = EqualsCondition(sgd_momentum, optimizer, "SGD")
        # nesterov_depends_on_sgd = EqualsCondition(nesterov, optimizer, 'SGD')
        adam_lr_depends_on_adam = EqualsCondition(adam_learning_rate, optimizer, "Adam")
        beta_depends_on_adam = EqualsCondition(beta1, optimizer, "Adam")

        cs.add_conditions([
            # sgd_lr_depends_on_sgd, sgd_momentum_depends_on_sgd, nesterov_depends_on_sgd,
            adam_lr_depends_on_adam, beta_depends_on_adam
        ])

        return cs
