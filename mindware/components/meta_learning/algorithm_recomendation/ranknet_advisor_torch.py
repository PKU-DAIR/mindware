import os
import numpy as np
import pickle as pk

from torch import nn, optim, from_numpy
import torch
from torch.utils.data import Dataset, DataLoader

from mindware.utils.logging_utils import get_logger
from mindware.components.meta_learning.algorithm_recomendation.base_advisor import BaseAdvisor
from mindware.components.meta_learning.algorithm_recomendation.arutils import EarlyStopping


class CategoricalHingeLoss(nn.Module):
    def forward(self, input, target):
        pos = (1. - target) * (1. - input) + target * input
        neg = target * (1. - input) + (1. - target) * input
        return torch.sum(torch.max(torch.zeros_like(neg - pos + 1.), neg - pos + 1.)) / len(input)


class PairwiseDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1_array, self.X2_array, self.y_array = X1, X2, y.reshape(y.shape[0], 1)

    def __getitem__(self, index):
        data1 = from_numpy(self.X1_array[index]).float()
        data2 = from_numpy(self.X2_array[index]).float()
        y_true = from_numpy(self.y_array[index]).float()
        return data1, data2, y_true

    def __len__(self):
        return self.X1_array.shape[0]


class RankNet(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, activation):
        super(RankNet, self).__init__()
        self.model = nn.Sequential()
        self.input_shape = input_shape
        self.output_sigmoid = nn.Sigmoid()
        self.act_func_dict = {'relu': nn.ReLU(inplace=True), 'tanh': nn.Tanh()}
        self.model.add_module('BatchNorm', nn.BatchNorm1d(input_shape))
        self.model.add_module('linear_' + str(hidden_layer_sizes[0]), nn.Linear(input_shape, hidden_layer_sizes[0]))
        self.model.add_module('act_func_' + str(0), self.act_func_dict[activation[0]])
        for i in range(1, len(hidden_layer_sizes)):
            self.model.add_module('linear_' + str(hidden_layer_sizes[i]),
                                  nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
            self.model.add_module('act_func_' + str(i),
                                  self.act_func_dict[activation[i]])
        self.model.add_module('output', nn.Linear(hidden_layer_sizes[-1], 1))

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        return self.output_sigmoid(s1 - s2)

    def predict(self, input):
        return self.model(input).detach()


class RankNetAdvisor(BaseAdvisor):
    def __init__(self,
                 rep=3,
                 metric='acc',
                 task_type=None,
                 total_resource=1200,
                 exclude_datasets=None,
                 meta_dir=None,
                 use_gpu=True):
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        super().__init__(task_type, metric, rep, total_resource,
                         'ranknet', exclude_datasets, meta_dir)
        self.model = None
        self.device = torch.device('cpu')
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        print("Device:", self.device)

    @staticmethod
    def create_pairwise_data(X, y):
        X1, X2, labels = list(), list(), list()
        n_algo = y.shape[1]

        for _X, _y in zip(X, y):
            if np.isnan(_X).any():
                continue
            meta_vec = _X
            for i in range(n_algo):
                for j in range(i + 1, n_algo):
                    if (_y[i] == -1) or (_y[j] == -1):
                        continue

                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x1 = list(meta_vec.copy())
                    meta_x1.extend(vector_i.copy())

                    meta_x2 = list(meta_vec.copy())
                    meta_x2.extend(vector_j.copy())

                    X1.append(meta_x1)
                    X1.append(meta_x2)
                    X2.append(meta_x2)
                    X2.append(meta_x1)
                    _label = 1 if _y[i] > _y[j] else 0
                    labels.append(_label)
                    labels.append(1 - _label)
        return np.asarray(X1), np.asarray(X2), np.asarray(labels)

    @staticmethod
    def create_model(input_shape, hidden_layer_sizes, activation):
        return RankNet(input_shape, hidden_layer_sizes, activation)

    def weights_init(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight.data)  # use xavier instead of default he_normal
            model.bias.data.zero_()


    def _val(self, model, data_loader, loss_fun):

        model.eval()
        train_loss = 0
        train_samples = 0
        train_acc = 0
        for i, (data1, data2, y_true) in enumerate(data_loader):
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)
            y_true = y_true.to(self.device)

            y_pred = model(data1, data2)
            loss = loss_fun(y_pred, y_true)
            train_loss += loss.item() * len(data1)
            train_samples += len(data1)
            train_acc += np.sum(y_pred.detach().cpu().numpy().round() == y_true.detach().cpu().numpy())

        loss = train_loss / train_samples
        acc = train_acc / train_samples

        return loss, acc


    def fit(self, **kwargs):
        l1_size = kwargs.get('layer1_size', 256)
        l2_size = kwargs.get('layer2_size', 128)
        act_func = kwargs.get('activation', 'tanh')
        batch_size = kwargs.get('batch_size', 128)
        epochs = 200

        _X, _y = self.metadata_manager.load_meta_data()
        X1_all, X2_all, y_all = self.create_pairwise_data(_X, _y)

        from sklearn.model_selection import KFold
        ss = KFold(n_splits=5, random_state=1, shuffle=True)
        self.model = [None] * 5
        fold = 0
        for train_index, test_index in ss.split(range(len(y_all) // 2)):
            print("========== Fold %d ==========\n" % (fold+1))

            meta_learner_dir = os.path.join(self.meta_dir, "meta_learner", "ranknet_model_%s_%s" % (self.meta_algo, self.metric))
            meta_learner_filename = os.path.join(meta_learner_dir, 'ranknet_model_%s_%s_%d_%s.pth' % (self.meta_algo, self.metric, fold, self.hash_id))
            if not os.path.exists(meta_learner_dir):
                os.makedirs(meta_learner_dir)
            if os.path.exists(meta_learner_filename):
                # print("load model...")
                self.model[fold] = torch.load(meta_learner_filename, map_location=self.device)
            else:
                train_mask = np.zeros(y_all.shape, dtype=bool)
                train_mask[2*train_index] = True
                train_mask[2*train_index+1] = True
                test_mask = np.zeros(y_all.shape, dtype=bool)
                test_mask[2*test_index] = True
                test_mask[2*test_index+1] = True

                X1_train, X2_train, y_train = X1_all[train_mask], X2_all[train_mask], y_all[train_mask]
                X1_val, X2_val, y_val = X1_all[test_mask], X2_all[test_mask], y_all[test_mask]

                print("train: X.shape:", X1_train.shape, X2_train.shape, "y.shape", y_train.shape)
                print("val: X.shape:", X1_val.shape, X2_val.shape, "y.shape", y_val.shape)

                train_data = PairwiseDataset(X1_train, X2_train, y_train)
                train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
                val_data = PairwiseDataset(X1_val, X2_val, y_val)
                val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=2)
                self.input_shape = X1_all.shape[1]

                es = EarlyStopping()
                # print("fit model...")
                self.model[fold] = RankNet(X1_all.shape[1], (l1_size, l2_size,), (act_func, act_func,)).to(self.device)
                self.model[fold].apply(self.weights_init)
                optimizer = optim.Adam(self.model[fold].parameters(), lr=1e-3)

                loss_fun = CategoricalHingeLoss()

                train_loss, train_acc = self._val(model=self.model[fold], data_loader=train_loader, loss_fun=loss_fun)
                val_loss, val_acc = self._val(model=self.model[fold], data_loader=val_loader, loss_fun=loss_fun)

                print('Initial, train_loss : {}, train_acc : {} | val_loss : {}, val_acc : {}'.format(train_loss, train_acc, val_loss, val_acc))

                for epoch in range(epochs):
                    self.model[fold].train()
                    train_loss = 0
                    train_samples = 0
                    train_acc = 0
                    for i, (data1, data2, y_true) in enumerate(train_loader):
                        data1 = data1.to(self.device)
                        data2 = data2.to(self.device)
                        y_true = y_true.to(self.device)
                        
                        optimizer.zero_grad()
                        y_pred = self.model[fold](data1, data2)
                        loss = loss_fun(y_pred, y_true)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * len(data1)
                        train_samples += len(data1)
                        train_acc += np.sum(y_pred.detach().cpu().numpy().round() == y_true.detach().cpu().numpy())

                    val_loss, val_acc = self._val(model=self.model[fold], data_loader=val_loader, loss_fun=loss_fun)
                    print('Epoch {}, train_loss : {}, train_acc : {} | val_loss : {}, val_acc : {}'.format(epoch, train_loss / train_samples, train_acc / train_samples, val_loss, val_acc))
                    es(val_loss=val_loss, model=self.model[fold], path=meta_learner_filename)

                    if es.early_stop:
                        print("Early stop after %d iterations!" % es.patience)
                        break

                # print("save model...")
                # torch.save(self.model[fold], meta_learner_filename)

                self.model[fold] = torch.load(meta_learner_filename, map_location=self.device)

            fold += 1

    def predict(self, dataset_meta_feat):
        n_algo = self.n_algo_candidates
        _X = list()
        for i in range(n_algo):
            vector_i = np.zeros(n_algo)
            vector_i[i] = 1
            _X.append(list(dataset_meta_feat.copy()) + list(vector_i))
        X = np.asarray(_X)
        X = from_numpy(X).float().to(self.device)
        pred = list()
        for model in self.model:
            model.eval()
            _pred = model.predict(X).cpu().numpy()
            pred.append(_pred)

        pred = np.mean(pred, axis=0)

        return pred.ravel()
