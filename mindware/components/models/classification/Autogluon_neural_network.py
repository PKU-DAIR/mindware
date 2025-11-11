#for cashfe， you may need the mp.set_start_method('spawn')
from torch import nn
import torch
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition
from torch.utils.data import DataLoader, Dataset
from mindware.components.models.base_model import BaseClassificationModel
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class MyDataset(Dataset):
    def __init__(self, data, label):
        assert len(data) == len(label)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]



class NNClassifier(BaseClassificationModel):
    def __init__(self, activation, optimizer, learning_rate, weight_decay, dropout_prob, 
                 epochs_wo_improve, num_layers, hidden_size, max_batch_size, epoch_num,
                 max_embedding_dim, embed_exponent, embedding_size_factor, 
                 random_state=None, use_batchnorm=False, device='cuda',
                 loss_function='cross_entropy', embed_min_categories=5):
        
        BaseClassificationModel.__init__(self)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.activation = activation
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_batchnorm = use_batchnorm
        self.loss_function = loss_function
        self.max_embedding_dim = max_embedding_dim
        self.embed_exponent = embed_exponent
        self.embedding_size_factor = embedding_size_factor

        self.epochs_wo_improve = epochs_wo_improve
        self.epoch_num = epoch_num

        self.max_batch_size = max_batch_size
        self.embed_min_categories = embed_min_categories

        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)

    def build_model(self,input_shape, output_shape):
        self.model = EmbedNet(input_shape, output_shape,
                              self.ind_number_map,
                              self.max_embedding_dim,
                              self.activation,
                              self.dropout_prob,
                              self.hidden_size,
                              self.num_layers,
                              self.use_batchnorm,
                              self.embed_exponent,
                              self.embedding_size_factor,
                              self.device,
                              self.embed_min_categories)
        self.model.to(self.device)


    def _init_model(self, model):

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if self.activation == 'relu':
                    nonlinearity = 'relu'
                elif self.activation == 'elu':
                    nonlinearity = 'linear'  
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity=nonlinearity)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def recover_X(self ,X):
        X_output = X.copy()
        feature_map = self.feature_map 
        map_dict = dict()
        for i, key in enumerate(feature_map):
            if key not in map_dict:
                map_dict[key] = [i]
            else:
                map_dict[key].append(i)
        
        need_pro = []

        for key, inds in map_dict.items():
            if len(inds) >= self.embed_min_categories:
                need_pro.append(inds)

        start_n = len(feature_map) - sum([len(inds) for inds in need_pro])
        ind_number_map = dict()
        delete_inds = []

        for i,inds in enumerate(need_pro):
            ind_number_map[start_n+ i] = len(inds)
            new_X = np.argmax(X_output[:, inds], axis=1, keepdims=True)
            X_output = np.hstack((X_output, new_X))
            delete_inds += inds
    

        X_output = np.delete(X_output, np.s_[delete_inds],  axis=1)
        self.ind_number_map = ind_number_map
        return X_output



    def fit(self, X, Y, X_val=None, y_val=None):
        import sklearn.multiclass
        from sklearn.utils.multiclass import unique_labels
        self.classes_ = unique_labels(Y)

        output_shape = len(np.unique(Y))
        X = self.recover_X(X)
        input_shape = X.shape[1]
      

        
        self.build_model(input_shape=input_shape,output_shape=output_shape)
        
        self._init_model(self.model)  
        
        if X_val is not None:
            X_val = torch.tensor(X_val)
            y_val = torch.tensor(y_val)

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer specified: {self.optimizer}")
        
        self.batch_size = min(int(2 ** (3 + np.floor(np.log10(len(X))))), self.max_batch_size)

       
        train_loader = DataLoader(dataset=MyDataset(X, Y), batch_size=self.batch_size, shuffle=True)
        
        if self.loss_function == 'cross_entropy':
            l = nn.CrossEntropyLoss()

        epoch = 0
        
        self.best_model_stats = None
        best_val_loss = np.inf
        best_epoch = 0
        while epoch < self.epoch_num:
            epoch_avg_loss = 0
            num_train_samples = 0
            self.model.train()
           
            for X_train, y_train in train_loader:
               
                if len(X_train) < 2:
                    continue
                num_train_samples += len(X_train)
                y_pred = self.model(X_train)
                
                loss = l(y_pred, y_train.long().to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_avg_loss += loss.to('cpu').detach() * len(X_train)
                

            epoch_avg_loss /= num_train_samples
            
            if epoch % 50 == 0:
                print('Epoch %d: Train loss %.4f.' % (epoch, epoch_avg_loss))

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    y_pred_ = self.model(X_val)
                    val_loss = l(y_pred_, y_val.long().to(self.device))
                    val_loss = val_loss.to('cpu').detach()

                if val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    self.best_model_stats = self.model.state_dict()
            
                if best_epoch + self.epochs_wo_improve < epoch:
                    break
            
                if epoch % 50 == 0:
                    print('Epoch %d: Val loss %.4f.' % (epoch, val_loss))
            
            epoch += 1
        
        if self.best_model_stats is not None:
            self.model.load_state_dict(self.best_model_stats)
        return self
    
    def predict(self, X):
        if self.model is None:
                raise NotImplementedError()
        self.model.eval()
        proba = self.model.predict_proba(X)
        return np.argmax(proba, axis=1)
        
    def predict_proba(self, X):
        if self.model is None:
            raise NotImplementedError()
        self.model.eval()
        return self.model.predict_proba(X)
    
    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ANN',
                'name': 'Autogluon Neural Network Classifier',
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
        optimizer = CategoricalHyperparameter('optimizer', ['adam'], default_value='adam')
        weight_decay = UniformFloatHyperparameter('weight_decay',lower=1e-12,upper=1.0, default_value=1e-6,log=True)
        activation = CategoricalHyperparameter('activation',['relu'], default_value='relu')
        learning_rate = UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=3e-2, default_value=3e-4, log=True)
        dropout_prob = CategoricalHyperparameter('dropout_prob', [0.1, 0.0, 0.5, 0.2, 0.3, 0.4], default_value=0.1)
        epochs_wo_improve = Constant('epochs_wo_improve',20)
        num_layers = UniformIntegerHyperparameter('num_layers', lower=2, upper=4, default_value=4)
        hidden_size = CategoricalHyperparameter('hidden_size',[128,256,512], default_value=128)
        max_batch_size = Constant('max_batch_size',512)
        use_batchnorm = CategoricalHyperparameter('use_batchnorm', [True, False])
        epoch_num = UnParametrizedHyperparameter("epoch_num", 500)
        max_embedding_dim = UnParametrizedHyperparameter('max_embedding_dim', 100)
        embed_exponent = UnParametrizedHyperparameter('embed_exponent', 0.56)
        embedding_size_factor = UnParametrizedHyperparameter('embedding_size_factor', 1.0)

        cs.add_hyperparameters([
            optimizer,weight_decay,activation,learning_rate,dropout_prob,epochs_wo_improve,num_layers,
            hidden_size,max_batch_size,use_batchnorm,epoch_num,max_embedding_dim,embed_exponent,embedding_size_factor
        ])
        return cs



class EmbedNet(nn.Module):
    def __init__(self, input_shape, output_shape, ind_number_map, max_embedding_dim, activation,
                 dropout_prob, hidden_size, num_layers, use_batchnorm,
                 embed_exponent, embedding_size_factor, device, embed_min_categories=5):
        super().__init__()
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm

        self.ind_number_map = ind_number_map
        self.embed_blocks = dict()

        self.device = device

        input_shape_ = 0
        

        self.embed_blocks_ = nn.ModuleList()
        i = 0
        self.embed_blocks = dict()
        for ind in ind_number_map:
            if ind_number_map[ind] >= embed_min_categories:
                embed_dim = int(embedding_size_factor * max(2, min(max_embedding_dim, 1.6 * ind_number_map[ind] ** embed_exponent)))
                self.embed_blocks_.append(nn.Embedding(num_embeddings=ind_number_map[ind], embedding_dim=embed_dim))

                self.embed_blocks[ind] = i
                i += 1
                input_shape_ += embed_dim


        self.v_ind = []
        for i in range(input_shape):
            if i not in self.embed_blocks.keys():
                self.v_ind.append(i)
 
        input_shape_ += len(self.v_ind)

        if self.activation == 'relu':
            act_fn = nn.ReLU()
        elif self.activation == 'elu':
            act_fn = nn.ELU()
        elif self.activation == 'tanh':
            act_fn = nn.Tanh()
        
        layers = []
        if self.use_batchnorm:
            layers.append(nn.BatchNorm1d(input_shape_))
        layers.append(nn.Dropout(self.dropout_prob))
        layers.append(nn.Linear(input_shape_,self.hidden_size))
        layers.append(act_fn)
        for i in range(self.num_layers-1):
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.Dropout(self.dropout_prob))
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(act_fn)
        layers.append(nn.Linear(self.hidden_size, output_shape))
        self.mainblock = nn.Sequential(*layers)
        

    def forward(self, X):
        input_data = []
        input_data.append(X[:, self.v_ind].float().to(self.device))
        
        for ind in self.embed_blocks:
            
            i = self.embed_blocks[ind]
            input_data.append(self.embed_blocks_[i](X[:,ind].long().to(self.device)))

        
        if len(input_data) >1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]

        
        output_data = self.mainblock(input_data)
       
        return output_data
        

   
    def predict_proba(self,X):
        with torch.no_grad():
            proba = nn.functional.softmax(self(torch.Tensor(X)),dim=1)
            proba = proba.detach().cpu().numpy()
        return proba