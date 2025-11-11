from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.modules.autogluon_zeroshot.utils import *
import numpy as np

class DataManager():
    def __init__(self,
                 label,   
                 task_type,
                 train_data,
                 test_data=None,):
        from autogluon.tabular.learner.default_learner import DefaultLearner
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator

        self.label = label
        self.problem_type = transform_mindware2autogluon_tasktype(task_type)

        self.learner = DefaultLearner(
            path_context ='',
            label = label,
            problem_type = self.problem_type,
            feature_generator = AutoMLPipelineFeatureGenerator(),
        )
        self.train_data = train_data
        self.test_data = test_data

    def get_train_node_and_test_node(self):
        a = self.learner.general_data_processing(self.train_data)
        train_data_node = DataNode(data=[a[0], a[1]])

        test_data_node =None
        if  self.test_data is not None:
            if self.label in self.test_data.columns:
                test_X = self.learner.transform_features(self.test_data.drop(columns=self.label))
                test_Y = self.learner.transform_labels(self.test_data[self.label])
                mask = test_Y.notna()
                test_data_node = DataNode(data = [test_X[mask], test_Y[mask]])
            else:
                test_X = self.learner.transform_features(self.test_data)
                test_data_node = DataNode(data = [test_X, None])
        else:
            test_data_node = None
        return train_data_node, test_data_node
    
    @staticmethod
    def split_data(data, label_col=-1, test_size=0.3, random_state=1, task_type=CLASSIFICATION):
        if task_type in CLS_TASKS:
            from sklearn.model_selection import StratifiedShuffleSplit
            spliter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            from sklearn.model_selection import ShuffleSplit
            spliter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in spliter.split(data, data.iloc[:, label_col]):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

        return train_data, test_data

    def _post_process_predict_proba(self, y_pred_proba, index=None):
        return self.learner._post_process_predict_proba(y_pred_proba=y_pred_proba, index=index)

    def _post_process_predict(self, y_pred, index=None):
        return self.learner._post_process_predict(y_pred=y_pred, index=index)
