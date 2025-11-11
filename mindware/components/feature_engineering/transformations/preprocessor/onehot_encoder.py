from mindware.components.feature_engineering.transformations.base_transformer import *


class OneHotTransformation(Transformer):
    type = 2

    def __init__(self):
        super().__init__("onehot_encoder")
        self.input_type = CATEGORICAL

    def operate(self, input_datanode: DataNode, target_fields=None):
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder

        if target_fields is None:
            target_fields = collect_fields(input_datanode.feature_types, self.input_type)
        X, y = input_datanode.data
        # Fetch the fields to transform.
        self.target_fields = target_fields

        if isinstance(X, pd.DataFrame):
            X = X.values
        X_input = X[:, target_fields]

        if self.model is None:
            self.model = OneHotEncoder(handle_unknown='ignore')
            self.model.fit(X_input)
        new_X = self.model.transform(X_input).toarray()
        X_output = X.copy()

        categories = self.model.categories_
        input_feature_map = input_datanode.feature_map.copy()
        new_feature_map = [map_ for i,map_ in enumerate(input_feature_map) if i not in target_fields]
        n_max = max(new_feature_map)+1 if len(new_feature_map)!=0 else 0
        
        new_feature_map += [i+n_max  for i in range(len(categories)) for j in range(len(categories[i]))]
    
        # Delete the original columns.
        X_output = np.delete(X_output, np.s_[target_fields], axis=1)
        X_output = np.hstack((X_output, new_X))
        feature_types = input_datanode.feature_types.copy()
        feature_types = list(np.delete(feature_types, target_fields))
        feature_types.extend([CATEGORICAL] * new_X.shape[1])
        output_datanode = DataNode((X_output, y), feature_types, input_datanode.task_type, feature_map=new_feature_map)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode
