from mindware.components.utils.constants import *

def transform_mindware2autogluon_tasktype(tasktype):
    if tasktype == BINARY_CLS:
        return 'binary'
    elif tasktype == MULTICLASS_CLS:
        return 'multiclass'
    elif tasktype == REGRESSION:
        return 'regression'

    # (options: 'binary', 'multiclass', 'regression')




DEFAULT_MODEL_PRIORITY = dict(
    TABPFN=110,  # highest priority due to its very fast training time
    KNN=100,
    GBM=90,
    RF=80,
    CAT=70,
    XT=60,
    FASTAI=50,
    XGB=40,
    LR=30,
    NN_TORCH=25,
    VW=10,
    FASTTEXT=0,
    AG_TEXT_NN=0,
    AG_IMAGE_NN=0,
    AG_AUTOMM=0,
    TRANSF=0,
    custom=0,
    # interpretable models
    IM_RULEFIT=0,
    IM_GREEDYTREE=0,
    IM_FIGS=0,
    IM_HSTREE=0,
    IM_BOOSTEDRULES=0,
)



def trans_portfolio_to_params_list(portfolio, task_type):

    all_configs_with_priority = []
    for model, configs in portfolio.items():
        for config in configs:
            ag_args = config.get("ag_args", {}) if isinstance(config, dict) else {}
            if 'problem_types' in ag_args:
                if task_type not in ag_args['problem_types']: continue
            if isinstance(ag_args, dict) and "priority" in ag_args:
                priority = ag_args['priority']
            else:
                priority = DEFAULT_MODEL_PRIORITY[model]
            all_configs_with_priority.append([priority, [model, config]])


    sorted_configs = sorted(all_configs_with_priority, key=lambda X: -X[0])
    sorted_configs = [X[1] for X in sorted_configs]
    return sorted_configs