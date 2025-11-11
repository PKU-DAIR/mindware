
def equal_ens(ens1, ens2):
    """
        "meta_learner": "best",
        "stack_layers": 2,
        "ensemble_size": 38,
        "ratio": 26,
        "dropout": 20
    """
    assert ens1 is not None or ens2 is not None

    if ens1 is None or ens2 is None:

        return False

    if 'ensemble_size' in ens1[0] and ens1[0]['ensemble_size'] != ens2[0]['ensemble_size']:
        return False

    if 'ratio' in ens1[0] and ens1[0]['ratio'] != ens2[0]['ratio']:
        return False

    if 'meta_learner' in ens1[0] and ens1[0]['meta_learner'] != ens2[0]['meta_learner']:
        return False

    if 'stack_layers' in ens1[0] and ens1[0]['stack_layers'] != ens2[0]['stack_layers']:
        return False

    if 'dropout' in ens1[0]:
        if ens1[0]['stack_layers'] == 0:
            return True
        else:
            return ens1[0]['dropout'] == ens2[0]['dropout']


def better_ens(ens1, ens2):

    assert ens1 is not None or ens2 is not None

    if ens1 is None or ens2 is None:

        return ens2 is None

    # Perf
    priority_fields = ['val', 'val_2', 'train', 'train_2']
    for field in priority_fields:
        if field in ens1[1]:
            if ens1[1][field] != ens2[1][field]:
                return ens1[1][field] > ens2[1][field]

    if 'ensemble_size' in ens1[0] and 'ratio' in ens1[0]:
        if ens1[0]['ensemble_size'] * ens1[0]['ratio'] != ens2[0]['ensemble_size'] * ens2[0]['ratio']:
            return ens1[0]['ensemble_size'] * (ens1[0]['ratio'] + 1) > ens2[0]['ensemble_size'] * (ens2[0]['ratio'] + 1)

    if 'stack_layers' in ens1[0]:
        if ens1[0]['stack_layers'] != ens2[0]['stack_layers']:
            return ens1[0]['stack_layers'] < ens2[0]['stack_layers']

    return False