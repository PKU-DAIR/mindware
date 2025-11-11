from mindware.components.optimizers.block_optimizers.joint_optimizer import JointOptimizer
from mindware.components.optimizers.block_optimizers.alternative_optimizer import AlternativeOptimizer
from mindware.components.optimizers.block_optimizers.conditional_optimizer import ConditionalOptimizer


# Define different execution plans
def get_opt_execution_tree(execution_id):
    # Each node represents (parent_id, node_type)
    trees = {0: [('joint', [])],
             1: [('condition', [1]), ('joint', [])],  # Default strategy
             2: [('condition', [1]), ('alternate', [2, 3]), ('joint', []), ('joint', [])],
             3: [('alternate', [1, 2]), ('joint', []), ('joint', [])],
             4: [('alternate', [1, 2]), ('joint', []), ('condition', [3]), ('joint', [])]}
    return trees[execution_id]


def get_opt_node_type(node_list, index):
    if node_list[index][0] == 'joint':
        root_class = JointOptimizer
    elif node_list[index][0] == 'condition':
        root_class = ConditionalOptimizer
    else:
        root_class = AlternativeOptimizer

    return root_class
