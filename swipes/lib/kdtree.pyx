import numpy as np

class Node:
    def __init__(self):
        """
        node has 4 attributes
        left: left child
        right: right child
        dim: dimension that it is split by
        val: the value it is split by
        """
        self.left = None
        self.right = None
        self.dim = None
        self.val = None

def build_kdtree(vectors, node, dimension, max_dimension):
    # first find the 1-d array in the dimension of interest
    vals = vectors[:,dimension]

    # the argsorted index for this dimension
    idx = np.argsort(vals,kind='quicksort')

    # reorder the N-d vector
    vectors = vectors[idx]

    # base case
    if len(vals) == 1:
        # This node is a leaf node
        node.val = vals[0]
        return node
    if len(vals) == 2:
        node.val = vals[1]
        node.left = Node()
        node.left.val = vals[0]
        return node

    # find the middle index of the sorted 1-d array along the dimension of interest
    mid = int((len(vals) - 1) / 2)

    # find the value of the split (median value along dimension)
    node.val = vals[mid]

    # find the dimension currently of interest
    node.dim = dimension % max_dimension

    # recurse through left tree
    node.left = build_kdtree(vectors[:mid,:], Node(), (dimension+1) % max_dimension, max_dimension)

    # recurse through right tree
    node.right = build_kdtree(vectors[:mid,:], Node(), (dimension+1) % max_dimension, max_dimension)
    return node
