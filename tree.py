import numpy as np
from support import *
from evaluation import accuracy


class Node():
    """A node in the decision tree.

    Attributes:
        value (int): The predicted class value if the node is a leaf. None for internal nodes.
        depth (int): The depth of the node in the tree (root is 0).
        left (Node): The left child node.
        right (Node): The right child node.
        parent (Node): The parent of this node.
        threshold (float): The feature value used to split at this node. None for leaf nodes.
        feature (int): The index of the feature used for splitting at this node. None for leaf nodes.
        dominant_y (int): The most frequent class label in the data subset that reached this node. Used for pruning.
    """
    def __init__(self, value = None, left = None, right = None, parent = None, threshold = None, feature = None, depth = 0, dominant_y = None):
        """
        Constructor for Node, initialises the member attributes
        """
        self.value = value
        self.depth = depth
        self.left = left
        self.right = right
        self.parent = parent
        self.threshold = threshold
        self.feature = feature
        self.dominant_y = dominant_y

    
    def is_terminal_node(self):
        """Checks if the node is a terminal node.

        Returns:
            bool: True if the node is a terminal node, False otherwise.
        """
        if self.value is None:
            return False
        else:
            return True
        
        
    def get_max_depth(self):
        """Recursively calculates the depth of the subtree rooted at this node.

        The depth is the number of nodes on the longest pathfrom this node to a leaf (inclusive). A single leaf node has depth 1.

        Returns:
            int: The depth of the subtree starting from this node.
        """
        #base case
        if self.is_terminal_node():
            return 1
        
        #recursive case
        left_depth = 0
        right_depth = 0
        if self.left:
            left_depth = self.left.get_max_depth()
        if self.right:
            right_depth = self.right.get_max_depth()
        
        return max(left_depth, right_depth) + 1


class DecisionTree():
    """ DecisionTree class that contains the root node and is used to create decision tree (with fit()) and to predict using that tree (with predict())
    """
    def __init__(self):
        self.root = None


    def fit(self, X, y, depth = 0, parent = None):
        """ Recursively splits the input data based off maximum information gain.

        Args:
            X (nd.array) : ndarray containing features that will be split by decision tree.
            y (nd.array) : ndarray containing the output values that are trying to be classified by decision tree.
            depth (int)  : Integer keeping trach of the depth of the current node.

        Returns:
            Node : Node class that contains information on its value and depth (if terminal node), otherwise contains its depth, child nodes, the threshold
            that it splits, and which feature that threshold belongs to in X
        
        """

        # Check for most frequent y class
        dominant_y = np.argmax(np.bincount(y))

        if len(np.unique(y)) == 1:
            return Node(value = np.argmax(np.bincount(y)), depth = depth, parent = parent)
        
        best_feature_index, best_split_value = find_best_split_feature(X, y)

        # calculate the left and right split masks
        left_mask = X[:, best_feature_index] <= best_split_value
        right_mask = X[:, best_feature_index] > best_split_value
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # If no split is possible, return a terminal node
        if len(y_left) == 0 or len(y_right) == 0:
            return Node(value = dominant_y, depth = depth, parent = parent, dominant_y = dominant_y)

        
        node = Node(
            depth = depth,
            threshold=best_split_value,
            feature=best_feature_index,
            left = None,
            right = None,
            parent = parent,
            dominant_y = dominant_y
        )

        # recursively build the left and right subtrees
        node.left = self.fit(X_left, y_left, depth = depth + 1, parent=node)
        node.right = self.fit(X_right, y_right, depth = depth + 1, parent=node)

        # Set root node
        if depth == 0:
            self.root = node
        return node
    

    def _traverse_tree(self, x, node=None):
        """ Traverses the tree based on information from x (1-D array) and outputs value of terminal node reached.

        Args:
            x (1-D ndarray) : feature space that is being predicted.
            node (Node) : Node class that x is currently being checked against in the decision tree.

        Returns:
            int : Value of terminal node based off input data x
        """
        if node is None:
            node = self.root
        if node.is_terminal_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    

    def predict(self, X):
        return np.array([self._traverse_tree(row) for row in X])
    
    
    def _get_node_with_leaves(self, node=None, result_list=[]):
        """ Fetch all nodes with two leaves as its child

        Args:
            node (Node) : Node class that are currently being checked
            result_list (List) : List to store nodes with two leaves

        Returns:
            None
        """
        if node is None:
            node = self.root

        # basecase
        if node.left.is_terminal_node() and node.right.is_terminal_node():
            result_list.append(node)
            return

        # recursion body
        if not node.left.is_terminal_node():
            self._get_node_with_leaves(node.left, result_list)

        if not node.right.is_terminal_node():
            self._get_node_with_leaves(node.right, result_list)


    def _try_merge(self, parent_node, mergable_node, merged_node, X_val, y_val):
        """ Try to merge nodes with two leaves if accuracy improves

        Args:
            parent_node (Node) : parent of the Node Class that are currently being checked
            mergable_node (Node) : Node class that are currently being checked
            merged_node (Node) : Node class after its two leaves being removed
            X_val (nd.array) : ndarray containing features on validation dataset
            y_val (nd.array) : ndarray containing labels on validation dataset

        Returns:
            bool : True if node is being pruned
        """
        # Check accuracy before merging
        predicted_classes = self.predict(X_val)
        accuracy_pre_merge = accuracy(y_val, predicted_classes)
        
        # Perform the merge
        side = "left" if mergable_node is parent_node.left else "right"
        original = getattr(parent_node, side)
        setattr(parent_node, side, merged_node)

        # Check accuracy after merging
        predicted_classes = self.predict(X_val)
        accuracy_post_merge = accuracy(y_val, predicted_classes)

        # If accuracy decreased, revert the merge
        if accuracy_post_merge < accuracy_pre_merge:
            setattr(parent_node, side, original)
            return False
        return True
    
    def prune_tree(self, X_val, y_val):
        """ Continually prune the decision tree until no further improvements (accuracy) are seen

        Args:
            X_val (nd.array) : ndarray containing features on validation dataset
            y_val (nd.array) : ndarray containing labels on validation dataset

        Returns:
            None
        """
        layer = 0
        max_depth_before_pruning = self.root.get_max_depth()
        
        # Iteratively prune the tree
        while True:
            num_of_pruned_nodes = 0
            mergable_nodes = []
            self._get_node_with_leaves(result_list=mergable_nodes)

            # Create merged nodes and try to prune
            for mergable_node in mergable_nodes:
                parent_node = mergable_node.parent

                merged_node = Node(
                    value=mergable_node.dominant_y, 
                    depth = mergable_node.depth,
                    parent = parent_node
                )

                if self._try_merge(parent_node, mergable_node, merged_node, X_val, y_val):
                    num_of_pruned_nodes += 1

            # If no nodes were pruned in this iteration, stop pruning
            if num_of_pruned_nodes == 0:
                break

            layer += 1
        
        # Get max depth after pruning
        max_depth_after_pruning = self.root.get_max_depth()

        return max_depth_before_pruning, max_depth_after_pruning
    


    # Plotting functions moved to visualisation.py. This is to ensure the lab machine can execute the core tree code without needing to import matplotlib

    # def _plot_tree_recursive(self, node=None, coordinates = (0,0), depth=0, vertical_distance=0.8, horizontal_distance=50):
    #     x, y = coordinates
    #     #plt.xlim(-90, 110)
    #     #plt.ylim(-60, 1)
        
    #     if node.is_terminal_node():
    #         #print(node.depth)
    #         node_label = f"leaf: {node.value}"
    #         plt.text(x, y, node_label, va="center", ha="center", color='green', bbox=dict(facecolor='lightblue', edgecolor='blue', boxstyle='round'), fontsize=5)
    #     else:
    #         node_label = f"[X{node.feature}≤{node.threshold}]"
    #         plt.text(x, y, node_label, va="center", ha="center", bbox=dict(facecolor='lightgreen', edgecolor='green', boxstyle='round'), fontsize=5)

    #     if node.left is not None:
    #         x_left = x - max(horizontal_distance / (2**depth+1), 2)
    #         y_left = y - vertical_distance
    #         plt.plot([x, x_left], [y, y_left], "-", linewidth=0.7, color="orange")
    #         self._plot_tree_recursive(node.left, coordinates=(x_left,y_left), depth=depth+1)
        
    #     if node.right is not None:
    #         x_right = x + max(horizontal_distance / (2**depth+1), 2)
    #         y_right = y - vertical_distance
    #         plt.plot([x, x_right], [y, y_right], "-", linewidth=0.7, color="black")
    #         self._plot_tree_recursive(node.right, coordinates=(x_right,y_right), depth=depth+1)
        
    #     #plt.tight_layout()
    #     #plt.axis('equal')
    #     #plt.axis('off')
    #     #plt.autoscale()


    # def plot_tree(self, figsize=(16, 10), vertical_distance=0.8, horizontal_distance=50):
    #     # Check if the root contains anything
    #     if self.root is None:
    #         print("Tree is empty.")
    #         return
    #     plt.figure(figsize=figsize)
    #     plt.title("Decision tree visualisation")

    #     # Call the recursive plotting logic
    #     self._plot_tree_recursive(self.root, (0,0), 0, vertical_distance, horizontal_distance)
    #     plt.show()