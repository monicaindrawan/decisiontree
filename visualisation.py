import matplotlib.pyplot as plt
from tree import DecisionTree


def _plot_tree_recursive(node=None, coordinates = (0,0), depth=0, vertical_distance=0.8, horizontal_distance=50):
    """ Plot the visualisation of a decision tree starting from a node 
        and explore its left and right child nodes nd recurse until reaching the terminal node (leaf).

        Args:
            node (Node instance): A node in the decision tree
            coordinates (tuple): The coordinates of the node in the final plot of the tree
            depth (int): Depth of the tree up to the node being explored
            vertical_distance (float): Vertical distance from the node to its child node in the plot
            horizontal_distance (float): Horizontal distance from the node to its left or right child node 
                                        in the plot before scaling.
        

    """
    (x, y) = coordinates
    
    # Create a green textbox and label the node as a leaf and its majority class label if it is a terminal node
    if node.is_terminal_node():
        node_label = f"leaf: {node.value}"
        plt.text(x, y, node_label, va="center", ha="center", color='green', bbox=dict(facecolor="lightblue", edgecolor="blue", boxstyle="round"), fontsize=5)
    
    # Otherwise the node is a decision node
    else:
        # Label the node with the corresponding feature to split the data and its value
        node_label = f"[X{node.feature}≤{node.threshold}]"
        plt.text(x, y, node_label, va="center", ha="center", bbox=dict(facecolor="lightgreen", edgecolor="green", boxstyle="round"), fontsize=5)

    # Explore the left child of the node and apply recursion
    if node.left is not None:
        scaled_horizontal_distance = max(horizontal_distance / (2**depth+1), 2)
        # Locate the left child node below and to the left of the parent node
        x_left = x - scaled_horizontal_distance
        y_left = y - vertical_distance
        # Plot the connecting line from the parent to the left child node
        plt.plot([x, x_left], [y, y_left], "-", linewidth=0.7, color="orange")
        _plot_tree_recursive(node.left, coordinates=(x_left, y_left), depth=depth+1)
    
    # Explore the right child of the node
    if node.right is not None:
        scaled_horizontal_distance = max(horizontal_distance / (2**depth+1), 2)
        # Locate the right child node below and to the right of the parent node
        x_right = x + scaled_horizontal_distance
        y_right = y - vertical_distance
        # Plot the connecting line from the parent to the right child node
        plt.plot([x, x_right], [y, y_right], "-", linewidth=0.7, color="black")
        _plot_tree_recursive(node.right, coordinates=(x_right,y_right), depth=depth+1)
    


def plot_tree(tree: DecisionTree, figsize=(16, 10)):
    """ Plot the visualisation of the decision tree starting from the root of the tree and 
        show the plot.

        Args:
            tree (DecisionTree instance): The decision tree to visualise
            figsize (tuple): The size of the figure to produce
    """
    # Check if the root contains anything
    if tree.root is None:
        print("Tree is empty.")
        return
    
    # Create a figure for the plot
    plt.figure(figsize=figsize)
    plt.title("Decision tree visualisation")
    plt.axis("off")

    # Call the recursive plotting logic
    _plot_tree_recursive(node=tree.root)
    plt.show()