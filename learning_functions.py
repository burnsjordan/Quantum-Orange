import numpy as np
import random
import monte_carlo


# Greedy algorithm that calculates the norm for each possible gate and returns
# the best one
def greedy_algo(target_matrix, current_matrix, gates_list, trained_ai):
    cost_list = {}
    for x in gates_list:
        cost_list[x] = np.linalg.norm(
            target_matrix-np.dot(gates_list[x], current_matrix))
    return gates_list[min(cost_list, key=cost_list.get)]


# Greedy algorithm that searches up to a maximum depth
def deep_greedy_algo(target_matrix, current_matrix, gates_list, trained_ai):
    depth = 5
    trees = []
    for x in gates_list:
        trees.append(monte_carlo.Tree(matrix=np.dot(gates_list[x], current_matrix), gate=gates_list[x]).root)
    current_level = trees
    current_depth = 1
    current_best = monte_carlo.Node(matrix=current_matrix, gate=current_matrix)
    error = 0.1
    while(current_depth < depth):
        next_level = []
        for x in current_level:
            if(np.linalg.norm(target_matrix-x.matrix) < error):
                return x.root.gate
            elif(np.linalg.norm(target_matrix-x.matrix) < np.linalg.norm(target_matrix-current_best.matrix)):
                current_best = x
        for x in current_level:
            for y in gates_list:
                x.add_child(matrix=np.dot(gates_list[y], x.matrix), gate=gates_list[y])
            for y in x.children:
                next_level.append(y)
        current_level = next_level
        depth += 1
    return current_best.gate
            


# Algorithm that returns a random gate
def random_algo(target_matrix, current_matrix, gates_list, trained_ai):
    return gates_list[random.choice(list(gates_list.keys()))]


# Standard Monte-Carlo Tree Search Algorithm
def monte_carlo_algo(target_matrix, current_matrix, gates_list, trained_ai):
    trained_ai.update(current_matrix)
    return monte_carlo.untuple_array(trained_ai.get_play())