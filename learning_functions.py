import numpy as np
import random
import monte_carlo


# Greedy algorithm that calculates the norm for each possible gate and returns
# the best one
def greedy(target_matrix, current_matrix, gates_list, trained_ai):
    cost_list = {}
    for x in gates_list:
        cost_list[x] = np.linalg.norm(
            target_matrix-np.dot(gates_list[x], current_matrix))
    return gates_list[min(cost_list, key=cost_list.get)]


# Algorithm that returns a random gate
def random_algo(target_matrix, current_matrix, gates_list, trained_ai):
    return gates_list[random.choice(list(gates_list.keys()))]


# Standard Monte-Carlo Tree Search Algorithm
def monte_carlo_algo(target_matrix, current_matrix, gates_list, trained_ai):
    trained_ai.update(current_matrix)
    return monte_carlo.untuple_array(trained_ai.get_play())