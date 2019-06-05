import numpy as np
import random


# Greedy algorithm that calculates the norm for each possible gate and returns
# the best one
def greedy(target_matrix, current_matrix, gates_list):
    cost_list = {}
    for x in gates_list:
        cost_list[x] = np.linalg.norm(
            target_matrix-np.dot(gates_list[x], current_matrix))
    return min(cost_list, key=cost_list.get)


# Algorithm that returns a random gate
def random_algo(target_matrix, current_matrix, gates_list):
    return random.choice(list(gates_list.keys()))
