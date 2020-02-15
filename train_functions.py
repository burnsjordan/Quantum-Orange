import monte_carlo
import monte_carlo_nn
import numpy as np


# Pre-train a monte carlo search tree
def train_monte_carlo(target_matrix, current_matrix, gates_list, **kwargs):
    move_time = kwargs.get('time', 5)
    max_games = kwargs.get('games', 100)
    monte = monte_carlo.Monte_Carlo_Tree(N=int(
        np.log2(np.size(current_matrix, 0))), target=target_matrix, list=gates_list, time=move_time)
    circuit = monte_carlo.Circuit(N=int(
        np.log2(np.size(current_matrix, 0))), target=target_matrix, list=gates_list)
    monte = monte_carlo.MonteCarlo(circuit, time=move_time)
    monte.update(monte_carlo.untuple_array(circuit.start()))
    count = 0
    while(count < max_games):
        monte.run_simulation()
        count += 1
    return monte


# Create new monte carlo search tree
def no_train_monte_carlo(target_matrix, current_matrix, gates_list, **kwargs):
    move_time = kwargs.get('time', 5)
    monte = monte_carlo.Monte_Carlo_Tree(N=int(
        np.log2(np.size(current_matrix, 0))), target=target_matrix, list=gates_list, time=move_time, matrix=current_matrix)
    return monte


# Create new monte carlo search tree
def no_train_monte_carlo_nn(target_matrix, current_matrix, gates_list, **kwargs):
    move_time = kwargs.get('time', 5)
    monte = monte_carlo_nn.Monte_Carlo_Tree(N=int(
        np.log2(np.size(current_matrix, 0))), target=target_matrix, list=gates_list, time=move_time, matrix=current_matrix)
    return monte
