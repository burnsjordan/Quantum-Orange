import monte_carlo
import numpy as np

def train_monte_carlo(target_matrix, current_matrix, gates_list, **kwargs):
    move_time = kwargs.get('time', 5)
    max_games = kwargs.get('games', 100)
    max_moves = kwargs.get('moves', 100)
    circuit = monte_carlo.Circuit(N=int(np.log2(np.size(current_matrix, 0))), target=target_matrix, list=gates_list)
    monte = monte_carlo.MonteCarlo(circuit, time=move_time)
    monte.update(monte_carlo.untuple_array(circuit.start()))
    for t in range(max_games):
        monte.run_simulation()
    return monte