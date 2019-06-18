import numpy as np
import gates
import learning_functions
import train_functions

# Setup test matrices
i_test = np.identity(4)
current_matrix = np.identity(np.size(i_test, 0))
gates_list = gates.get_gates(int(np.log2(np.size(current_matrix, 0))), 'small')
max_depth = 10
error_threshold = 0.1

def test_algo(target_matrix, current_matrix, trained_ai, learning_function, temp_str):
    high_error = True
    count = 0
    while(high_error and count < max_depth):
        current_matrix = np.dot(learning_function(
            target_matrix, current_matrix, gates_list, trained_ai), current_matrix)
        if(np.linalg.norm(target_matrix-current_matrix) < error_threshold):
            high_error = False
        count += 1
    if(high_error):
        print(temp_str + 'Failed')
    else:
        print(temp_str + 'Passed')

# Test Random Algorithm
print('')
print('Testing Random Algorithm:')
test_algo(i_test, i_test, 0, learning_functions.random_algo, 'i_test: ')


# Test Greedy Algorithm
print('')
print('Testing Greedy Algorithm')
test_algo(i_test, i_test, 0, learning_functions.greedy_algo, 'i_test: ')


# Test Deep Greedy Algorithm
print('')
print('Testing Deep Greedy Algorithm')
test_algo(i_test, i_test, 0, learning_functions.deep_greedy_algo, 'i_test: ')


# Test Monte Carlo Algorithm
print('')
print('Testing Monte Carlo Algorithm')
test_algo(i_test, i_test, train_functions.train_monte_carlo(i_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'i_test: ')
