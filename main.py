import gates
import learning_functions
import train_functions
import numpy as np


# General Setup
target_matrix = np.array([[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
current_matrix = np.identity(np.size(target_matrix, 0))
# Which algorithm to try to approximate the matrix with
learning_function = learning_functions.monte_carlo_algo
max_depth = 10
error_threshold = 0.1
trained_ai = 0

print('Target Matrix:')
print(target_matrix)

gates_list = gates.get_gates(int(np.log2(np.size(current_matrix, 0))), 'small')
if(learning_function == learning_functions.monte_carlo_algo):
    trained_ai = train_functions.train_monte_carlo(target_matrix, current_matrix, gates_list)


# Try to approximate the matrix
high_error = True
count = 0
while(high_error and count < max_depth):
    current_matrix = np.dot(learning_function(
        target_matrix, current_matrix, gates_list, trained_ai), current_matrix)
    if(np.linalg.norm(target_matrix-current_matrix) < error_threshold):
        high_error = False
    count += 1

out = 'Resulting Matrix:'
if(high_error):
    out += ' (Target Matrix is not well approximated)'
print(out)
print(current_matrix)
