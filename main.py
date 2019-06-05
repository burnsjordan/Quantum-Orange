import gates
import learning_functions
import numpy as np


# General Setup
target_matrix = np.array([[0, 1], [1, 0]])
current_matrix = np.array([[1, 0], [0, 1]])
# Which algorithm to try to approximate the matrix with
learning_function = learning_functions.greedy
max_depth = 100
error_threshold = 0.1

gates_list = gates.get_gates(int(np.log2(np.size(current_matrix, 0))), 'small')


# Try to approximate the matrix
high_error = True
count = 0
while(high_error and count < max_depth):
    current_matrix = np.dot(gates_list[learning_function(
        target_matrix, current_matrix, gates_list)], current_matrix)
    if(np.linalg.norm(target_matrix-current_matrix) < error_threshold):
        high_error = False
    count += 1

if(high_error):
    print('Matrix is not well approximated.')
print(current_matrix)
