import helpers
import learning_functions
import train_functions
import numpy as np


class Colors():
    HEADING = '\033[38;5;63m'
    PASSED = '\033[38;5;47m'
    FAILED = '\033[38;5;160m'
    ENDC = '\033[0m'


# General Setup
max_depth = 15
error_threshold = 0.1
target_matrix = np.identity(4)
current_matrix = np.identity(np.size(target_matrix, 0))
gates_list = gates.get_gates(int(np.log2(np.size(current_matrix, 0))), 'small')
for i in range(7):
    target_matrix = np.dot(learning_functions.random_algo(
        target_matrix, 0, gates_list, 0), target_matrix)

# The algorithm to try to approximate the matrix with
learning_function = learning_functions.deep_greedy_algo
trained_ai = 0
if(learning_function == learning_functions.monte_carlo_algo):
    trained_ai = train_functions.train_monte_carlo(
        target_matrix, current_matrix, gates_list)

print(Colors.HEADING + 'Target Matrix:' + Colors.ENDC)
print(target_matrix)

# Try to approximate the matrix
high_error = True
count = 0
while(high_error and count < max_depth):
    current_matrix = np.dot(learning_function(
        target_matrix, current_matrix, gates_list, trained_ai), current_matrix)
    if(np.linalg.norm(target_matrix-current_matrix) < error_threshold):
        high_error = False
    count += 1

print(Colors.HEADING + 'Resulting Matrix:' + Colors.ENDC)
print(current_matrix)
print(Colors.HEADING + 'Normed Difference:' + Colors.ENDC +
      str(np.linalg.norm(target_matrix-current_matrix)))
if(high_error):
    print(Colors.FAILED + 'Target Matrix is not well approximated' + Colors.ENDC)
