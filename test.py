import numpy as np
import helpers
import learning_functions
import train_functions
from scipy.stats import unitary_group
# import argparse


# # Define the parser
# parser = argparse.ArgumentParser(description='Short sample app')
# # Declare an argument (`--algo`), telling that the corresponding value should be stored in the `algo` field, and using a default value if the argument isn't given
# parser.add_argument('--foo', action="store_false")
# parser.add_argument('--model_name', action="store", dest='model_name', default='nntest.h5')
# # Now, parse the command line arguments and store the values in the `args` variable
# args = parser.parse_args()

# print(args)


class Colors():
    HEADING = '\033[38;5;63m'
    PASSED = '\033[38;5;47m'
    FAILED = '\033[38;5;160m'
    ENDC = '\033[0m'


# Setup test matrices
# Test using an identity matrix
i_test = np.identity(4)
current_matrix = np.identity(np.size(i_test, 0))
gates_list = gates.get_gates(int(np.log2(np.size(current_matrix, 0))), 'small')
max_depth = 5
error_threshold = 0.1
# Test using one gate from the gates list
one_gate_test = np.matmul(learning_functions.random_algo(
    i_test, 0, gates_list, 0), i_test)
# Test using five gates from the gates list
five_gate_test = i_test
for i in range(5):
    five_gate_test = np.matmul(learning_functions.random_algo(
        five_gate_test, 0, gates_list, 0), five_gate_test)
# Test using random unitary gate
unitary_test = unitary_group.rvs(2**int(np.log2(np.size(current_matrix, 0))))


# Attempt to approximate the given matrix
def test_algo(target_matrix, current_matrix, trained_ai, learning_function, temp_str, verbose=False):
    high_error = True
    count = 0
    rolling_matrix = current_matrix
    while(high_error and count < max_depth):
        if(verbose):
            print(count)
        #rolling_matrix = learning_function(target_matrix, rolling_matrix, gates_list, trained_ai)
        rolling_matrix = np.matmul(learning_function(
            target_matrix, rolling_matrix, gates_list, trained_ai), rolling_matrix)
        if(np.linalg.norm(target_matrix-rolling_matrix) < error_threshold):
            high_error = False
        count += 1
    if(high_error):
        print(temp_str + Colors.FAILED + 'Failed' + Colors.ENDC)
    else:
        print(temp_str + Colors.PASSED + 'Passed' + Colors.ENDC)


# Attempt to approximate the given matrix
def test_algo_monte(target_matrix, current_matrix, trained_ai, learning_function, temp_str, verbose=False):
    high_error = True
    count = 0
    rolling_matrix = current_matrix
    while(high_error and count < max_depth):
        if(verbose):
            print(count)
        rolling_matrix = learning_function(target_matrix, rolling_matrix, gates_list, trained_ai)
        #rolling_matrix = np.matmul(learning_function(
            #target_matrix, rolling_matrix, gates_list, trained_ai), rolling_matrix)
        if(np.linalg.norm(target_matrix-rolling_matrix) < error_threshold):
            high_error = False
        count += 1
    if(high_error):
        print(temp_str + Colors.FAILED + 'Failed' + Colors.ENDC)
        return False
    else:
        print(temp_str + Colors.PASSED + 'Passed' + Colors.ENDC)
        return True


# Test Random Algorithm
print('')
print(Colors.HEADING + 'Random Algorithm:' + Colors.ENDC)
test_algo(i_test, i_test, 0, learning_functions.random_algo, 'i_test: ')
test_algo(one_gate_test, i_test, 0,
          learning_functions.random_algo, 'one_gate_test: ')
test_algo(five_gate_test, i_test, 0,
          learning_functions.random_algo, 'five_gate_test: ')
test_algo(unitary_test, i_test, 0,
          learning_functions.random_algo, 'unitary_test: ')


# Test Greedy Algorithm
print('')
print(Colors.HEADING + 'Greedy Algorithm' + Colors.ENDC)
test_algo(i_test, i_test, 0, learning_functions.greedy_algo, 'i_test: ')
test_algo(one_gate_test, i_test, 0,
          learning_functions.greedy_algo, 'one_gate_test: ')
test_algo(five_gate_test, i_test, 0,
          learning_functions.greedy_algo, 'five_gate_test: ')
# test_algo(unitary_test, i_test, 0,
#           learning_functions.greedy_algo, 'unitary_test: ')


# Test Deep Greedy Algorithm
print('')
print(Colors.HEADING + 'Deep Greedy Algorithm' + Colors.ENDC)
test_algo(i_test, i_test, 0, learning_functions.deep_greedy_algo, 'i_test: ')
test_algo(one_gate_test, i_test, 0,
          learning_functions.deep_greedy_algo, 'one_gate_test: ')
test_algo(five_gate_test, i_test, 0,
          learning_functions.deep_greedy_algo, 'five_gate_test: ')
# test_algo(unitary_test, i_test, 0,
#         learning_functions.deep_greedy_algo, 'unitary_test: ')


# Test Monte Carlo Algorithm
print('')
print(Colors.HEADING + 'Monte Carlo Algorithm' + Colors.ENDC)
test_algo_monte(i_test, i_test, train_functions.no_train_monte_carlo(
    i_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'i_test: ')
test_algo_monte(one_gate_test, i_test, train_functions.no_train_monte_carlo(
    one_gate_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'one_gate_test: ')
test_algo_monte(five_gate_test, i_test, train_functions.no_train_monte_carlo(
    five_gate_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'five_gate_test: ')
# test_algo_monte(unitary_test, i_test, train_functions.no_train_monte_carlo(
#     unitary_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'unitary_test: ')

print("monte")
successes = 0
for i in range(100):
    five_gate_test = i_test
    for j in range(3):
        five_gate_test = np.matmul(learning_functions.random_algo(
            five_gate_test, 0, gates_list, 0), five_gate_test)
    if(test_algo_monte(five_gate_test, i_test, train_functions.no_train_monte_carlo(
        five_gate_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'five_gate_test: ')):
        successes = successes + 1
print(successes)


print("nn")
successes = 0
for i in range(100):
    five_gate_test = i_test
    for j in range(3):
        five_gate_test = np.matmul(learning_functions.random_algo(
            five_gate_test, 0, gates_list, 0), five_gate_test)
    if(test_algo_monte(five_gate_test, i_test, train_functions.no_train_monte_carlo_nn(
        five_gate_test, i_test, gates_list), learning_functions.monte_carlo_algo, 'five_gate_test: ')):
        successes = successes + 1
print(successes)
