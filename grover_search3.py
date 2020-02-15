import numpy as np
import gates
import learning_functions
import train_functions
from scipy.stats import unitary_group
import gc


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
max_depth = 7
error_threshold = 0.1
sgates = gates.small_dict
grover = np.identity(4)
temp = np.kron(sgates['Hadamard'], sgates['Hadamard'])
grover = np.matmul(grover, temp)
temp = np.kron(sgates['X'], sgates['X'])
grover = np.matmul(grover, temp)
temp = np.kron(sgates['Identity'], sgates['Hadamard'])
grover = np.matmul(grover, temp)
temp = sgates['CNOT']
grover = np.matmul(grover, temp)
temp = np.kron(sgates['Identity'], sgates['Hadamard'])
grover = np.matmul(grover, temp)
temp = np.kron(sgates['X'], sgates['X'])
grover = np.matmul(grover, temp)
temp = np.kron(sgates['Hadamard'], sgates['Hadamard'])
grover = np.matmul(grover, temp)

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
    print('Error: ' + str(np.linalg.norm(target_matrix-rolling_matrix)))

success = 0
total = 0
# Test Monte Carlo Algorithm
print('')
print(Colors.HEADING + '2 qubit Grover Test' + Colors.ENDC)
while(total < 25):
    if(test_algo_monte(grover, i_test, train_functions.no_train_monte_carlo(
        grover, i_test, gates_list), learning_functions.monte_carlo_algo, "Grover's Test: ")):
        success = success + 1
    total = total + 1
    gc.collect()

print("Successes: " + str(success))
print("Total: " + str(total))

print('nn')
success = 0
total = 0
# Test Monte Carlo Algorithm
print('')
print(Colors.HEADING + '2 qubit Grover Test' + Colors.ENDC)
while(total < 100):
    if(test_algo_monte(grover, i_test, train_functions.no_train_monte_carlo_nn(
        grover, i_test, gates_list), learning_functions.monte_carlo_algo, "Grover's Test: ")):
        success = success + 1
    total = total + 1
    gc.collect()

print("Successes: " + str(success))
print("Total: " + str(total))