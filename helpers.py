
import numpy as np
import random

Identity = np.array([[1, 0], [0, 1]])

def tuple_array(arr):
    temp = []
    temp.append(np.size(arr, 0))
    for i in range(np.size(arr, 0)):
        for j in range(np.size(arr, 1)):
            temp.append(arr[i][j])
    return tuple(temp)


def untuple_array(arr):
    temp = []
    N = arr[0]
    for i in range(N):
        temp2 = []
        for j in range(N):
            temp2.append(arr[N*i+j+1])
        temp.append(temp2)
    return np.array(temp)

# Small set of quantum gates
small_dict = {
    'Identity': np.array([[1, 0], [0, 1]]),
    'X': np.array([[0, 1], [1, 0]]),
    'CNOT': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    'TONC': np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    'Hadamard': (1/(2**(1/2)))*np.array([[1, 1], [1, -1]]),
    'Pi8': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
}


# Return all possible gates from a given list of size N
def get_gates(N, list_size):
    list = {}
    if(N == 1):
        count = 0
        list[1] = small_dict['Identity']
        list[2] = small_dict['X']
        list[3] = small_dict['Hadamard']
        list[4] = small_dict['Pi8']
    else:
        count = 0
        for i in small_dict:
            for j in range(1, N+1):
                marker = 0
                if(j == 1):
                    temp_matrix = small_dict[i]
                    if(i == 'CNOT' or i == 'TONC'):
                        marker = 1
                else:
                    temp_matrix = Identity
                identity = Identity
                k = 2 + marker
                double = False
                while(k < N+1):
                    if(k == j):
                        if(k == N and (i == 'CNOT' or i == 'TONC')):
                            double = True
                        else:
                            temp_matrix = np.kron(temp_matrix, small_dict[i])
                        if(i == 'CNOT' or i == 'TONC'):
                            k += 1
                    else:
                        temp_matrix = np.kron(temp_matrix, identity)
                    k += 1
                if(j!=1 and i=='Identity'):
                    double = True
                if(not double):
                    list[count] = temp_matrix
                    count += 1
    return(list)


# Produce training data for a 4x4 matrix
# Returns an array which has the array of inputs as the first element
# and the array of outputs as the second element
def get_training_data(N):
    # Setup to build training data
    inputs = []
    outputs = []
    gates_list = get_gates(N, "small")
    identity_hit_count = 0
    inverse_hit_count = 0

    # Build training data
    # i*j is the number of data points
    for i in range(200):
        input_matrix = np.kron(Identity, Identity)
        for j in range(50):
            identity = np.identity(2**N)
            dont_add = True
            while(dont_add):
                dont_add = False
                temp = gates_list[random.choice(list(gates_list.keys()))]
                if(tuple_array(temp) == tuple_array(identity)):
                    dont_add = True
                    identity_hit_count += 1
                if(outputs):
                    if(tuple_array(np.matmul(outputs[-1], temp)) == tuple_array(identity)):
                        dont_add = True
                        inverse_hit_count += 1
                if(not dont_add):
                    input_matrix = np.dot(input_matrix, temp)
                    inputs.append(input_matrix)
                    outputs.append(temp)

    # Print the first 10 inputs and outputs if you want
    # for i in range(10):
    #     print(str(i+1) + ':')
    #     print(inputs[i])
    #     print(outputs[i])

    # Print number of times the identity or an inverse is chosen and ignored
    # print("identity_hit_count: " + str(identity_hit_count))
    # print("inverse_hit_count: " + str(inverse_hit_count))

    return [inputs, outputs]
