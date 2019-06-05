import numpy as np

# Small set of quantum gates
small_dict = {
    'Identity': np.array([[1, 0], [0, 1]]),
    'X': np.array([[0, 1], [1, 0]]),
    # 'CNOT': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    # 'TONC': np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    'Hadamard': (1/(2**(1/2)))*np.array([[1, 1], [1, -1]]),
    'Pi8': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
}


# Return all possible gates from a given list of size N
# TODO Make this work with two qubit gates
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
                if(j == 1):
                    temp_matrix = small_dict[i]
                else:
                    temp_matrix = small_dict['Identity']
                identity = small_dict['Identity']
                for k in range(2, N+1):
                    if(k == j):
                        temp_matrix = np.kron(temp_matrix, small_dict[i])
                    else:
                        temp_matrix = np.kron(temp_matrix, identity)
                list[count] = temp_matrix
                count += 1
    return(list)
