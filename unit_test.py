import pytest
import numpy as np
import learning_functions
import monte_carlo


@pytest.fixture
def supply_size_1_gates_list():
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    H = (1/(2**(1/2)))*np.array([[1, 1], [1, -1]])
    Pi8 = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
    gate_list = {
        'Identity': np.array([[1, 0], [0, 1]]),
        'X': np.array([[0, 1], [1, 0]]),
        'Hadamard': (1/(2**(1/2)))*np.array([[1, 1], [1, -1]]),
        'Pi8': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
    }
    return [I, X, H, Pi8, gate_list]


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


def test_learning_functions_greedy_algo(supply_size_1_gates_list):
    target = supply_size_1_gates_list[1]
    current = supply_size_1_gates_list[0]
    assert tuple_array(supply_size_1_gates_list[1]) == tuple_array(
        learning_functions.greedy_algo(target, current, supply_size_1_gates_list[4], 0))


def test_learning_functions_deep_greedy_algo(supply_size_1_gates_list):
    target = supply_size_1_gates_list[0]
    target = np.dot(supply_size_1_gates_list[1], target)
    target = np.dot(supply_size_1_gates_list[2], target)
    current = supply_size_1_gates_list[0]
    assert tuple_array(supply_size_1_gates_list[1]) == tuple_array(
        learning_functions.deep_greedy_algo(target, current, supply_size_1_gates_list[4], 0))
    current = np.dot(learning_functions.deep_greedy_algo(
        target, current, supply_size_1_gates_list[4], 0), current)
    assert tuple_array(supply_size_1_gates_list[2]) == tuple_array(
        learning_functions.deep_greedy_algo(target, current, supply_size_1_gates_list[4], 0))


def test_monte_carlo_tuple_array():
    arr = np.array([[1, 0], [0, 1]])
    assert monte_carlo.tuple_array(arr) == (2, 1, 0, 0, 1)


def test_monte_carlo_untuple_array():
    tup = (2, 1, 0, 0 ,1)
    assert monte_carlo.untuple_array(tup)[0][0] == 1
    assert monte_carlo.untuple_array(tup)[0][1] == 0
    assert monte_carlo.untuple_array(tup)[1][0] == 0
    assert monte_carlo.untuple_array(tup)[1][1] == 1
