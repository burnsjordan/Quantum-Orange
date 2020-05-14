import helpers
import numpy as np
import datetime
from math import log, sqrt
from random import choice

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 

model = load_model('real_matrix1gates.h5')
nn = False

inputs_real = []
inputs_imag = []
input_r = np.asarray(inputs_real)
input_i = np.asarray(inputs_imag)

floating_point_error_tolerance = 0.00001

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


def backpropogate(node, result):
    if(result == "win"):
        node.times_visited += 1
        node.times_won += 1
    else:
        node.times_visited +=1
    if(node.parent != 0):
        backpropogate(node.parent, result)


class Node():
    def __init__(self, **kwargs):
        self.depth = kwargs.get('depth', 0)
        self.parent = kwargs.get('parent', 0)
        self.children = []
        self.matrix = kwargs.get('matrix', gates.Identity)
        self.gate = kwargs.get('gate', gates.Identity)
        self.root = kwargs.get('root', self)

    def add_child(self, matrix, gate):
        self.children.append(
            Node(depth=self.depth+1, parent=self, matrix=matrix, gate=gate, root=self.root))


class Tree():
    def __init__(self, **kwargs):
        self.root = Node(matrix=kwargs.get('matrix'), gate=kwargs.get('gate'))


class Monte_Carlo_Node(Node):
    def __init__(self, **kwargs):
        Node.__init__(self, **kwargs)
        self.times_visited = 0
        self.times_won = 0


    def add_child(self, matrix, gate):
        self.children.append(
            Monte_Carlo_Node(depth=self.depth+1, parent=self, matrix=matrix, gate=gate, root=self.root))

    
    def check_children(self):
        temp = True
        if(not self.children):
            temp = False
        for x in self.children:
            if(x.times_visited == 0):
                temp = False
        return temp


    def get_best_child(self):
        current_best_ratio = -1
        current_best_child = 0
        current_most_times_won = 0
        total_plays = 0
        for x in self.children:
            total_plays += x.times_visited
        for x in self.children:
            if(x.times_visited > 0):
                ratio = (x.times_won/x.times_visited)
            else:
                ratio = 0
            if(ratio > current_best_ratio):
                current_best_ratio = ratio
                current_best_child = x
                current_most_times_won = x.times_won
            elif (ratio == current_best_ratio):
                if (x.times_won > current_most_times_won):
                    current_best_ratio = ratio
                    current_best_child = x
                    current_most_times_won = x.times_won 
        return current_best_child

    def get_best_monte_carlo_candidate(self, C):
        current_best_ratio = -1
        current_best_child = 0
        total_plays = 0
        for x in self.children:
            total_plays += x.times_visited
        for x in self.children:
            ratio = (x.times_won/x.times_visited)+(C*sqrt(log(total_plays)/x.times_visited))
            if(ratio > current_best_ratio):
                current_best_ratio = ratio
                current_best_child = x
        return current_best_child


def predict(input):
    temp = model.predict(input)
    K.clear_session()
    return temp


def get_nn_node(tree, node_list, current):
    # inputs_real.append(current.matrix.real)
    # inputs_imag.append(current.matrix.imag)
    # prediction = model.predict(np.stack([np.asarray(inputs_real), np.asarray(inputs_imag)], 3))
    # prediction = model.predict(np.array([current.matrix.real]))
    prediction = predict(np.array([current.matrix.real]))
    current_best = prediction[0][0]
    current_best_index = 0
    for i in range(len(prediction[0])):
        if(prediction[0][i] > current_best):
            current_best = prediction[0][i]
            current_best_index = i - 1
    for x in node_list:
        if(tuple_array(x.matrix) == tree.gates_list[current_best_index]):
            return x
    return choice(node_list)

class Monte_Carlo_Tree():
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 4)
        self.error_threshold = kwargs.get('error', 0.00001)
        self.target = tuple_array(kwargs.get(
            'target', np.array([[0, 1], [1, 0]])))
        temp = kwargs.get('list', gates.get_gates(self.N, 'small'))
        self.gates_list = []
        for x in temp:
            self.gates_list.append(tuple_array(temp[x]))
        self.root = Monte_Carlo_Node(matrix=kwargs.get('matrix'), gate=kwargs.get('gate'))
        self.visited_states = []
        self.visited_states.append(self.root)
        self.max_depth = kwargs.get('max_depth', 10)
        self.C = kwargs.get('C', 1.4)
        self.notDone = True


    def get_equivalent_node(self, matrix):
        for x in self.visited_states:
            if(np.linalg.norm(matrix-x.matrix) < floating_point_error_tolerance):
                return x
        raise ValueError
        

    def get_random_node(self, node_list):
        return choice(node_list)


    def add_children(self, node):
        if(node.children):
            return
        identity = np.identity(2**self.N)
        for x in self.gates_list:
            add = True
            if(node.parent):
                if(tuple_array(np.matmul(node.matrix,untuple_array(x))) == tuple_array(node.parent.matrix)):
                    add = False
            if(tuple_array(np.matmul(node.matrix,untuple_array(x))) == tuple_array(identity)):
                add = False
            if(add):
                node.add_child(np.matmul(node.matrix,untuple_array(x)), x)


    def update(self, node):
        self.visited_states.append(node)


    # TODO Fix C in get_best_child
    def play_round(self, starting_node):
        self.add_children(starting_node)
        visited_children = []
        unvisited_children = []

        for x in starting_node.children:
            if(x.times_visited < 1):
                unvisited_children.append(x)
            else:
                visited_children.append(x)
        if(not unvisited_children):
            current_node = starting_node.get_best_monte_carlo_candidate(1.4)
        else:
            if(nn):
                current_node = get_nn_node(self, unvisited_children, starting_node)
            else:
                current_node = self.get_random_node(unvisited_children)
        self.add_children(current_node)
        if(current_node.times_visited == 0 and False):
            self.visited_states.append(current_node)
        if(np.linalg.norm(untuple_array(self.target)-current_node.matrix) < self.error_threshold):
            backpropogate(current_node, "win")
            self.notDone = False
        elif(current_node.depth == self.max_depth):
            backpropogate(current_node, "loss")
        else:
            self.play_round(current_node)


    def get_best_move(self, matrix):
        #print(matrix)
        try:
            starting_node = self.get_equivalent_node(matrix)
        except:
            print('Invalid Matrix Given')
            print(matrix)
            print(self.visited_states)
            return
        if(not starting_node.check_children()):
            count = 0
            while(count < 500 and self.notDone):
                self.play_round(starting_node)
                count += 1
        return starting_node.get_best_child()
