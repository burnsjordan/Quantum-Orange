import gates
import numpy as np
import datetime
from math import log, sqrt
from random import choice

floating_point_error_tolerance = 0.001

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


    def get_best_child(self, C):
        current_best_ratio = -1
        current_best_child = 0
        total_plays = 0
        for x in self.children:
            total_plays += x.times_visited
        for x in self.children:
            if(x.times_won/x.times_visited > current_best_ratio):
                current_best_ratio = x.times_won/x.times_visited
                current_best_child = x
        return current_best_child


class Monte_Carlo_Tree():
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 4)
        self.error_threshold = kwargs.get('error', 0.01)
        self.target = tuple_array(kwargs.get(
            'target', np.array([[0, 1], [1, 0]])))
        temp = kwargs.get('list', gates.get_gates(self.N, 'small'))
        self.gates_list = []
        for x in temp:
            self.gates_list.append(tuple_array(temp[x]))
        self.root = Monte_Carlo_Node(matrix=kwargs.get('matrix'), gate=kwargs.get('gate'))
        self.visited_states = []
        self.visited_states.append(self.root)
        self.max_depth = kwargs.get('max_depth', 100)
        self.C = kwargs.get('C', 1.4)


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
        for x in self.gates_list:
            node.add_child(untuple_array(x), x)


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
            current_node = starting_node.get_best_child(1.4)
        else:
            current_node = self.get_random_node(unvisited_children)
        self.add_children(current_node)
        if(np.linalg.norm(untuple_array(self.target)-current_node.matrix) < self.error_threshold):
            backpropogate(current_node, "win")
        elif(current_node.depth == self.max_depth):
            backpropogate(current_node, "loss")
        else:
            self.play_round(current_node)


    def get_best_move(self, matrix):
        try:
            starting_node = self.get_equivalent_node(matrix)
        except:
            print('Invalid Matrix Given')
            print(matrix)
            print(self.visited_states)
            return
        if(not starting_node.check_children()):
            count = 0
            while(count < 100):
                self.play_round(starting_node)
                count += 1
        return starting_node.get_best_child(self.C)


class Circuit():
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 4)
        self.error_threshold = kwargs.get('error', 0.1)
        self.target = tuple_array(kwargs.get(
            'target', np.array([[0, 1], [1, 0]])))
        temp = kwargs.get('list', gates.get_gates(self.N, 'small'))
        self.gates_list = []
        for x in temp:
            self.gates_list.append(tuple_array(temp[x]))

    def start(self):
        return tuple_array(np.identity(2**self.N))

    def current_player(self, state):
        return 1

    def next_state(self, state, play):
        return tuple_array(np.dot(untuple_array(play), untuple_array(state)))

    def legal_plays(self, state_history):
        return self.gates_list

    def winner(self, state_history):
        if(np.linalg.norm(untuple_array(self.target)-untuple_array(state_history[-1])) < self.error_threshold):
            return 1
        else:
            return 2


class MonteCarlo():
    def __init__(self, circuit, **kwargs):
        self.circuit = circuit
        self.states = []
        seconds = kwargs.get('time', 5)
        self.verbose = kwargs.get('verbose', False)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 100)
        self.wins = {}
        self.plays = {}
        self.C = kwargs.get('C', 1.4)
        self.max_depth = 0

    def update(self, state):
        self.states.append(tuple_array(state))

    def get_play(self):
        self.max_depth = 0
        state = self.states[-1]
        player = self.circuit.current_player(state)
        legal = self.circuit.legal_plays(self.states[:])

        if(not legal):
            return
        if(len(legal) == 1):
            return legal[0]

        games = 0
        begin = datetime.datetime.utcnow()
        while(datetime.datetime.utcnow()-begin < self.calculation_time):
            self.run_simulation()
            games += 1

        moves_states = [(p, self.circuit.next_state(state, p)) for p in legal]

        if(self.verbose):
            print(games, datetime.datetime.utcnow() - begin)

        percent_wins, move = max(
            (self.wins.get((player, S), 0) /
             self.plays.get((player, S), 1),
             p)
            for p, S in moves_states
        )

        if(self.verbose):
            for x in sorted(((100 * self.wins.get((player, S), 0) / self.plays.get((player, S), 1), self.wins.get((player, S), 0), self.plays.get((player, S), 0), p) for p, S in moves_states), reverse=True):
                print("{3}: {0:.2f}% ({1} / {2})".format(*x))

            print('Maximum depth searched:', self.max_depth)

        return move

    def run_simulation(self):
        plays, wins = self.plays, self.wins

        visited_states = set()
        states_copy = self.states[:]
        state = states_copy[-1]
        player = self.circuit.current_player(states_copy)

        expand = True
        for t in range(1, self.max_moves + 1):
            legal = self.circuit.legal_plays(states_copy)
            moves_states = [(p, self.circuit.next_state(state, p))
                            for p in legal]

            if(all(plays.get((player, S)) for p, S in moves_states)):
                log_total = log(
                    sum(plays[(player, S)] for p, S in moves_states)
                )
                value, move, state = max(
                    ((wins[(player, S)] / plays[(player, S)] +
                      self.C * sqrt(log_total / plays[(player, S)]), p, S)
                     for p, S in moves_states)
                )
            else:
                move, state = choice(moves_states)

            states_copy.append(state)

            if(expand and (player, state) not in self.plays):
                expand = False
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0

            visited_states.add((player, state))

            player = self.circuit.current_player(state)
            winner = self.circuit.winner(states_copy)
            if(len(states_copy) > self.max_depth):
                self.max_depth = len(states_copy)
            if(winner):
                break

        for player, state in visited_states:
            if((player, state) not in self.plays):
                continue
            self.plays[(player, state)] += 1
            if(player == winner):
                self.wins[(player, state)] += 1
