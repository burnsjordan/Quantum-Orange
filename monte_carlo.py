import gates
import numpy as np
import datetime
from math import log, sqrt
from random import choice


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


class Node():
    def __init__(self, **kwargs):
        self.depth = kwargs.get('depth', 0)
        self.parent = kwargs.get('parent', 0)
        self.children = []
        self.matrix = kwargs.get('matrix', gates.Identity)
        self.gate = kwargs.get('gate', gates.Identity)
        self.root = kwargs.get('root', self)

    def add_child(self, matrix, gate):
        self.children.append(Node(depth=self.depth+1, parent=self, matrix=matrix, gate=gate, root=self.root))


class Tree():
    def __init__(self, **kwargs):
        self.root = Node(matrix=kwargs.get('matrix'), gate=kwargs.get('gate'))


class Circuit():
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 4)
        self.error_threshold = kwargs.get('error', 0.1)
        self.target = tuple_array(kwargs.get('target', np.array([[0, 1], [1, 0]])))
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