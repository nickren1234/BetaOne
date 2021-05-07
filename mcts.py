import numpy as np
import config


class Node():
    def __init__(self, state):
        self.state = state
        self.turn = state.turn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0


class Edge():
    def __init__(self, in_node, out_node, prior, action):
        self.id = (in_node.id, out_node.id)
        self.in_node = in_node
        self.out_node = out_node
        self.turn = in_node.turn # ?
        self.action = action
        self.stats = {'N': 0,
                      'W': 0,
                      'Q': 0,
                      'P': prior}


class MCTS():
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.insert(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self, id):
        # plays out until leaf node is reached, 1 path only
        path = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.is_leaf():

            maxQU = -99999

            if current_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            Nb = 0
            for action, edge in current_node.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(current_node.edges):

                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                # take the "best" path with epsilon to introduce randomness
                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            id += 1
            new_state = current_node.state.take_action(simulationAction, id)
            value = new_state.value
            done = new_state.ended
            current_node = simulationEdge.out_node
            path.append(simulationEdge)

        return current_node, value, done, path, id

    def backFill(self, leaf, value, path):
        # so a game is over on the leaf.turn
        # then if the value is one, that player have played the winning move
        # so if the turn is the same as the leaf.turn we add 1 to W
        # otherwise subtract one to signal this move lead to a lose
        cur_turn = leaf.state.turn

        for edge in path:
            edge.stats['N'] += 1
            if edge.turn == cur_turn:
                edge.stats['W'] += value
            else:
                edge.stats['W'] -= value

            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def insert(self, node):
        self.tree[node.id] = node

