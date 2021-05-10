import numpy as np
import random

from mcts import Node, Edge, MCTS

import config

import matplotlib.pyplot as plt

def print_leaf(leaf):
    output = f'{leaf.state.board[0]} {leaf.state.board[1]} {leaf.state.board[2]} \n'
    output += f'{leaf.state.board[3]} {leaf.state.board[4]} {leaf.state.board[5]} \n'
    output += f'{leaf.state.board[6]} {leaf.state.board[7]} {leaf.state.board[8]} \n'
    return output


def print_mcts(leaf, value, done, path):
    with open('log.txt', 'a+') as f:
        f.write(print_leaf(leaf))
        f.write(f'value {value}, done {done}, turn {leaf.turn} \n')
        f.write(f'id: {leaf.id} \n')
        for edge in path:
            f.write(f'in node id: {edge.in_node.id} \n')
            f.write(f'N {edge.stats["N"]} W {edge.stats["W"]} Q {edge.stats["Q"]} \n')


class Player():
    def __init__(self, name):
        self.name = name

    def action(self):
        '''
        this is just for tictactoe for now
        will work on human players later, i guess
        '''
        action = input('Enter your move: ')
        return action # will need to decide what form this should be


class Agent():
    def __init__(self, name, input_dim, policy_dim, mcts_sims, cpuct, model):
        self.name = name
        self.cpuct = cpuct
        self.input_dim = input_dim
        self.policy_dim = policy_dim

        self.MCTSsimulations = mcts_sims
        self.model = model

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self, id):
        # TODO

        leaf, value, done, path, id = self.mcts.move_to_leaf(id)

        value, id = self.evaluate_leaf(leaf, value, done, id)

        self.mcts.backFill(leaf, value, path)

        # print_mcts(leaf, value, done, path)

        return id

    def act(self, state, tau, id):
        # TODO
        if self.mcts is None or state.id not in self.mcts.tree.keys():
            self.build_MCTS(state)
        else:
            self.change_root(state)

        # run the simulation to build the tree
        for _ in range(self.MCTSsimulations):
            id = self.simulate(id)

        # get action values
        pi, values = self.get_policy(1) # this should be tau instead of 1??
        # okay so this tau does not look like the temp

        ####pick the action
        action, value = self.choose_action(pi, values, tau)

        id += 1
        nextState = state.take_action(action, id)

        NN_value = -self.get_preds(nextState)[0]

        return action, pi, value, NN_value, id

    def get_preds(self, state):
        # predict the leaf
        input_to_model = np.array([state.state_to_input()])

        preds = self.model.predict(input_to_model)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]

        logits = logits_array[0]
        # logits is the policy??
        # interesting but why not make this a layer in the policy head, just do the softmax there lol
        allowed_actions = state.valid_moves

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowed_actions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        policy = odds / np.sum(odds)  ###put this just before the for?

        return value, policy, allowed_actions

    def evaluate_leaf(self, leaf, value, done, id):
        # TODO
        # if game is not done, add all allowed actions as new leaf nodes
        # if game is done, return the result
        if done == 0:

            value, policy, allowed_actions = self.get_preds(leaf.state)

            policy = policy[allowed_actions]

            for idx, action in enumerate(allowed_actions):
                id += 1
                new_state = leaf.state.take_action(action, id)
                if new_state.id not in self.mcts.tree:
                    node = Node(new_state)
                    self.mcts.insert(node)
                else:
                    node = self.mcts.tree[new_state.id]

                new_edge = Edge(leaf, node, policy[idx], action)
                leaf.edges.append((action, new_edge))

        return value, id

    def get_policy(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.policy_dim, dtype=np.integer)
        values = np.zeros(self.policy_dim, dtype=np.float32)

        for action, edge in edges:
            # tau should be 1 at the beginning and then dropped to very small values
            # with low values of tau, the tree will only play the most visited move
            # this makes the tree search less moves
            # thus lowering performance for improved runtime
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        # then normalize to a probability distribution
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def choose_action(self, pi, values, tau):
        if tau == 0:
            # pick the max
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def build_training_set(self, records):
        # so randomly sample some number of games from the file
        # and then convert them to model input
        game_states = []
        target_size = config.TRAINING_SIZE
        for version in range(records.version, max(-1, records.version-1), -1):
            if target_size > records.meta[version]['total']:
                target_size -= records.meta[version]['total']
                for game in records.meta[version].keys():
                    if game != 'total':
                        with open(f'{config.RECORDS}/version{version}/game{game}.txt', 'r+') as f:
                            game_states.extend(f.read().splitlines())
            else:
                for game in records.meta[version].keys():
                    if game != 'total':
                        if target_size > records.meta[version][game]:
                            target_size -= records.meta[version][game]
                            with open(f'{config.RECORDS}/version{version}/game{game}.txt', 'r+') as f:
                                game_states.extend(f.readlines())
        return game_states

    def train(self, game_states, game):
        # TODO

        X_train, Y_train = game.convert_to_model_input(game_states)
        for i in range(config.TRAINING_LOOPS):
            # minibatch = random.sample(training_set, min(config.BATCH_SIZE, len(ltmemory)))
            # we build training set before calling this function
            '''
            training_states = np.array([self.model.convertToModelInput(row['state']) for row in training_set])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
                , 'policy_head': np.array([row['AV'] for row in minibatch])}
            '''
            fit = self.model.fit(X_train, Y_train, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))
        '''
        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')

        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
        plt.show()
        '''

    def predict(self, input_to_model):
        # TODO
        preds = self.model.predict(input_to_model)
        return preds

    def build_MCTS(self, state):
        root = Node(state)
        self.mcts = MCTS(root, self.cpuct)

    def change_root(self, state):
        self.mcts.root = self.mcts.tree[state.id]