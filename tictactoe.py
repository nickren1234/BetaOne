import numpy as np


class Tictactoe:
    def __init__(self, id):
        self.state = State(id)
        self.turn = 1 # -1 -> o, 1 -> x
        self.input_dim = (3, 3, 3)
        self.policy_dim = 9
        self.name = 'tictactoe'

    def move(self, action, id):
        new_state = self.state.take_action(action, id)
        self.state = new_state
        self.turn = -self.turn

        return self.state, self.state.value, self.state.ended

    def convert_to_model_input(self, game_states):
        X_train = []
        Y_train = {'value_head': [],
                   'policy_head': []}
        for game in game_states:
            board, pi, turn, value = game.split(',')
            board_plane = np.zeros((2, 3, 3))
            for idx, char in enumerate(board):
                if char == '1':
                    board_plane[0][int(idx / 3)][idx % 3] = 1
                elif char == '2':
                    board_plane[1][int(idx / 3)][idx % 3] = -1
            turn_plane = np.full((1, 3, 3), int(turn))
            X_train.append(np.vstack([board_plane, turn_plane]))
            Y_train['value_head'].append(float(value))
            Y_train['policy_head'].append([float(val) for val in pi.split(' ')[1:]])
        Y_train['value_head'] = np.array(Y_train['value_head'])
        Y_train['policy_head'] = np.array(Y_train['policy_head'])
        return np.array(X_train), Y_train


class State:
    def __init__(self, id, board=None, turn=None):
        if board is None:
            self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            self.board = board

        if turn is None:
            self.turn = 1
        else:
            self.turn = turn

        self.winners = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6]
        ]

        self.id = id
        self.valid_moves = self._valid_moves()
        self.ended = self._check_game_end()
        self.value = self._get_value()

    def _check_game_end(self):
        if np.count_nonzero(self.board) == 9:
            return 1

        for x, y, z in self.winners:
            if self.board[x] + self.board[y] + self.board[z] == 3 * -self.turn:
                return 1
        return 0

    def _valid_moves(self):
        moves = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                moves.append(i)
        return moves

    def _get_value(self):
        # todo
        # This is the value of the state for the player who is about to move
        # which is the player represented by state.turn
        # i.e. if the previous player played a winning move, you lose
        for x, y, z in self.winners:
            if self.board[x] + self.board[y] + self.board[z] == 3 * -self.turn:
                return -1
        return 0

    def take_action(self, action, id):
        new_board = np.array(self.board)
        new_board[action] = self.turn

        new_state = State(id, new_board, -self.turn)

        return new_state

    def print_board(self):
        if self.turn == 1:
            print('x to move')
        else:
            print('o to move')
        to_print = [' ' if x == 0 else 'x' if x == 1 else 'o' for x in self.board]
        print(to_print[0] + '|' + to_print[1] + '|' + to_print[2])
        print('-+-+-')
        print(to_print[3] + '|' + to_print[4] + '|' + to_print[5])
        print('-+-+-')
        print(to_print[6] + '|' + to_print[7] + '|' + to_print[8])

    def convert_to_record(self):
        plain_text = ''
        for x in self.board:
            if x == -1:
                plain_text += '2'
            else:
                plain_text += str(x)
        plain_text += ','
        return plain_text

    def state_to_input(self):
        board_plane = np.zeros((2, 3, 3))
        for idx, val in enumerate(self.board):
            if val == 1:
                board_plane[0][int(idx / 3)][idx % 3] = 1
            elif val == -1:
                board_plane[1][int(idx / 3)][idx % 3] = 1
        turn_plane = np.full((1, 3, 3), self.turn)
        return np.vstack([board_plane, turn_plane])


