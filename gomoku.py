import numpy as np


class Gomoku:
    def __init__(self, id):
        self.state = State(id)
        self.turn = 1  # -1 -> o, 1 -> x
        self.input_dim = (3, 15, 15)
        self.policy_dim = 225
        self.name = 'gomoku'

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
            board_plane = np.zeros((2, 15, 15))
            for idx, char in enumerate(board):
                if char == '1':
                    board_plane[0][int(idx / 15)][idx % 15] = 1
                elif char == '2':
                    board_plane[1][int(idx / 15)][idx % 15] = -1
            turn_plane = np.full((1, 15, 15), int(turn))
            X_train.append(np.vstack([board_plane, turn_plane]))
            Y_train['value_head'].append(float(value))
            Y_train['policy_head'].append([float(val) for val in pi.split(' ')[1:]])
        Y_train['value_head'] = np.array(Y_train['value_head'])
        Y_train['policy_head'] = np.array(Y_train['policy_head'])
        return np.array(X_train), Y_train


class State:
    def __init__(self, id, board=None, turn=None):
        if board is None:
            self.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(15)]
        else:
            self.board = board

        if turn is None:
            self.turn = 1
        else:
            self.turn = turn

        self.id = id
        self.valid_moves = self._valid_moves()
        self.value = 0
        self.ended = 0

    def check_row(self, pos):
        connected = 0
        for i in range(max(0, pos[0] - 4), min(pos[0] + 5, 15)):
            if self.board[i][pos[1]] == -self.turn:
                connected += 1
                if connected == 5:
                    return True
            else:
                connected = 0
        return False

    def check_col(self, pos):
        connected = 0
        for i in range(max(0, pos[1] - 4), min(pos[1] + 5, 15)):
            if self.board[pos[0]][i] == -self.turn:
                connected += 1
                if connected == 5:
                    return True
            else:
                connected = 0
        return False

    def check_diag(self, pos):
        nw_range = min(4, pos[0], pos[1])
        ne_range = min(4, pos[0], 14 - pos[1])
        sw_range = min(4, 14 - pos[0], pos[1])
        se_range = min(4, 14 - pos[0], 14 - pos[1])

        connected = 0
        for i in range(-nw_range, se_range + 1):
            if self.board[pos[0] + i][pos[1] + i] == -self.turn:
                connected += 1
                if connected == 5:
                    return True
            else:
                connected = 0

        connected = 0
        for i in range(-ne_range, sw_range + 1):
            if self.board[pos[0] + i][pos[1] - i] == -self.turn:
                connected += 1
                if connected == 5:
                    return True
            else:
                connected = 0

        return False

    def _check_game_end(self, action):
        pos = [int(action / 15), action % 15]
        if np.count_nonzero(self.board) == 225:
            return 1

        # check if new stone forms a chain of 5 in all directions
        if self.check_row(pos) or self.check_col(pos) or self.check_diag(pos):
            self.value = -1
            return 1
        return 0

    def _valid_moves(self):
        moves = []
        for i in range(15):
            for j in range(15):
                if self.board[i][j] == 0:
                    moves.append(i * 15 + j)
        return moves

    def take_action(self, action, id):
        pos = [int(action / 15), action % 15]
        new_board = np.array(self.board)
        new_board[pos[0]][pos[1]] = self.turn

        new_state = State(id, new_board, -self.turn)
        new_state.ended = new_state._check_game_end(action)

        return new_state

    def print_board(self):
        if self.turn == 1:
            print('x to move')
        else:
            print('o to move')
        for i in range(15):
            to_print = ['o' if x == -1 else 'x' if x == 1 else '_' for x in self.board[i]]
            print(' '.join(to_print) + ' ' + str(i))
        print('0 1 2 3 4 5 6 7 8 9 A B C D E')

    def convert_to_record(self):
        plaintext = ''
        for i in range(15):
            to_print = ['1' if x == 1 else '2' if x == -1 else '0' for x in self.board[i]]
            plaintext += ''.join(to_print)
        plaintext += ','
        return plaintext

    def state_to_input(self):
        board_plane = np.zeros((2, 15, 15))
        for i in range(15):
            for j in range(15):
                if self.board[i][j] == 1:
                    board_plane[0][i][j] = 1
                elif self.board[i][j] == -1:
                    board_plane[1][i][j] = 1
        turn_plane = np.full((1, 15, 15), self.turn)
        return np.vstack([board_plane, turn_plane])
