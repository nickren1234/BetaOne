import numpy as np


class Tictactoe:
    def __init__(self, id):
        self.state = TState(id)
        self.turn = 1 # -1 -> o, 1 -> x
        self.input_dim = (3, 3, 3)
        self.policy_dim = 9
        self.name = 'tictactoe'

    def move(self, action, id):
        new_state = self.state.take_action(action, id)
        self.state = new_state
        self.turn = -self.turn

        return self.state, self.state.value, self.state.ended


class TState:
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
        # self.binary = self._binary()
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

    def _state_to_id(self):
        # this is not used :p
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))
        return id

    def _get_value(self):
        # todo
        # This is the value of the state for the player who is about to move
        # which is the player represented by state.turn
        # i.e. if the previous player played a winning move, you lose
        for x, y, z in self.winners:
            if self.board[x] + self.board[y] + self.board[z] == 3 * -self.turn:
                return 1
        return 0

    def allowed_actions(self):
        actions = []
        for idx, val in enumerate(self.board):
            if val == 0:
                actions.append(idx)
        return actions

    def take_action(self, action, id):
        new_board = np.array(self.board)
        new_board[action] = self.turn

        new_state = TState(id, new_board, -self.turn)

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

