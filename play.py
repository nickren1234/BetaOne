import random
# from tictactoe import Tictactoe
from gomoku import Gomoku


def play_one_game(player1, player2, turns_until_tau0, records, id):
    id += 1
    # game = Tictactoe(id)
    game = Gomoku(id)
    state = game.state
    done = 0
    turn = 0
    player1.mcts = None
    player2.mcts = None

    while done == 0:
        turn = turn + 1

        if turn < turns_until_tau0:
            if state.turn == 1:
                action, pi, MCTS_value, NN_value, id = player1.act(state, 1, id)
            else:
                action, pi, MCTS_value, NN_value, id = player2.act(state, 1, id)
        else:
            if state.turn == 1:
                action, pi, MCTS_value, NN_value, id = player1.act(state, 0, id)
            else:
                action, pi, MCTS_value, NN_value, id = player2.act(state, 0, id)

        if records is not None:
            records.append(state, pi)

        id += 1
        state, value, done = game.move(action, id)
    if records is not None:
        # If the game is finished, assign the values correctly to the game moves
        for idx, move in enumerate(records.records):
            if int(move[-1]) == max(0, state.turn):
                records.records[idx] += ',' + str(-value)
            else:
                records.records[idx] += ',' + str(value)
        # if value is 0, game have not finished or is a draw
        # if value is 1, then the player who made the last move have won
        # notice that you cannot make a move and then lose, you only lose when your opponent moves
    return state, value, records, id


def play_matches(player1, player2, EPISODES, turns_until_tau0, id, records=None):
    # TODO
    scores = {player1.name: 0, "drawn": 0, player2.name: 0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}
    p1_goes_first = random.randint(0, 1)
    for e in range(EPISODES):
        p1_goes_first = 1 - p1_goes_first
        if p1_goes_first:
            state, value, records, id = play_one_game(player1, player2, turns_until_tau0, records, id)
        else:
            state, value, records, id = play_one_game(player2, player1, turns_until_tau0, records, id)

        print(f'player1 goes first? {p1_goes_first}')

        if records is not None:
            records.store_to_file(e)

        if value != 0:
            print(f'game {e} finished, winner is {-state.turn}')
        else:
            print(f'game {e} finished, game was drawn')
        state.print_board()

        if value == 1: # dont think this is ever 1, maybe delete later
            print("WOAH WTF IS GOING ON")

        elif value == -1:
            if state.turn == 1:
                if p1_goes_first:
                    scores[player2.name] += 1
                else:
                    scores[player1.name] += 1
                sp_scores['nsp'] = sp_scores['nsp'] + 1
            else:
                if p1_goes_first:
                    scores[player1.name] += 1
                else:
                    scores[player2.name] += 1
                sp_scores['sp'] = sp_scores['sp'] + 1

        else:
            # logger.info('DRAW...')
            scores['drawn'] = scores['drawn'] + 1
            sp_scores['drawn'] = sp_scores['drawn'] + 1

            # pts = state.score
        if state.turn == 1:
            points[player1.name].append(value)
            points[player2.name].append(-value)

    return scores, records, points, sp_scores, id


