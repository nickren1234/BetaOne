from tensorflow.keras.utils import plot_model

# from tictactoe import Tictactoe, State
from gomoku import Gomoku, State
from agent import Agent
from model import Res_CNN
from play import play_matches
from records import Records

import config


if __name__ == '__main__':
    id_count = 1
    # game = Tictactoe(id_count)
    game = Gomoku(id_count)
    cur_model = Res_CNN(config.REG, config.LEARNING_RATE, game.input_dim, game.policy_dim, config.MODEL_ARCH)
    best_model = Res_CNN(config.REG, config.LEARNING_RATE, game.input_dim, game.policy_dim, config.MODEL_ARCH)

    records = Records(config.BEST_VER)
    records.load_meta()

    # print(config.BEST_VER)
    if config.BEST_VER is None:
        best_version = 0
        best_model.model.set_weights(cur_model.model.get_weights())
    else:
        best_version = config.BEST_VER
        best_model.load(game.name, config.BEST_VER, best_version)
        cur_model.load(game.name, config.BEST_VER, best_version)

    plot_model(cur_model.model)

    cur_agent = Agent('cur_agent', game.input_dim, game.policy_dim, config.MCTS_CYCLES, config.CPUCT, cur_model)
    best_agent = Agent('best_agent', game.input_dim, game.policy_dim, config.MCTS_CYCLES, config.CPUCT, best_model)

    for i in range(5):
        print('Starting iteration ' + str(i))
        print('Best version: ' + str(best_version))

        # self play
        _, records, _, _, id_count = play_matches(best_agent, best_agent, config.SELF_PLAY_GAMES, config.TURNS_UNTIL_TAU0, id_count, records)

        # now we train the model with the games we just played
        game_states = cur_agent.build_training_set(records)
        cur_agent.train(game_states, game)

        # compare the new agent with the best agent
        scores, _, points, sp_scores, id_count = play_matches(best_agent, cur_agent, config.SELF_PLAY_GAMES, config.TURNS_UNTIL_TAU0, id_count)
        print(f"best agent won {scores['best_agent']}, current agent won {scores['cur_agent']}, {scores['drawn']} games drawn")
        print(f"1st agent won {sp_scores['sp']}, 2nd agent won {sp_scores['nsp']}, {sp_scores['drawn']} games drawn")
        if scores['cur_agent'] > scores['best_agent'] * 1.22:
            # if new player wins 55% or more ignoring draws
            # 45 * 1.22 = 54.9
            # since we are only doing 20 games, we will say 2 more wins is good enough
            best_version += 1
            records.version = best_version
            best_agent.model.model.set_weights(cur_agent.model.model.get_weights())
            best_model.write(game.name, best_version)
        else:
            cur_agent.model.model.set_weights(best_agent.model.model.get_weights())
        print(f'best version is now {best_version}')

    '''
    test_states = [State(1, [1, -1, 0, 0, 1, 0, 0, 0, -1], 1)]  # this position is won for 1, if pos 5 or 6 is played
    test_states.append(State(2, [1, -1, 0, 0, 1, 0, 1, 0, -1], -1))  # this position is lost for -1 but it should play 2 or 3
    test_states.append(State(3, [1, 1, -1, -1, -1, 0, 1, 0, 0], 1))  # this position should be a draw, it must play 5
    test_states.append(State(4, [1, 1, -1, 0, -1, 0, 0, 0, 0], 1))  # should be a draw, but 6 must be played
    test_states.append(State(5, [0, -1, 0, 0, 1, 0, 0, 0, 0], 1))  # should be won for 1 if 7 is not played
    test_states.append(State(6, [1, -1, 0, 0, 1, 0, 0, 0, 0], -1))  # should be lost for -1 but play 8 to delay
    for state in test_states:
        print(best_agent.get_preds(state))
    '''


# note for tomorrow, kernel size, the inputs are 3x3 which is bad
# cause kernel reduces dimension so we add padding?????
# also add history :p
# more dense layers maybe??


