import config
import numpy as np
import os

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

from loss import softmax_cross_entropy_with_logits


'''
For tictactoe, input to model is 3 turn history * (x plane + o plane) = 6 * 3x3 planes
P.S i forgot to do the history :p maybe tomorrow
and 1 * 3x3 plane for color (turn) total 7 * 3x3 planes 

output of policy head is length 9 vector of probability distribution
'''


class My_Model():
    def __init__(self, reg_const, learning_rate, input_dim, output_dim):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)

    def write(self, game, version):
        # add generation?? dont forget to change read too
        if not os.path.exists(f'{config.MODEL}/{game}'):
            os.makedirs(f'{config.MODEL}/{game}')
        self.model.save(f'{config.MODEL}/{game}/version{version}.h5')

    def read(self, game, run_number, version):
        return load_model(f'{config.MODEL}/version{version},h5')

    def load(self, game, run_number, version):
        self.model = load_model(f'{config.MODEL}/{game}/version{version}.h5',
                                custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

class Res_CNN(My_Model):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, mode_architecture):
        My_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
        self.mode_architecture = mode_architecture
        self.num_layers = len(mode_architecture)
        self.model = self._build_model()

    def residual_block(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size) # maybe don't normalize and activate here?

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return x

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def value_head(self, x):

        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(20, use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)

        x = LeakyReLU()(x)

        x = Dense(1, use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(self.reg_const), name='value_head')(x)

        return x

    def policy_head(self, x):

        x = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.output_dim*5,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x) # remember that this is for tictactoe only

        x = Dense(
            self.output_dim,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='policy_head'
        )(x)

        return x

    def _build_model(self):

        model_input = Input(shape=self.input_dim, name='model_input')

        x = self.conv_layer(model_input, self.mode_architecture[0]['filters'], self.mode_architecture[0]['kernel_size'])

        if len(self.mode_architecture) > 1:
            for h in self.mode_architecture[1:]:
                x = self.residual_block(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[model_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
                      loss_weights={'value_head': 0.25, 'policy_head': 0.75}
                      )

        return model

    def convert_to_model_input(self, game_states):
        # this takes records format (strings)
        X_train = []
        Y_train = {'value_head': [],
                   'policy_head': []}
        for game in game_states:
            board, pi, turn, value = game.split(',')
            board_plane = np.zeros((2, 3, 3))
            for idx, char in enumerate(board):
                if char == '1':
                    board_plane[0][int(idx/3)][idx%3] = 1
                elif char == '2':
                    board_plane[1][int(idx/3)][idx%3] = -1
            turn_plane = np.full((1, 3, 3), int(turn))
            X_train.append(np.vstack([board_plane, turn_plane]))
            Y_train['value_head'].append(float(value))
            Y_train['policy_head'].append([float(val) for val in pi.split(' ')[1:]])
        Y_train['value_head'] = np.array(Y_train['value_head'])
        Y_train['policy_head'] = np.array(Y_train['policy_head'])
        return np.array(X_train), Y_train

    def state_to_input(self, state):
        # this takes in A SINGLE TState (the game states directly)
        board_plane = np.zeros((2, 3, 3))
        for idx, val in enumerate(state.board):
            if val == 1:
                board_plane[0][int(idx/3)][idx%3] = 1
            else:
                board_plane[1][int(idx/3)][idx % 3] = 1
        turn_plane = np.full((1, 3, 3), state.turn)
        return np.vstack([board_plane, turn_plane])


