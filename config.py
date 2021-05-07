EPSILON = 0.2
ALPHA = 0.8
MODEL = 'models'
RECORDS = 'game_records'
BEST_VER = 4

BATCH_SIZE = 256
TRAINING_SIZE = 3000 # might need to be lowered for chess?
EPOCHS = 1
REG = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10
MCTS_CYCLES = 200
SELF_PLAY_GAMES = 20
TURNS_UNTIL_TAU0 = 1
CPUCT = 2

MODEL_ARCH = [{'filters': 75, 'kernel_size': (2, 2)},
              {'filters': 75, 'kernel_size': (2, 2)},
              {'filters': 75, 'kernel_size': (2, 2)},
              {'filters': 75, 'kernel_size': (3, 3)},
              {'filters': 75, 'kernel_size': (2, 2)}]

