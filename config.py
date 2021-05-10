EPSILON = 0.2
ALPHA = 0.8
MODEL = 'models'
RECORDS = 'game_records'
BEST_VER = 1

BATCH_SIZE = 256
TRAINING_SIZE = 3000 # might need to be lowered for chess?
EPOCHS = 1
REG = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 15
MCTS_CYCLES = 200
SELF_PLAY_GAMES = 10
TURNS_UNTIL_TAU0 = 30
CPUCT = 4

MODEL_ARCH = [{'filters': 75, 'kernel_size': (3, 3)},
              {'filters': 75, 'kernel_size': (3, 3)},
              {'filters': 75, 'kernel_size': (3, 3)},
              {'filters': 75, 'kernel_size': (3, 3)},
              {'filters': 75, 'kernel_size': (3, 3)}]

