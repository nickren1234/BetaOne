
import config
import os
from collections import defaultdict

class Records():
    def __init__(self, version, threshold=50):
        self.threshold = threshold
        self.version = version
        if version is None:
            self.version = 0
        self.meta = defaultdict() # meta[version] = size of the corresponding file
        self.total = 0
        self.records = []

    def load_meta(self):
        with open(f'{config.RECORDS}/meta.txt', 'a+') as f:
            for line in f.read().splitlines():
                version, game_num, count = line.split(',')
                if int(version) not in self.meta.keys():
                    self.meta[int(version)] = {'total': 0}
                self.meta[int(version)][int(game_num)] = int(count)
                self.meta[int(version)]['total'] += int(count)
                self.total += int(count)

    def append(self, state, pi):
        # string is board,pi,turn and then value will be appended after mcts
        # and the turn is 0 and 1 instead of -1 and 1
        plain_text = state.convert_to_record()
        for x in pi:
            plain_text += ' ' + str(x)
        plain_text += ',' + str(max(state.turn, 0))
        self.records.append(plain_text)

    def store_to_file(self, game_num):
        if not os.path.exists(f'{config.RECORDS}/version{self.version}'):
            os.makedirs(f'{config.RECORDS}/version{self.version}')
        with open(f'{config.RECORDS}/version{self.version}/game{game_num}.txt', 'w+') as f:
            f.write('\n'.join(self.records) + '\n')
        self.total += len(self.records)
        with open(f'{config.RECORDS}/meta.txt', 'a+') as f:
            f.write(f'{self.version},{game_num},{len(self.records)}\n')
        if self.version not in self.meta.keys():
            self.meta[self.version] = {'total': 0}
        self.meta[self.version]['total'] += len(self.records)
        self.meta[self.version][game_num] = len(self.records)
        self.records = []

    def read(self):
        with open(f'{config.RECORDS}/version{self.version}.txt', 'r') as f:
            self.records = f.readlines()

