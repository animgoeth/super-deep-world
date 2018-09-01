import numpy as np
from infra.logger import Logger
import os


class DataGenerator:
    def __init__(self, config):
        self.config = config
        np.random.seed(123)

        sequences = []
        for training_file in os.listdir(config.data_dir):
            if "_train" in training_file:
                file_contents = np.load(os.path.join(config.data_dir, training_file))
                for row in file_contents:
                    sequences.append(row)

        sequences = np.expand_dims(np.squeeze(sequences), 4)
        shuffled_idxs = np.arange(sequences.shape[0])
        np.random.shuffle(shuffled_idxs)
        sequences = sequences[shuffled_idxs]

        Logger.info('Data shape: %s' % str(sequences.shape))

        self.train_sequences = sequences[:config.train_sequences_count]
        self.test_sequences = sequences[config.train_sequences_count:]

    def next_batch(self):
        while True:
            idx = np.random.choice(self.config.train_sequences_count, self.config.batch_size)
            current_sequence = self.train_sequences[idx]

            return current_sequence[:, :self.config.truncated_steps + 1], current_sequence[:, self.config.truncated_steps:2 * self.config.truncated_steps + 1]

    def test_all(self):
        test_data = []
        for i in range(0, self.test_sequences.shape[0]):
            current_sequence = self.test_sequences[np.array([i])]
            test_data.append((current_sequence[:, :self.config.truncated_steps + 1],
                              current_sequence[:,self.config.truncated_steps:2 * self.config.truncated_steps + 1]))
        return test_data
