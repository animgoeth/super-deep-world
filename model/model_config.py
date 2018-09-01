class BaseConfig(object):
    def __init__(self):
        self.input_shape = [64, 64, 1]
        # self.input_shape = [224, 256, 1]
        self.epochs_count = 701
        self.iterations_per_epoch = 450
        self.truncated_steps = 9
        self.learning_rate = 3 * 1e-4
        self.train_sequences_count = 700
        self.max_checkpoints_to_keep = 5
        self.checkpoint_save_interval = 100


class FullVideoPixelNetworkConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.rmb_c = 128
        self.encoder_rmb_num = 8
        self.decoder_rmb_num = 12
        self.conv_lstm_filters = 256


class MediumVideoPixelNetworkConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.rmb_c = 32
        self.encoder_rmb_num = 4
        self.decoder_rmb_num = 6
        self.conv_lstm_filters = 64


class SmallVideoPixelNetworkConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.rmb_c = 32
        self.encoder_rmb_num = 2
        self.decoder_rmb_num = 3
        self.conv_lstm_filters = 64
