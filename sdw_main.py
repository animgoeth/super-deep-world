import tensorflow as tf
from model.model_config import *
from model.vpn_model import VideoPixelNetworkModel
from infra.data_generator import DataGenerator
from infra.trainer import Trainer
from infra.logger import Logger
import argparse as ap


def prepare_config(args, config):
    if args.arch_size == 'full':
        config = FullVideoPixelNetworkConfig()
    if args.arch_size == 'medium':
        config = MediumVideoPixelNetworkConfig()
    elif args.arch_size == 'small':
        config = SmallVideoPixelNetworkConfig()

    config.train = True
    config.load = args.resume
    config.batch_size = 1
    config.data_dir = args.data_path
    config.train_output = args.train_output_path + args.arch_size + '/'
    config.checkpoint_dir = config.train_output + 'checkpoints/'
    config.test_output = args.test_output_path

    return config


def execute(args):
    Logger.init()
    Logger.info("SDW started")
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    config = prepare_config(args, config)

    model = VideoPixelNetworkModel(config)
    data_generator = DataGenerator(config)
    trainer = Trainer(session, model, data_generator, config)

    trainer.train()
    trainer.test_all()


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("--arch_size", dest="arch_size", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--data_path", dest="data_path", default="./data/")
    parser.add_argument("--train_output_path", dest="train_output_path", default="./output/")
    parser.add_argument("--test_output_path", dest="test_output_path", default="./test_output/")
    parser.add_argument("--resume", dest="resume", default=True)
    args = parser.parse_args()

    execute(args)
