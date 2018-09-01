import tensorflow as tf
import logging


class Logger(object):
    @staticmethod
    def init():
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler('tensorflow.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    @staticmethod
    def summarize_images(images_tensor, name, collection, images_num=10):
        tf.summary.image(name, images_tensor, images_num, collections=[collection])

    @staticmethod
    def info(msg):
        tf.logging.info(msg)

    @staticmethod
    def warn(msg):
        tf.logging.warn(msg)

    @staticmethod
    def error(msg):
        tf.logging.error(msg)

    @staticmethod
    def debug(msg):
        tf.logging.debug(msg)

    @staticmethod
    def shape(tensor):
        tf.logging.debug(tensor.get_shape().as_list())
