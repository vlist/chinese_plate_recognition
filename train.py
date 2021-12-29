from cnn import cnn_train
import tensorflow as tf
if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    cnn_train()