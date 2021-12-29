from cnn import cnn_train
import tensorflow as tf
if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        cnn_train()