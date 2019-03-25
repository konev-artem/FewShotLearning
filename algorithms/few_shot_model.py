import tensorflow as tf

class CosineLayer():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, input):
        # change initialisation
        self.W = tf.Variable(tf.zeros(input.get_shape()[1], self.num_classes))

        dot_product = tf.matmul(input, self.W, transpose_b=True)
        f_norm = tf.reduce_sum(tf.multiply(input, input), axis=1, keepdims=True)
        w_norm = tf.reduce_sum(tf.multiply(self.W, self.W), axis=0, keepdims=True)

        return dot_product / f_norm / w_norm


class FewShotModel():
    def __init__(self, sess, blackbone, num_classes):
        self.sess = sess
        self.blackbone = blackbone
        self.cosine_output = CosineLayer(num_classes)(blackbone.get_output())
        self.softmax_op = tf.nn.softmax(self.cosine_output, axis=-1)

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return self.sess.run(self.softmax_op, feed_dict={self.blackbone.get_input() : X_test})
