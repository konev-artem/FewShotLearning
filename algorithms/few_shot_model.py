import tensorflow as tf

class CosineLayer():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = tf.Variable(tf.zeros(input.get_shape()[1], self.num_classes))

    def __call__(self, input):
        # change initialisation

        dot_product = tf.matmul(input, self.W, transpose_b=True)
        f_norm = tf.pow(tf.reduce_sum(tf.multiply(input, input), axis=1, keepdims=True), 0.5)
        w_norm = tf.pow(tf.reduce_sum(tf.multiply(self.W, self.W), axis=0, keepdims=True), 0.5)

        return dot_product / f_norm / w_norm

    def reset(self):
        self.W.assign(tf.zeros(input.get_shape()[1], self.num_classes))


class FewShotModel():
    def __init__(self, sess, blackbone, num_classes):
        self.sess = sess
        self.blackbone = blackbone
        self.cosine_output = CosineLayer(num_classes)(blackbone.get_output())
        self.softmax_op = tf.nn.softmax(self.cosine_output, axis=-1)
        self.y_ph = tf.placeholder()

    def fit(self, X_train, y_train, augmentation, iters=1000):
        self.cosine_output.reset()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.cosine_output, labels=self.y_ph)
        minimize = tf.train.AdamOptimizer().minimize(loss, var_list=[CosineLayer.W])

        for i in range(iters):
            self.sess.run(minimize, feed_dict={
                self.blackbone.get_input() : augmentation(X_train), self.y_ph : y_train}
        )


    def predict(self, X_test):
        return self.sess.run(self.softmax_op, feed_dict={self.blackbone.get_input() : X_test})
