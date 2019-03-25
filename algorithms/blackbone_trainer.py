import tensorflow as tf

from .base_trainer import BaseTrainer

class BlackboneTrainer(BaseTrainer):
    def __init__(self, sess):
        self.sess = sess

    def train(self, blackbone_net, loss_func, optimizer, n_epochs, train_dataset, val_dataset, **kwargs):
        self._init_ops(blackbone_net, loss_func, optimizer)
        for epoch in range(n_epochs):
            self._train_step(blackbone_net, train_dataset)
            self._validation_step(blackbone_net, val_dataset)

    def _train_step(self, blackbone_net, train_dataset):
        for x_batch, y_batch in train_dataset.get_generator():
            loss, _ = self.sess.run(
                [self._loss, self._minimize],
                feed_dict={
                    blackbone_net.get_input() : x_batch,
                    self._target : y_batch
                }
            )


    def _validation_step(self, *args):
        pass

    def _init_ops(self, blackbone_net, loss_func, optimizer):
        self._target = tf.placeholder(tf.float32)
        self._loss = loss_func(self._target, blackbone_net.get_output())
        self._minimize = optimizer.minimize(self._loss)


