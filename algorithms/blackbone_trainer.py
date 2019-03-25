import tensorflow as tf

from .base_trainer import BaseTrainer

class BlackboneTrainer(BaseTrainer):
    def train(self, sess, blackbone_net, head_op, loss_func, optimizer, n_epochs, train_dataset, val_dataset, **kwargs):
        self._init_ops(blackbone_net, head_op, loss_func, optimizer)
        for epoch in range(n_epochs):
            self._train_step(blackbone_net, train_dataset)
            self._validation_step(blackbone_net, val_dataset)

        return blackbone_net

    def _train_step(self, sess, train_dataset):
        for x_batch, y_batch in train_dataset.get_generator():
            loss, _ = sess.run(
                [self._loss, self._minimize],
                feed_dict={
                    self._input : x_batch,
                    self._target : y_batch
                }
            )

    def _validation_step(self, *args):
        pass

    def _init_ops(self, blackbone_net, head_op, loss_func, optimizer):
        self._target = tf.placeholder(tf.float32)
        self._input = blackbone_net.get_input()
        self._loss = loss_func(self._target, head_op(blackbone_net.get_output()))
        self._minimize = optimizer.minimize(self._loss)


