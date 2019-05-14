import os
import tqdm
import datetime

import numpy as np
import tensorflow as tf


def baseline_fewshot_test(model,
                          generator,
                          optimizer,
                          batch_size=4,
                          support_epochs=100,
                          n_episodes=10000,
                          model_name='baseline',
                          tensorboard=False,
                          log_dir='../fewshot/logs',
                          period=True):

    if tensorboard:
        log_dir = os.path.join(log_dir,
            '{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        os.makedirs(log_dir, exist_ok=True)
        file_writer = tf.summary.FileWriter(log_dir)

    accuracies = []
    tbar = tqdm.tqdm(range(n_episodes), total=n_episodes)
    for episode_index in tbar:
        (support_x, support_y), (query_x, query_y) = generator[episode_index]
        model.fit(support_x, support_y,
                  optimizer=optimizer,
                  batch_size=batch_size,
                  epochs=support_epochs,
                  verbose=0)

        out = model.predict(query_x, batch_size=batch_size)

        accuracy = np.mean(np.argmax(out, axis=1) == np.where(query_y == 1)[1])
        accuracies.append(accuracy)

        tbar.set_description(
            "Average acc: {:.2f}%".format(np.mean(accuracies) * 100))

        if tensorboard and (episode_index % period == 0):
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                tf.Summary.Value(tag='average_accuracy', simple_value=np.mean(accuracies))
            ])
            file_writer.add_summary(summary, global_step=episode_index)
            file_writer.flush()

    return accuracies

def bootstrap(accuracy, sz=1000, seed=42, ci_lvl=0.95, verbose=True):
    np.random.seed(seed)
    bts = np.random.choice(accuracy, size=(sz, len(accuracy)), replace=True)
    bts = np.sort(np.mean(bts, 1))
    quant_left = int((1 - ci_lvl) * sz // 2)
    left_bound = bts[quant_left]
    right_bound = bts[-quant_left]
    if verbose:
        print('metric: accuracy, mean: {:.2f}, std: {:.2f}, 95% conf interval: [{:.2f} ,{:.2f}]'.format(
            np.mean(accuracy), np.std(accuracy), left_bound, right_bound))
    return np.mean(accuracy), np.std(accuracy), left_bound, right_bound
