from fewshot.experiment import BaselineExperiment

import argparse
import os


# parse arguments
parser = argparse.ArgumentParser(description='Baseline Few Shot Experiment')

parser.add_argument('--dataset_dir', type=str, required=True,
                    help="Path to dataset")
parser.add_argument('--img_width', type=int, default=84,
                    help="Image width")
parser.add_argument('--img_height', type=int, default=84,
                    help="Image height")
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed")

parser.add_argument('--backbone_type', type=str, default=BaselineExperiment.BackboneType.CONVNET,
                    help="Type of backbone network",
                    choices=BaselineExperiment.get_availible_backbone_types())
parser.add_argument('--backbone_num_classes', type=int, default=5,
                    help="Number of classes to train backbone network")
parser.add_argument('--backbone_train_size', type=float, default=0.5,
                    help="Train size of backbone dataset")
parser.add_argument('--backbone_batch_size', type=int, default=64,
                    help="Backbone training batch size")
parser.add_argument('--backbone_num_epoch', type=int, default=1,
                    help="Backbone training epochs")

parser.add_argument('--n_way', type=int, default=5,
                    help="Few Shot n_way parameter")
parser.add_argument('--k_shot', type=int, default=5,
                    help="Few Shot k_shot parameter")
parser.add_argument('--query_samples_per_class', type=int, default=16,
                    help="Number of samples for fewshot testing")
parser.add_argument('--fewshot_batch_size', type=int, default=4,
                    help="Few Shot training batch size")
parser.add_argument('--fewshot_batches_per_episode', type=int, default=4,
                    help="Few Shot number of iterations while training")
parser.add_argument('--n_episodes', type=int, default=2,
                    help="Number of episodes for testing")


args = parser.parse_args()


# check dataset existance
if not os.path.exists(args.dataset_dir):
    print("Use prepare_dataset.py script to download dataset.")
    exit()


# prepare config
config = {
    "name": "Baseline",

    "seed": args.seed,

    "dataset": {
        "dataset_dir": args.dataset_dir,
        "img_width": args.img_width,
        "img_height": args.img_height,
    },

    "backbone": {
        "type": args.backbone_type,
        "num_classes": args.backbone_num_classes,
        "train_size": args.backbone_train_size,
        "batch_size": args.backbone_batch_size,
        "num_epoch": args.backbone_num_epoch,
    },

    "fewshot": {
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "batch_size": args.fewshot_batch_size,
        "batches_per_episode": args.fewshot_batches_per_episode,
        "query_samples_per_class": args.query_samples_per_class,
        "n_episodes": args.n_episodes
    },
}


# ... create experiment ...
experiment = BaselineExperiment(config)

# ... run experiment ...
report = experiment.run()

# ... print report ...
print(report)
