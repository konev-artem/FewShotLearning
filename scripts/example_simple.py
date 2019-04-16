from fewshot.experiment import SimpleExperiment

import argparse
import os


# parse arguments
parser = argparse.ArgumentParser(description='Baseline Few Shot Experiment')
parser.add_argument('--dataset_dir', type=str, required=True,
                    help="Path to dataset")
parser.add_argument('--backbone_type', type=str, default=SimpleExperiment.BackboneType.CONVNET,
                    help="Type of backbone network",
                    choices=SimpleExperiment.get_availible_backbone_types())
parser.add_argument('--backbone_input_size', type=int, default=64,
                    help="Width and height of backbone first layer input"
                         "Images will be rescaled to this size")
parser.add_argument('--backbone_dataset_size', type=float, default=0.5,
                    help="Ratio of classes passed to backbone training")
parser.add_argument('--backbone_val_size', type=float, default=0.1,
                    help="Ratio of backbone dataset passed to validation for early stopping")
parser.add_argument('--backbone_n_epochs', type=int, default=10,
                    help="Number of epoch to train the backbone network")

args = parser.parse_args()


# check dataset existance
if not os.path.exists(args.dataset_dir):
    print("Use prepare_dataset.py script to download dataset.")
    exit()


# prepare config
config = {
    "dataset": {
        "dataset_dir": args.dataset_dir,
        "backbone_dataset_size": args.backbone_dataset_size,
        "backbone_val_size": args.backbone_val_size,
    },

    "backbone": {
        "type": args.backbone_type,
        "input_size": args.backbone_input_size,
    },

    "trainer": {
        "backbone_n_epochs": args.backbone_n_epochs,
    },

    "tester": {},
    "metrics": {}
}


# ... create experiment ...
experiment = SimpleExperiment(config)

# ... run experiment ...
report = experiment.run()

# ... print report ...
print(report)
