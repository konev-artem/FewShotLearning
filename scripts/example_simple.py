from fewshot.experiment import SimpleExperiment

import argparse
import os


# parse arguments
parser = argparse.ArgumentParser(description='Baseline Few Shot Experiment')
parser.add_argument('--dataset_dir', type=str, required=True,
                    help="Path to dataset")
parser.add_argument('--backbone_type', type=str, required=True,
                    help="Type of backbone network",
                    choices=SimpleExperiment.get_availible_backbone_types())
parser.add_argument('--backbone_input_size', type=int, default=64,
                    help="Width and height of backbone first layer input"
                         "Images will be rescaled to this size")
args = parser.parse_args()


# check dataset existance
if not os.path.exists(args.dataset_dir):
    print("Use prepare_dataset.py script to download dataset.")
    exit()


# prepare config
config = {
    "dataset": {
        "dataset_dir": args.dataset_dir,
    },
    "backbone": {
        "type": args.backbone_type,
        "input_size": args.backbone_input_size,
    },
    "trainer": {},
    "tester": {},
    "metrics": {}
}


# ... create experiment ...
experiment = SimpleExperiment(config)

# ... run experiment ...
report = experiment.run()

# ... print report ...
print(report)
