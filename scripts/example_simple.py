from fewshot.experiment import SimpleExperiment

import argparse


# ... parse arguments ...
config = {
    "dataset": {
        "datase_dir": "./data/omniglot",
    },
    "backbone": {},
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
