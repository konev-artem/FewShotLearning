from fewshot.experiment import SimpleExperiment

import argparse


# ... parse arguments ...
config = {}

# ... create experiment ...
experiment = SimpleExperiment()

# ... setup experiment ...
experiment.setup(config)

# ... run experiment ...
report = experiment.run()

# ... print report ...
print(report)
