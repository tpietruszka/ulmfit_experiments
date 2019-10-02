import sys
import os
import argparse
import cli_common
from ulmfit_experiments import experiments # has to be imported after cli_common
from fastai.text import *
from fastai.text.learner import RNNLearner
import torch

results_dir = (pathlib.Path(__file__).parent / 'trained_models').resolve()

def process_text(text: str, learner: RNNLearner, verbose=True):
    results = learner.predict("xxbos " + text)  # skipping building a DataBunch,
    # so we have to add BOS tokens here
    decision = results[0]
    probabilities = ','.join([str(x) for x in to_np(results[2])])
    if verbose:
        print(f"Predicted class: {decision}")
        print(f"Predicted probabilities of all classes: {probabilities}")
    else:
        print(f"{decision},{probabilities}")

def main():
    parser = argparse.ArgumentParser(description='Load a trained model and score texts')
    parser.add_argument('run_id', type=str, help='Model to load. Corresponds to a directory within "./trained_models/"')
    parser.add_argument('--batch', action='store_true', default=False, help='Batch mode - do not display prompts, read until EOF')
    parser.add_argument('--cpu', action='store_true', default=False, help='Run on CPU only')
    args = parser.parse_args()
    if args.cpu:
        defaults.device = torch.device('cpu')
    model_dir = results_dir / args.run_id
    learner = load_learner(model_dir, 'learner.pkl')  # TODO: move paths etc to a config
    if not args.batch:  # interactive mode
        while True:
            try:
                text = input("Enter a text to process (new line ends a text, Ctrl-C to exit): ")
                process_text(text, learner, verbose=True)
            except KeyboardInterrupt:
                print("Exiting")
                break
    else:
        for line in sys.stdin.readlines():
            process_text(line, learner, verbose=False)


if __name__ == "__main__":
    main()
