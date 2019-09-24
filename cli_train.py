import os
import sys
import json
import argparse
import pdb
import pathlib
import torch
import numpy as np
from typing import *
try:
    from ulmfit_experiments import experiments
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ulmfit_experiments import experiments

results_dir = (pathlib.Path(__file__).parent / 'trained_models').resolve()

def main():
    parser = argparse.ArgumentParser(description='Run an experiment and store a trained model')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    parser.add_argument('run_id', type=str, help='Name of subdirectory to store results in')
    args = parser.parse_args()
    output_dir = results_dir / args.run_id
    if output_dir.exists():
        raise ValueError("The selected run_id already exists. Select an unique one")


    config_file = pathlib.Path(args.config_file)
    config = json.load(open(config_file, 'r'))
    dataset_path = pathlib.Path(config['params']['dataset_path'])
    if not dataset_path.is_absolute():
        # if not absolute, calculate the path relative to the config file
        dataset_path = (config_file.parent / dataset_path).resolve()
        config['params']['dataset_path'] = str(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path in the config has to point to an "
        "existing directory with files like train.csv and with a directory with "
        "the encoder. The past can be absolute or relative to the config file. "
        f"Got: {dataset_path}")

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    ex = experiments.ExperimentCls.factory(config['type'], config['params'])
    results, learner = ex.run()

    # load the databunch again, if it was evicted during test set scoring
    # this is necessary to export the learner
    if learner.data.train_dl is None:
        learner.data = ex.get_data_bunch(*ex.get_dfs())

    output_dir.mkdir()
    print(f'Storing results in {output_dir}')
    learner.export(output_dir / 'learner.pkl')
    json.dump(results, open(output_dir/'results.json', 'w'))
    json.dump(config, open(output_dir/'config.json', 'w'))
    if 'test_score' in results.keys():
        print(f'Test score: {results["test_score"]}')

if __name__ == "__main__":
    main()
