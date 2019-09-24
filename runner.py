import pymongo
import time
import copy
import pandas as pd
import traceback
import torch
import os
import sys
import numpy as np
import argparse
from typing import *
try:
    from ulmfit_experiments import experiments
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ulmfit_experiments import experiments


class MongoExperimentRunner:
    completed_collection = 'completed'
    failed_collection = 'failed'
    queue_collection = 'queue'

    status_waiting = 'waiting'
    status_in_progress = 'in_progress'

    def __init__(self, experiment_factory: Callable[[str, Dict], experiments.ExperimentCls], trained_models_dir: str,
                 connection_string: str = 'mongodb://localhost', dbname: str = 'ulmfit_experiments'):
        self.experiment_factory = experiment_factory
        self.trained_models_dir = trained_models_dir
        self.connection_string = connection_string
        self.dbname = dbname

    def run_job(self, job: Dict):
        """Can be called directly or with an argument from the DB"""
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        ex = self.experiment_factory(job['type'], job['params'])
        try:
            results, learn = ex.run()
            job['results'] = results
            target_collection = self.completed_collection
        except Exception:
            error = traceback.format_exc()
            print("ERROR! Experiment run failed.\n" + error)
            job['exception'] = error
            target_collection = self.failed_collection
            learn = None
        del job['status']
        job['time_completed'] = str(pd.Timestamp('now'))
        entry_id = self._store_results(job, target_collection)
        if learn is not None:
            learn.save(os.path.join(self.trained_models_dir, str(entry_id)), with_opt=False)
        return target_collection, entry_id

    # def get_params_from_run(self, run_id, collection_name=None):
    #     if collection_name is None:
    #         collection_name = self.completed_collection
    #     db = self._get_db()
    #     run = db[collection_name].find_one(run_id)
    #     return run

    def submit_job(self, job: Dict):
        assert 'type' in job.keys() and 'params' in job.keys()
        job = copy.deepcopy(job)
        job['status'] = self.status_waiting
        job['time_submitted'] = str(pd.Timestamp('now'))
        db = self._get_db()
        inserted = db[self.queue_collection].insert_one(job)
        return inserted

    def run_job_from_queue(self):
        queue = self._get_db()[self.queue_collection]
        new_job = queue.find_one_and_update(filter={'status': self.status_waiting},
                                            update={'$set': {'status': self.status_in_progress,
                                                             'time_started': str(pd.Timestamp('now'))}},
                                            sort=[('time_submitted', pymongo.ASCENDING)])
        if new_job is None:
            print('No jobs in the queue, waiting')
            return
        job_meta = {k: v for k, v in new_job.items() if k != "params"}
        print(f'Found job: {job_meta}')
        result = self.run_job(new_job)
        queue.delete_one({'_id': new_job['_id']})
        print(f'Finished a job, results: {result}, job: {job_meta}')
        return result

    def process_queue(self):
        while True:
            res = self.run_job_from_queue()
            if res is None:
                time.sleep(5)

    def _store_results(self, results, collection_name):
        db = self._get_db()
        collection = db[collection_name]
        to_insert = copy.deepcopy(results)
        while True:
            try:
                last_used_id = collection.find_one(sort=[('_id', pymongo.DESCENDING)])['_id']
            except (TypeError, ValueError):
                last_used_id = 0
            first_free_id = last_used_id + 1
            to_insert['_id'] = first_free_id
            try:
                collection.insert_one(to_insert)
                break
            except pymongo.errors.DuplicateKeyError:
                print(f'Error saving to mongo - duplicate key {first_free_id}- retrying')
                pass
        return first_free_id

    def _get_db(self):
        client = pymongo.MongoClient(self.connection_string)
        return client[self.dbname]


def main():
    parser = argparse.ArgumentParser(description='Process experiments from a mongodb queue')
    parser.add_argument('result_models_dir', type=str, help='Directory to store trained models in')
    parser.add_argument('--mongo_dsn', type=str, default='mongodb://localhost')
    parser.add_argument('--mongo_collection', type=str, default='ulmfit_experiments')
    args = parser.parse_args()

    runner = MongoExperimentRunner(experiments.ExperimentCls.factory, args.result_models_dir,
                                   args.mongo_dsn, args.mongo_collection)
    runner.process_queue()


if __name__ == "__main__":
    main()
