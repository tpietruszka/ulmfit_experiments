from typing import *
import dataclasses
import pandas as pd
import pymongo
import json
from . import experiments


def run_list_stats(run_list: Collection[int], run_results: Dict[int, Dict]) -> pd.Series:
    """Takes a list of run ids to calculate stats for and a dict of run id -> 'results' field of
    that run, as retrieved from mongo"""
    batch_stats = []
    for run in run_list:
        batch_stats.append({'run_id': run,
                            'test_score': run_results[run]['test_score'],
                            'best_val_acc': run_results[run]['best_val_acc'],
                            'last_val_acc': run_results[run]['phase_stats'][-1][-1]['accuracy']})
    qdf = pd.DataFrame(batch_stats).set_index('run_id').astype(float)
    row_stats = pd.concat([qdf.mean().rename(lambda x: 'mean_' + x), qdf.std().rename(lambda x: 'std_' + x)])
    return row_stats


def get_results(db: pymongo.database.Database, exp_type: str, dicts_to_json: bool = True) -> Tuple[pd.DataFrame,
                                                                                                   Dict[int, Dict]]:
    results = list(db['completed'].find({'type': exp_type}))
    pdf = pd.DataFrame([dict(id=r['_id'], **r['params']) for r in results]).set_index('id')

    # fill empty params with default values. By convention, as new settings are added, their default values should be to
    # previous behaviour
    for f in dataclasses.fields(experiments.ExperimentCls.subclass_registry[exp_type]):
        if not isinstance(f.default, dataclasses._MISSING_TYPE) and f.name in pdf.columns:
            pdf.loc[pdf[f.name].isnull(), f.name] = f.default

    if dicts_to_json:
        # to make values hashable
        for col in ['training_phases', 'aggregation_params']:
            pdf.loc[:, col] = pdf.loc[:, col].apply(lambda x: json.dumps(x))

    # 1-el lists are incorrectly fetched as a simple int, lists have to be replaced with tuples
    listlike_columns = pdf.columns[pdf.applymap(lambda x: isinstance(x, List)).any()]
    for col in listlike_columns:
        pdf.loc[:, col] = pdf[col].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    params_df = pdf.drop_duplicates(keep='last')
    results_dct = {r['_id']: r['results'] for r in results}
    return params_df, results_dct


def grouped_results_stats(params_df, results_dct, min_runs=15, drop_run_lists = True):
    gb = params_df.groupby(list(set(params_df.columns) - {'subsample_id'}))
    run_lists = gb.apply(lambda d: tuple(d.index)).rename('run_list').to_frame()
    run_lists = run_lists.loc[run_lists.run_list.apply(lambda x: len(x)) > min_runs].reset_index()
    const_cols = run_lists.columns[run_lists.apply(lambda x: x.nunique()) == 1]
    const_values = run_lists[const_cols].head(1).squeeze().sort_index().rename('const_values')
    run_lists = run_lists.drop(columns=const_cols)
    stats_df = run_lists.run_list.apply(run_list_stats, run_results=results_dct)
    df = run_lists.join(stats_df).sort_values('mean_test_score')
    if drop_run_lists:
        df = df.drop(columns=['run_list'])
    return df, const_values
