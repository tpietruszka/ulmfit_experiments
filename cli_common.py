import pathlib
import os
import sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen' # prevents some fastai imports from causing a crash
try:
    from ulmfit_experiments import experiments
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ulmfit_experiments import experiments

results_dir = (pathlib.Path(__file__).parent / 'trained_models').resolve()
