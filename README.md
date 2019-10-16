## ULMFiT experiments

This project was created to queue, run and store results of experiments -
attempts to improve the ULMFiT algorithm, focusing on the classifier head
architecture.

Currently, this project can handle 3 different classification tasks on their
corresponding datasets: "IMDB", "Poleval Cyberbullying Detection" and "Fact
Checking Question Classification". For each of them, there is a subfolder within
`./data`, containing labeled training and test datasets, as well as one
language model for each, adapted for the specific domain.

Simple scripts have been added to enable training and evaluating classifiers
without setting up a database and preparing dedicated notebooks.

`visualize.py` lets the user feed text to the model, then inspect attention
scores associated with each token and features calculated for it. The interface
is browser-based, the script starts a simple local webserver based on Dash.

The primary mode of operation originally was to run
`python runner.py [storage directory]`, which creates a worker to process a
queue of experiments. Both the queue and the results are stored in a MongoDB
database, a DSN to which can be passed to the script as an additional parameter
(see `python runner.py -h`). Results analysis was performed using various
Jupyter notebooks, depending on the current research hypothesis.



### Training
To train a classifier, after the setup as described below,
it should be sufficient to run:
```
python cli_train.py [config_file] [run_id]
```
The script reads a given config file in JSON format. It should be structured
exactly the same way as the previously mentioned MongoDB documents. Examples are
provided in the `example_configs` directory. For IMDB, there are two configs -
one describes training on the full dataset, one on a subset of it. For the
remaining datasets there is one config for each.

The script creates the networks, trains it, checks metrics on the test set if
the config requires that, and stores several things in
`./trained_models/[run_id]`:

- a copy of the config,
- a JSON file with results (quite verbose, `test_score` is perhaps the most
  interesting)
- a trained model, which can later be used by `cli_deploy.py`.

### Visualizing attention

`python visualize.py [run_id]` loads a given model, and starts a webserver where
the user can supply texts, see the classification decisions and inspect
the attention scores and features calculated for each token.

To force evaluation on the CPU (e.g. if there is a CUDA device, but it is too
old), add a `--cpu` flag.

The webserver should be available on localhost, port 8050.

Trained models corresponding to the example configs are provided, and should be
ready to use as soon as requirements are met and setup (below) is completed.

### Model evaluation

`python cli_deploy.py [run_id]` simply loads a given classification model, and
then evaluates texts provided through standard input using that model. By
default, it prompts the user for each line of text and displays the results.

To force evaluation on the CPU (e.g. if there is a CUDA device, but it is too
old), add a `--cpu` flag.

With a `--batch` flag, it reads standard input until EOF is found, and outputs
results as CSV (the first column being the predicted class, and the rest -
  probabilities of belonging to each of the classes).

### Requirements
#### Training models - CUDA enabled:

- NVidia GPU with at least 8 GB RAM (for model training, less for evaluation)
- CUDA 9 drivers
- Conda package manager (https://www.anaconda.com/distribution/)
- Linux-based OS (might work on Windows, but has not been tested)

The conda environment has to be set up according to `environment.yml` (see below)

Write-access to the filesystem is necessary (including the data directory, for temporary files).
#### Evaluating models, visualization - possible without GPU

- Conda package manager
- Linux-based OS (might work on Windows, but has not been tested)

The conda environment can be set up either with `environment.yml`
(if the requirements for training models are met), or with `environment_cpu.yml`
(to run all calculations on the CPU, without CUDA acceleration).


### Setup
As described above, if the system is equipped with an appropriate CUDA-enabled
GPU, `environment.yml` should be used. Otherwise, `environment_cpu.yml` can
be used to prepare just for evaluating and visualizing attention scores.

To prepare the chosen Conda environment, run:
```
conda env create -f environment.yml
```
or
```
conda env create -f environment_cpu.yml
```


To activate the environment (for Conda >= 4.4):
```
conda activate ulmfit_experiments
```
or
```
conda activate ulmfit_experiments_cpu
```
### Example runs
Example usage of the provided `fact_checking_example` model:
```
(ulmfit_experiments) tomasz@pc1:~/ulmfit_experiments$ python cli_deploy.py fact_checking_example
Enter a text to process (new line ends a text, Ctrl-C to exit): Hey who is your favourite hairdresser?                                           
Predicted class: Opinion
Predicted probabilities of all classes: 0.029423103,0.66891485,0.30166206
Enter a text to process (new line ends a text, Ctrl-C to exit): What are the requirements to apply for permanent residence?
Predicted class: Factual
Predicted probabilities of all classes: 0.52320236,0.3709855,0.10581216
Enter a text to process (new line ends a text, Ctrl-C to exit): Hey, how is your day going?
Predicted class: Socializing
Predicted probabilities of all classes: 0.051203348,0.36467427,0.5841224
```

Training on the full IMDB dataset, with parameter AGG=1. Validation set is very small in this particular case, and not very representative.
```
(ulmfit_experiments) tomasz@pc1:~/ulmfit_experiments$ python cli_train.py example_configs/imdb_full_agg_1.json imdb_full_agg_1
Running tokenization...
Size of vocabulary: 60003                                                                                                                        
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', 'the', ',', '.', 'and', 'a', 'of', 'to', 'is', 'it', 'in', 'i', 'this']
epoch     train_loss  valid_loss  accuracy
1         0.472800    0.263956   0.920000                                      
Total time: 03:12
epoch     train_loss  valid_loss  accuracy
1         0.453303    0.246828    0.912000
Total time: 04:03
epoch     train_loss  valid_loss  accuracy
1         0.441725    0.221541    0.936000
Total time: 06:19
epoch     train_loss  valid_loss  accuracy
1         0.408355    0.204756    0.928000
2         0.423333    0.209902    0.924000
3         0.446167    0.207469    0.928000
4         0.401606    0.207810    0.924000
5         0.432790    0.198340    0.936000
6         0.415022    0.199554    0.924000
Total time: 42:38

Storing results in [...]/ulmfit_experiments/trained_models/imdb_full_agg_1
Test score: 0.9458000063896179
```
Evaluation using the same model:
```
(ulmfit_experiments) tomasz@pc1:~/ulmfit_experiments$ python cli_deploy.py imdb_full_agg_1
Enter a text to process (new line ends a text, Ctrl-C to exit): This was such a very deeply touching movie                                       
Predicted class: 1
Predicted probabilities of all classes: 0.009532124,0.99046785
```

## Webserver deployment

### Deploying `visualize.py` using Gunicorn

Command-line options cannot be passed to the script, using env variables instead.
Variables such as ip and port are meaningless - they have to be set in gunicorn.

Example run:

```
VIS_CPU=True VIS_RUN_ID="imdb_full_agg_1" gunicorn visualize:app.server -b 0.0.0.0:8050
```

### Docker, docker-compose

The `visualize.py` tool can be Dockerized, needs to mount a volume with the
`trained_models` dir contents and optionally a nginx proxy dealing with SSL.

An example config is provided in `docker-compose-visualize.yml`.
It should be adapted with the correct location of the `trained_models` dir,
and network name of the nginx proxy's network should be used.

Example run:
```
docker-compose -f docker-compose-visualize.yml up --build
```
