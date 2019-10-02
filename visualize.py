import argparse
import itertools
from typing import *
import os
import sys
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import cli_common
from ulmfit_experiments import experiments # has to be imported after cli_common
from fastai.text import learner, load_learner, to_np, defaults
from fastai.text.learner import RNNLearner
from cli_common import results_dir
import torch

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash('Attention visualization')

app.layout = html.Div(children=[
    html.H1(children='Attention visualization'),
    dcc.Textarea(id='userText', placeholder='Enter some text to analyze',
                 style={'width': '720px'}),
    html.Button('Evaluate', id='submitButton', style={'float': 'left'}),
    html.Div(children=[
        dcc.Graph(id='probabilitiesGraph', style={'width': '400px', 'float': 'left'}),
        html.Div(id='decisionDiv', style={'width': '400px', 'float': 'left', 'padding': 'auto'}),
        ], style={'width': '810px', 'float': 'left'}),
    html.Div(id='attentionWeightsDiv', style={'border': '1px solid black',
        'width': '720px', 'float': 'left'}),
], style={'float': 'left', 'width': '900px', 'margin': '0 auto'})

def process_sample(learn: RNNLearner, sample_raw: str) -> Tuple[str,
    Dict[str, float], pd.DataFrame]:
    """
    Process a sample using a fastai learner, collect results and attention
    """
    sample = 'xxbos ' + sample_raw
    proc = learn.data.train_ds.x.processor[0]
    results = learn.predict(sample)
    decision = str(results[0])
    probas = to_np(results[2])
    classes_probas = {str(c):p for c,p in zip(learn.data.train_ds.y.classes, probas)}
    weights = to_np(learn.model[1].attn.last_weights.squeeze())
    features = to_np(learn.model[1].attn.last_features.squeeze())

    tokens = proc.process_one(sample)
    weights = weights / weights.max() # highest one always 1

    single_text_df = pd.DataFrame([pd.Series(tokens), pd.Series(weights), pd.Series(features)]).T
    single_text_df.columns =['word', 'weight', 'sentiment']

    return decision, classes_probas, single_text_df

def render_word(word: str, sentiment: float, att_weight: float) -> html.Span:
    return html.Span(children=word, style={'background':
        f'hsla({sentiment}, 100%, 50%, {att_weight})'})

def render_decision(class_name: Union[str, None]) -> str:
    cls = str(class_name) if class_name else ''
    return f'Predicted class: {cls}'

def render_probabilities(probas: Dict[str, float]):
    figure = go.Figure(
        data=[go.Bar(x=[k], y=[v]) for k,v in probas.items()],
        layout=go.Layout(
            title='Predicted class probabilities',
            xaxis={
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'type': 'category',
            },
            yaxis={
                'showgrid': False,
                'showline': False,
                'zeroline': False,
                'range': [0,1],
            },
            margin={'l': 20, 'b': 20, 't': 30, 'r': 0},
            height=200,
            width=400,
            showlegend=False
            )
        )
    return figure


def main():
    parser = argparse.ArgumentParser(description='Load a trained model and score texts')
    parser.add_argument('run_id', type=str, help='Model to load. Corresponds to a directory within "./trained_models/"')
    parser.add_argument('--cpu', action='store_true', default=False, help='Run on CPU only')
    args = parser.parse_args()
    if args.cpu:
        defaults.device = torch.device('cpu')
    model_dir = results_dir / args.run_id
    learner = load_learner(model_dir, 'learner.pkl')  # TODO: move paths etc to a config

    @app.callback(
        [Output('decisionDiv', component_property='children'),
        Output('probabilitiesGraph', component_property='figure'),
        Output('attentionWeightsDiv', component_property='children')],
        [Input('submitButton', 'n_clicks')],
        state=[State(component_id='userText', component_property='value')]
    )
    def update_output_div(_, input_value):
        if not input_value:
            return render_decision(None), render_probabilities({}), ''
        decision, probas, att_df = process_sample(learner, input_value)
        att_df.sentiment *= 15
        att_df.sentiment += 50
        # features = ((features - features.mean()) / features.std()) * 15 + 50
        att_df.sentiment = att_df.sentiment.clip(0, 100)
        # red is 0, yellow 50, green 100
        att_word_spans = [render_word(r.word, r.sentiment, r.weight) for r in att_df.itertuples()]
        att_with_spaces = list(itertools.chain(*zip(att_word_spans, [' '] * len(att_word_spans))))
        return render_decision(decision), render_probabilities(probas), att_with_spaces

# TODO: rescaling and flipping sentiment as parameters
# TODO; choosing which feature to analyze
# TODO: histogram of the sentiment
# TODO: disable sentiment or attention
# TODO: show numbers on hover
# TODO: choose between models
    app.run_server(debug=True, host="0.0.0.0")

if __name__ == '__main__':
    main()
