import argparse
import itertools
from typing import *
import os
import sys
import pandas as pd
from dataclasses import dataclass
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import cli_common
from ulmfit_experiments import experiments # has to be imported after cli_common
from ulmfit_experiments import sequence_aggregations
from fastai.text import learner, load_learner, to_np, defaults
from fastai.text.learner import RNNLearner
from cli_common import results_dir
import torch

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

@dataclass
class AppConfig:
    run_id: str
    cpu: bool=False  # run only on CPU

    # section only relevant when running the development server
    port: Optional[int]=8050
    ip: str='localhost'
    debug: bool=False

    @classmethod
    def from_console(cls):
        parser = argparse.ArgumentParser(description='Load a trained model and score texts')
        parser.add_argument('run_id', type=str, help='Model to load. Corresponds to a directory within "./trained_models/"')
        parser.add_argument('--cpu', action='store_true', help='Run on CPU only')
        parser.add_argument('--port', help='Port to run the webserver on')
        parser.add_argument('--ip', help='IP to bind. To make the server available from other hosts use "0.0.0.0" (not recommended)')
        parser.add_argument('--debug', action='store_true', help='Run Dash in debug mode')
        parsed = parser.parse_args()
        return cls(**{k:v for k,v in vars(parsed).items() if v is not None})

    @classmethod
    def from_env(cls):
        params_dict = {}
        for name in cls.__dataclass_fields__.keys():
            name_env = 'VIS_' + name.upper()
            if name_env in os.environ.keys():
                params_dict[name] = os.environ[name_env]
        return cls(**params_dict)


max_text_len = 5000

app = dash.Dash('Attention visualization')
@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_folder, path)

featureNumberSlider = dcc.Slider(id='featureNumberSlider', min=0, max=0, step=1,
                                 value=0)
numFeaturesSpan = html.Span()

app.layout = html.Div(id='mainContainer', children=[
    html.Link(
        rel='stylesheet',
        href='/static/visualize.css'
    ),
    html.H1(children='Attention visualization'),
    html.Div(id='inputRow', className='row', children=[
        dcc.Textarea(id='userText', placeholder='Enter some text to analyze',
                     maxLength=max_text_len,
                     value="Great cast and director set my expectations very high. However, I do not consider it a good movie. It had the potential to be a decent thriller, but it was hampered by only having about twenty minutes worth of good script, which was mostly used up in the beginning. After that holes started to appear in the story that one could drive a truck through. The movie followed a descending curve from good to ordinary to bad to ludicrous by the time it concluded. It's not recommended.",
                     ),
        html.Button('Evaluate', id='submitButton'),
        ]),
    html.Div(children=[
    html.Label(htmlFor='featureNumberSlider',
               children=["Which feature to show? (Out of ", numFeaturesSpan, ")"]),
    featureNumberSlider
        ], className='row rowWithSlider'),
    html.Div(children=[
        html.Label(htmlFor='colorRangeSlider', children="Map this range of values into [red, green]:"),
        dcc.RangeSlider('colorRangeSlider', min=-15, max=15, value=[-3, 3], pushable=1,
                        marks={m: str(m) for m in range(-15, 18, 3)})
        ], className='row rowWithSlider'),
    html.Div(children=[
        html.Label(htmlFor='visualizeCheckbox', children="Show:"),
        dcc.Checklist(
            options=[
                {'label': 'Attention weights', 'value': 'weights'},
                {'label': 'Feature values', 'value': 'features'},
                {'label': 'Auxiliary tokens', 'value': 'extra_tokens'},
            ],
            value=['weights', 'features'],
            id='visualizeCheckbox')
        ], className='row'),

    html.Div(id='attentionWeightsDiv', children="Processed text will appear here"),
    html.Div(children=[
        dcc.Markdown(children="""
        **Opacity** of the color behind each token means the
        attention weight associated with it.

        **Hue** denotes the value of a feature calculated for that token,
        **considering its left context**, in red-to-green scale. It might
        be easily interpretable (e.g. as sentiment), but does not have to be,
        especially if there are more features.

        To examine just the attention weights or just the feature values for all
        tokens, uncheck the unnecessary checkbox above.
        
        Some auxiliary tokens were added to denote e.g. capitalization of the 
        next word ("xxmaj"), beginning of sequence ("xxbos"), words repeated
        more than 3 times ("xxwrep NUMBER"). Most common ones are not shown
        by default, but can be restored using the "auxiliary tokens" checkbox.
        
        """, id="instructionManual")
        ], className='row'),
    html.Div(children=[
        dcc.Graph(id='probabilitiesGraph', config={'displayModeBar': False}),
        html.Div(id='decisionDiv'),
        ], className='row'),



    html.Div(id='processedTextData', style={'display': 'none'}, children='{}'),
])


def process_sample(learn: RNNLearner, sample_raw: str) -> Tuple[str,
    Dict[str, float], pd.DataFrame]:
    """
    Process a sample using a fastai learner, collect results and attention
    """
    sample = 'xxbos ' + sample_raw
    if len(sample) > max_text_len:
        sample = sample[:max_text_len]  # trim extremely long text
    proc = learn.data.train_ds.x.processor[0]
    time_before = pd.Timestamp.now()
    results = learn.predict(sample)
    eval_time = (pd.Timestamp.now() - time_before)
    decision = str(results[0])
    probas = to_np(results[2])
    classes_probas = {str(c):p for c,p in zip(learn.data.train_ds.y.classes, probas)}
    weights = to_np(learn.model[1].attn.last_weights.squeeze())
    features = to_np(learn.model[1].attn.last_features.squeeze(0))

    tokens = proc.process_one(sample)
    print(f'Processed {len(tokens)} tokens in {eval_time}')
    weights = weights / weights.max() # highest one always 1

    feats_df = pd.DataFrame(features)
    feats_df.columns = 'feat_' + feats_df.columns.astype(str)
    single_text_df = pd.concat([pd.Series(tokens, name='word'),
                                   pd.Series(weights, name='weight'),
                                   feats_df], axis=1)
    return decision, classes_probas, single_text_df


def render_word(word: str, att_weight: float, feature: float, feature_color: float) -> html.Span:

    tooltip = html.Span(children=f'weight: {round(att_weight, 2)}\nvalue: {round(feature, 2)}', className='tooltiptext')
    return html.Span(children=[word, tooltip], style={'background':
        f'hsla({feature_color}, 100%, 50%, {att_weight})'}, className='tooltip')

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
            showlegend=False,
            )
        )
    return figure

@app.callback(Output('attentionWeightsDiv', component_property='children'),
    [Input('processedTextData', 'children'),
    Input('featureNumberSlider', 'value'),
    Input('colorRangeSlider', 'value'),
    Input('visualizeCheckbox', 'value')]
)
def display_attention(att_json, feat_number, color_range, visualize_what):
    att_df = pd.read_json(att_json).sort_index()
    if att_df.empty:
        return ''
    att_df = att_df.assign(feature = att_df.loc[:, 'feat_' + str(feat_number)])

    if 'features' in visualize_what:
        crange = color_range[1] - color_range[0]
        att_df = att_df.assign(feature_color =
                               ((att_df.feature - color_range[0]) / crange * 100).clip(0,100))
    else:
        att_df = att_df.assign(feature_color = 100)

    if 'weights' not in visualize_what:
        att_df.loc[:, 'weight'] = 1
    # features = ((features - features.mean()) / features.std()) * 15 + 50
    # red is 0, yellow 50, green 100

    if 'extra_tokens' not in visualize_what:
        extra_tokens = ['xxmaj', 'xxup', 'xxbos']  # most common tokenization artifacts, easy to "undo"
        att_df.loc[(att_df.word == "xxmaj").shift(1).fillna(False), 'word'] = att_df.word.str.capitalize()
        att_df.loc[(att_df.word == "xxup").shift(1).fillna(False), 'word'] = att_df.word.str.upper()
        att_df = att_df.loc[~att_df.word.isin(extra_tokens)]

    att_word_spans = [render_word(r.word, r.weight, r.feature, r.feature_color) for r in att_df.itertuples()]
    att_with_spaces = list(itertools.chain(*zip(att_word_spans, [' '] * len(att_word_spans))))
    return att_with_spaces


def setup_app(args: argparse.Namespace):

    if args.cpu:
        defaults.device = torch.device('cpu')
    model_dir = results_dir / args.run_id
    learner = load_learner(model_dir, 'learner.pkl')  # TODO: move paths etc to a config

    assert type(learner.model[1].attn) == sequence_aggregations.BranchingAttentionAggregation
    num_features = learner.model[1].attn.agg_dim
    featureNumberSlider.max = num_features - 1
    featureNumberSlider.marks = {m: str(m) for m in range(num_features)}
    numFeaturesSpan.children = num_features

    @app.callback(
        [Output('decisionDiv', 'children'),
        Output('probabilitiesGraph', 'figure'),
        Output('processedTextData', 'children')],
        [Input('submitButton', 'n_clicks')],
        state=[State(component_id='userText', component_property='value')]
    )
    def update_output_div(_, input_value):
        if not input_value:
            return render_decision(None), render_probabilities({}), '{}'
        decision, probas, att_df = process_sample(learner, input_value)
        return render_decision(decision), render_probabilities(probas), att_df.to_json()

# TODO: histogram of the sentiment
# TODO: choose between models


if __name__ == '__main__':
    args = AppConfig.from_console()
    print("Config:")
    print(args)
    setup_app(args)
    app.run_server(host=args.ip, port=args.port, debug=args.debug)
else:
    args = AppConfig.from_env()
    print("Config:")
    print(args)
    setup_app(args)
