import pandas as pd
url=f"comparison.csv"
dataframe=pd.read_csv(url)
dataframe
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
app = Dash(__name__)
server= app.server
app.layout = html.Div([
    html.H1(children='RMSE comparison', style={'textAlign':'center'}),
    dcc.Dropdown(dataframe.id_H.unique(), '1', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])
@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    record = dataframe[dataframe.id_H==value]
    return px.bar(record,x='trial', y='RMSE',hover_data=['n_inputs', 'mlp_units', 'scaler_type', 'dropout', 'n_pool_kernel_size', 'pooling_mode', 'interpolation_mode', 'n_blocks'],color='scaler_type', title='RMSE comparison', orientation='v' )
if __name__ == '__main__':
    app.run(debug=True)
