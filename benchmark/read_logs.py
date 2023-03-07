import pandas
import numpy
from plotly.subplots import make_subplots
import plotly.graph_objects as graph_objects

def get_times(dataframe, key):
    return dataframe.filter(like = key, axis = 0)['time'].to_numpy()
benchmarks = {}
s = []

runs = [ 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
funcs = ['run', 'getSpikes', 'stimulate']

for n_neurons in runs:
    benchmark = pandas.read_csv("logs/benchmark_{}.log".format(n_neurons), dtype = {'time' : numpy.float32}, index_col = 0, names = ['time'])
    benchmarks[n_neurons] = { key: get_times(benchmark, key) for key in funcs }

    sd = {'n_neurons': n_neurons}
    sd.update({key: benchmarks[n_neurons][key].mean() for key in funcs})
    s.append( pandas.Series(sd) )

df = pandas.concat(s)

fig = make_subplots(rows = len(funcs), cols = 1)

for i, func in enumerate(funcs):
    fig.add_trace( graph_objects.Scatter( x = df['n_neurons'], y = df[func], name=func ), row=i+1, col=1 )

fig.show()


