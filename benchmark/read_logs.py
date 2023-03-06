import pandas
import numpy

def get_times(dataframe, key):
    return dataframe.filter(like = key, axis = 0)['time'].to_numpy()
benchmarks = {}

#run 1 = [ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
#run 2 = [ 25, 75 , 125, 175, 225, 275, 325, 375, 425, 475]

for n_neurons in [ 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]:
    benchmark = pandas.read_csv("logs/benchmark_{}.log".format(n_neurons), dtype = {'time' : numpy.float32}, index_col = 0, names = ['time'])
    benchmarks[n_neurons] = { key: get_times(benchmark, key) for key in ['run', 'getSpikes', 'stimulate'] }

