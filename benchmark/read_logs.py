import pandas
import numpy

benchmarks = {}

#run 1 = [ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
#run 2 = [ 25, 75 , 125, 175, 225, 275, 325, 375, 425, 475]
for n_neurons in [ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    benchmark = pandas.read_csv("logs/benchmark_{}.log".format(n_neurons), dtype = {'time' : numpy.float32}, index_col = 0, names = ['time'])
    benchmarks[n_neurons] = { key: benchmark.filter(like = key, axis = 0) for key in ['run', 'getSpikes', 'stimulate'] }