from RedWongWang import wong_wang
import logging
import os

env_vars = ['N_NEURONS']
vars = {key: int(os.getenv(key)) for key in env_vars}

logger = logging.getLogger('benchmark')
logh = logging.FileHandler("logs/benchmark_{}.log".format(vars['N_NEURONS']))
logh.setLevel(200)
logh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(logh)

wong_wang(n_neurons = vars['N_NEURONS'])
