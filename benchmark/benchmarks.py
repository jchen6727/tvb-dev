import os, subprocess

env = os.environ.copy()

for n_neurons in [ 25, 75 , 125, 175, 225, 275, 325, 375, 425, 475]:
    env['N_NEURONS'] = str(n_neurons)
    print("running with n_neurons = {}".format(n_neurons))
    proc = subprocess.run(['python', 'benchmark.py'], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(proc.stderr)