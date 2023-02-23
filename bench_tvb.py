"""

NetpyneModule class in tvb_multiscale.tvb_netpyne.netpyne.module ..
In stimulate() it polls all the input devices for instantaneous firing rates of TVB, generates spike trains based on it and feeds it to artificial cells that eventually fire those spikes (it is sort of modified NetStims). This method could be profiled as a whole or by parts.
in getSpikes() it obtains all spikes fired by netpyne network during the last period to pass it back to TVB
and run() contains the netpyne simulation itself (and from here we trigger the above-mentioned stimulate())

"""

"""
Import
"""
import os, time
import numpy as np

"""
tvb root imports
"""
from tvb.basic.profile import TvbProfile
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray
from tvb.datatypes.connectivity import Connectivity


from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import Linear
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold, EEG
"""
tvb multiscale core imports
"""
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorSerialBuilder
from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.interfaces.base.transformers.models.red_wong_wang import \
    ElephantSpikesRateRedWongWangExc, ElephantSpikesRateRedWongWangInh
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.data_analysis.spiking_network_analyser import SpikingNetworkAnalyser

from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
"""
tvb multiscale netpyne import
"""
from tvb_multiscale.tvb_netpyne.config import *
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import load_netpyne
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
from tvb_multiscale.tvb_netpyne.interfaces.builders import TVBNetpyneInterfaceBuilder

"""
netpyne import
"""
from netpyne import specs

from tvb_multiscale.tvb_netpyne.netpyne.module import NetpyneModule

NetpyneModule.stimulate()

NetpyneModule.getSpikes()

model_params = {
    'G': np.array([2.0]),
    'lambda': np.array([0.0]),
    'w_p': np.array([1.4]),
    'J_i': np.array([1.0]),
}

connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
connectivity.configure()

connectivity.weights /= np.percentile(connectivity.weights, 99)

simulator_builder = CoSimulatorSerialBuilder()
simulator_builder.config = config
simulator_builder.model = ReducedWongWangExcIOInhI()
simulator_builder.connectivity = connectivity
simulator_builder.model_params = model_params
simulator_builder.initial_conditions = np.array([0.0])

simulator_builder.configure()
simulator_builder.print_summary_info_details(recursive=1)

simulator = simulator_builder.build()

simulator.configure()
simulator.print_summary_info_details(recursive=1)


plotter = Plotter(config.figures)

netParams = specs.NetParams()

# NetPyNE specification for E and I synaptic mechanisms:
receptor_type_E = 'exc'
receptor_type_I = 'inh'
netParams.synMechParams[receptor_type_E] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 1.5, 'e': 0}  # NMDA
netParams.synMechParams[receptor_type_I] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}  # GABA

# Basic description of Hogkin-Huxley single-compartment cell in NetPyNE:
cellModelLabel = 'PYR'
PYRcell = {'secs': {}}
PYRcell['secs']['soma'] = {'geom': {}, 'mechs': {}}  # soma params dict
PYRcell['secs']['soma']['geom'] = {'diam': 18.8, 'L': 18.8, 'Ra': 123.0}  # soma geometry
PYRcell['secs']['soma']['mechs']['hh'] = {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}  # soma hh mechanism
netParams.cellParams[cellModelLabel] = PYRcell

cfg = specs.SimConfig()
cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.1