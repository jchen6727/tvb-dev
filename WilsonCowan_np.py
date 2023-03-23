########################################################################################################################
"""
Import
"""
import os, time
import numpy as np
from collections import OrderedDict

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

########################################################################################################################
"""
code
"""
# Minimal example constants
# see connectivity.*
N_REGIONS = 4 # 4 TVB regions (/nodes): None means all (68) regions/nodes
N_NEURONS = 100 # 100 neurons per population
NETP_INDS = np.array([0, 1]) # NetPyNE simulator will handle regions/nodes 0 and 1 (of nodes [0, 1, 2, 3])

# Interface through "RATE" #TODO "SPIKES", "CURRENT" in NetPyNE
INTERFACE_MODEL = 'RATE' #TVB will give a rate, NetPyNE will generate spikes
INTERFACE_COUPLING_MODE = 'TVB' # Compute coupling as 1 node, or else use 'spikeNet' where one node assigned per proxy

work_path = os.getcwd()
outputs_path = os.path.join(
    work_path, "outputs/WilsonCowan_{}_{}".format(INTERFACE_MODEL, INTERFACE_COUPLING_MODE))

config = Config(output_base=outputs_path)
config.figures.SHOW_FLAG = True
config.figures.SAVE_FLAG = True
config.figures.FIG_FORMAT = 'png'
config.figures.DEFAULT_SIZE= config.figures.NOTEBOOK_SIZE
FIGSIZE = config.figures.DEFAULT_SIZE

# PLOT from tvb_multiscale.core.plot.plotter import Plotter
# PLOT plotter = Plotter(config.figures)

# For interactive plotting:
# %matplotlib notebook

# Otherwise:
# %matplotlib inline

########################################################################################################################
"""
TVB connectivity
TVB simulator
"""
# Wilson Cowan oscillatory regime
model_params = {
        "r_e": np.array([0.0]),
        "r_i": np.array([0.0]),
        "k_e": np.array([1.0]),
        "k_i": np.array([1.0]),
        "tau_e": np.array([10.0]),
        "tau_i": np.array([10.0]),
        "c_ee": np.array([10.0]),
        "c_ei": np.array([6.0]),
        "c_ie": np.array([10.0]),
        "c_ii": np.array([1.0]),
        "alpha_e": np.array([1.2]),
        "alpha_i": np.array([2.0]),
        "a_e": np.array([1.0]),
        "a_i": np.array([1.0]),
        "b_e": np.array([0.0]),
        "b_i": np.array([0.0]),
        "c_e": np.array([1.0]),
        "c_i": np.array([1.0]),
        "theta_e": np.array([2.0]),
        "theta_i": np.array([3.5]),
        "P": np.array([0.5]),
        "Q": np.array([0.0]),
        "shift_sigmoid": np.array([False])
    }

"""
Connectivity (minimal)
"""
connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
connectivity.configure()

connectivity.number_of_regions = N_REGIONS
connectivity.region_labels = connectivity.region_labels[:N_REGIONS]
connectivity.centres = connectivity.centres[:N_REGIONS]
connectivity.areas = connectivity.areas[:N_REGIONS]
connectivity.orientations = connectivity.orientations[:N_REGIONS]
connectivity.hemispheres = connectivity.hemispheres[:N_REGIONS]
connectivity.cortical = connectivity.cortical[:N_REGIONS]
connectivity.weights = connectivity.weights[:N_REGIONS][:, :N_REGIONS]
connectivity.tract_lengths = connectivity.tract_lengths[:N_REGIONS][:, :N_REGIONS]

# default, diagonal is already zeroed
# np.fill_diagonal(connectivity.weights, 0.0)

connectivity.weights /= np.percentile(connectivity.weights, 99)

########################################################################################################################
# Build cosimulator

simulator = CoSimulatorSerial()
simulator.model = WilsonCowan(**model_params)
simulator.connectivity = connectivity
simulator.coupling = Linear()
simulator.integrator = HeunStochastic()
simulator.integrator.dt = 0.1
simulator.integrator.noise.nsig = np.array([config.DEFAULT_NSIG, config.DEFAULT_NSIG])
simulator.initial_conditions = np.zeros((1, 2, simulator.connectivity.number_of_regions, 1))

mon_raw = Raw(period=1.0)
simulator.monitors = (mon_raw,)

simulator.configure()

simulator.print_summary_info_details(recursive=1)

# PLOT plotter.plot_tvb_connectivity(simulator.connectivity)

########################################################################################################################
netParams = specs.NetParams()

# NetPyNE specification for E and I synaptic mechanisms:
# see 'receptor_type' in populations_connections
exc_syn = 'exc'
inh_syn = 'inh'
netParams.synMechParams[exc_syn] = {'mod': 'Exp2Syn', 'tau1': 0.15, 'tau2': 1.5, 'e': 0}  # NMDA
netParams.synMechParams[inh_syn] = {'mod': 'Exp2Syn', 'tau1': 0.07, 'tau2': 9.1, 'e': -80}  # GABA

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

# Build a NetPyNE network model

# Load NetPyNE
netpyne = load_netpyne(config = config)

netpyne_model_builder = NetpyneNetworkBuilder(simulator, spiking_nodes_inds=NETP_INDS, netpyne_instance=netpyne, config=config)
netpyne_model_builder.population_order = N_NEURONS
netpyne_model_builder.tvb_to_spiking_dt_ratio = 2
netpyne_model_builder.monitor_period = 1.0

scale_e = 1.2
scale_i = 0.4

netpyne_model_builder.populations = [
    {
        "label": "E", # Population label, inherent to TVB mean-field model
        "model": cellModelLabel, # NetPyNE cell model
        "nodes": None,  # None means all regions specified for netpyne
        #"params": {}, # custom parameters ?
        "scale": scale_e # multiply netpyne_model_builder.population_order for the exact populations' size
    },
    {
        "label": "I",
        "model": cellModelLabel,
        "nodes": None,
        "scale": scale_i
    }
]

netpyne_model_builder.populations_connections = [
    {#E->E
        "source": "E", "target": "E",
        "conn_spec": {"rule": {"prob": 0.01}},
        "weight": 0.01,
        "delay": 1,
        "receptor_type": exc_syn,
        "nodes": None, # None <- "all": implements rule for all regions specified for netpyne
    },
    {#E->I
        "source": "E", "target": "I",
        "conn_spec": {"rule": {"prob": 0.01}},
        "weight": 0.01,
        "delay": 1,
        "receptor_type": exc_syn,
        "nodes": None, # implements rule for all regions specified for netpyne
    },
    {#I->E
        "source": "I", "target": "E",
        "conn_spec": {"rule": {"prob": 0.01}},
        "weight": 0.01,
        "delay": 1,
        "receptor_type": inh_syn,
        "nodes": None, # I->E
    },
    {#I->I
        "source": "I", "target": "I",
        "conn_spec": {"rule": {"prob": 0.01}},
        "weight": 0.01,
        "delay": 1,
        "receptor_type": inh_syn,
        "nodes": None,
    }
]

"""
connectivity weights source->target * global coupling scaling      * netpyne synaptic weight w/ tvb coupling mode
from DEFAULT_CONNECTIVITY_ZIP       * 0.00390625 (netpyne default) * 1e-2 (scaling with TVB)
"""
def tvb_weight_fun(source_node, target_node):
    return simulator.connectivity.weights[target_node, source_node] * 0.00390625 * 1e-2
def tvb_delay_fun(source_node, target_node):
    return simulator.connectivity.delays[target_node, source_node]

def synaptic_weight_scale_func(is_coupling_mode_tvb):
    return 1e-2

netpyne_model_builder.nodes_connections = [
    {
        'source': 'E', 'target': 'I',
        'conn_spec': {'rule': {'prob': 0.01}},
        'weight': tvb_weight_fun,
        'delay': tvb_delay_fun,
        'receptor_type': exc_syn,
        'source_nodes': None,
        'target_nodes': None
    }
]

connections = OrderedDict( (l,l) for l in ["E", "I"] )
spike_recorder = {
    "model": "spike_recorder",
    "params": config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy(),
    "connections": connections,
    "nodes": None
}

netpyne_model_builder.output_devices = [spike_recorder]

duration = 350
simulator.simulation_length = duration
config.simulation_length = duration

netpyne_model_builder.configure(netParams, cfg, autoCreateSpikingNodes=True)

netpyne_network = netpyne_model_builder.build()

netpyne_network.configure()
netpyne_network.print_summary_info_details(recursive=3, connectivity=True)

########################################################################################################################
# STEP 3. BUILD THE TVB-NETPYNE INTERFACE
# Build a TVB-NetPyNE interface with all the appropriate connections between the
# TVB and NetPyNE modelled regions
########################################################################################################################
tvb_netpyne_interface_builder = TVBNetpyneInterfaceBuilder()
tvb_netpyne_interface_builder.config = config
tvb_netpyne_interface_builder.tvb_cosimulator = simulator
tvb_netpyne_interface_builder.spiking_network = netpyne_network
tvb_netpyne_interface_builder.model = INTERFACE_MODEL

tvb_netpyne_interface_builder.input_flag = True   # If True, NetPyNE->TVB update will be implemented
tvb_netpyne_interface_builder.output_flag = True  # If True, TVB->NetPyNE coupling will be implemented

tvb_netpyne_interface_builder.default_coupling_mode = INTERFACE_COUPLING_MODE

# Number of neurons in excitatory and inhibitory populations
tvb_netpyne_interface_builder.N_E = int(netpyne_model_builder.population_order * scale_e)
tvb_netpyne_interface_builder.N_I = int(netpyne_model_builder.population_order * scale_i)

tvb_netpyne_interface_builder.proxy_inds = NETP_INDS

#tvb_netpyne_interface_builder.exclusive_nodes = True
# default is True

tvb_netpyne_interface_builder.synaptic_weight_scale_func = synaptic_weight_scale_func

proxy_inds = NETP_INDS

tvb_netpyne_interface_builder.output_interfaces = [{
    'voi': np.array(["R_e"]),         # TVB state variable to get data from
    'populations': np.array(["E"]), # NetPyNE populations to couple to
    # --------------- Arguments that can default if not given by the user:------------------------------
    'model': tvb_netpyne_interface_builder.model, # This can be used to set default tranformer and proxy models
    'transformer_params': {'scale_factor': np.array([1.0])}, # due to the way Netpyne generates spikes, no scaling by population size is needed
    'proxy_params': {'number_of_neurons': tvb_netpyne_interface_builder.N_E},
    'receptor_type': exc_syn,
}]

tvb_netpyne_interface_builder.input_interfaces = []
for sVars, pop in zip([("S_e", "R_e"), ("S_i", "R_i")], ["E", "I"]):
    tvb_netpyne_interface_builder.input_interfaces.append(
        {'voi': sVars,
        'populations': np.array([pop]),
        'proxy_inds': NETP_INDS,
        # --------------- Arguments that can default if not given by the user:------------------------------
        # Set the enum entry or the corresponding label name for the "proxy_model",
        # options "SPIKES" (i.e., spikes per neuron), "SPIKES_MEAN", "SPIKES_TOTAL"
        # (the last two are identical for the moment returning all populations spikes together)
        'proxy_model': "SPIKES_MEAN",
        }
    )

#TODO 6:91 Elephant or Wilson Cowan -->
tau_re = np.array([1.0])
tau_ri = np.array([1.0])
from tvb_multiscale.core.interfaces.base.transformers.models.red_wong_wang import \
    ElephantSpikesRateRedWongWangExc, ElephantSpikesRateRedWongWangInh
for interface, model, N, tau_s, tau_r, gamma in \
        zip(tvb_netpyne_interface_builder.input_interfaces,
            [ElephantSpikesRateRedWongWangExc, ElephantSpikesRateRedWongWangInh],
            [tvb_netpyne_interface_builder.N_E, tvb_netpyne_interface_builder.N_I],
            [simulator.model.tau_e, simulator.model.tau_i], [tau_re, tau_ri], [simulator.model.gamma_e, simulator.model.gamma_i]):
    interface["transformer_model"] = model
    interface["transformer_params"] = \
        {"scale_factor": np.array([1.0]) / N,
            "state": np.zeros((2, len(tvb_netpyne_interface_builder.proxy_inds))),
            "tau_s": tau_s, "tau_r": tau_r, "gamma": gamma}

# Configure and build:
tvb_netpyne_interface_builder.configure()
tvb_netpyne_interface_builder.print_summary_info_details(recursive=1)

# This is how the user defined TVB -> Spiking Network interface looks after configuration
print("\noutput (TVB->NetPyNE coupling) interfaces' configurations:\n")
print(tvb_netpyne_interface_builder.output_interfaces)

# This is how the user defined Spiking Network -> TVB interfaces look after configuration
print("\ninput (NetPyNE->TVB update) interfaces' configurations:\n")
print(tvb_netpyne_interface_builder.input_interfaces)

from tvb_multiscale.core.interfaces.base.transformers.models.models import Transformers

#TODO 6:100
from tvb_multiscale.core.interfaces.base.transformers.builders import \
    DefaultTVBtoSpikeNetTransformers, DefaultSpikeNetToTVBTransformers, \
    DefaultTVBtoSpikeNetModels, DefaultSpikeNetToTVBModels
from tvb_multiscale.tvb_netpyne.interfaces.builders import \
    TVBtoNetpyneModels, NetpyneInputProxyModels, DefaultTVBtoNetpyneModels, \
    NetpyneToTVBModels, NetpyneOutputProxyModels, DefaultNetpyneToTVBModels

def printe(enum):
    for name, member in enum.__members__.items():
        print("{}= {}".format(name, member.value))

print("Available TVB<->NetPyNE Coupling")
printe(TVBtoNetpyneModels)
printe(NetpyneToTVBModels)

print("Available TVB<->Spike Net I/O")
printe(DefaultTVBtoSpikeNetModels)
printe(DefaultTVBtoSpikeNetTransformers)

printe(DefaultSpikeNetToTVBTransformers)
printe(DefaultSpikeNetToTVBModels)


from tvb_multiscale.tvb_netpyne.interfaces.builders import TVBNetpyneInterfaceBuilder
tvb_spikenet_model_builder = TVBNetpyneInterfaceBuilder()

