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

########################################################################################################################
"""
code
"""
# Minimal example constants
# see connectivity.*
N_REGIONS = 68
N_NEURONS = 100
NODES_INDS = np.array([0, 1])

# Interface through "RATE" #TODO "SPIKES", "CURRENT" in NetPyNE
INTERFACE_MODEL = 'RATE'
INTERFACE_COUPLING_MODE = 'TVB'

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
cfg = specs.SimConfig()
netpyne = load_netpyne(config = config)

netpyne_model_builder = NetpyneNetworkBuilder(simulator, spiking_nodes_inds=NODES_INDS, netpyne_instance=netpyne, config=config)

# Set the simulation time:
simulator.simulation_length = 350.0

# This will be transferred to NetPyNE
config.simulation_length = simulator.simulation_length

netpyne_model_builder.configure(netParams, cfg, autoCreateSpikingNodes=True)
