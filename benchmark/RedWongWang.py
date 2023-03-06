import os, time, sys
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import *

# Minimal example:
n_regions = None  # total TVB brain regions
netpyne_nodes_inds = np.array([0, 1])  # the brain region nodes to place spiking networks from [0, n_regions-1] interval
# number of neurons per spiking population

# Interface basic configurations:
interface_model = "RATE"  # The only available option for NetPyNE so far
interface_coupling_mode = "spikeNet"  # "spikeNet" # "TVB"


def wong_wang(n_regions = None, netpyne_nodes_inds = np.array([0,1]), n_neurons = 100, interface_model = "RATE",
              interface_coupling_mode = "spikeNet"):
    work_path = os.getcwd()
    outputs_path = os.path.join(work_path, "outputs/NetPyNE_RedWongWang_%s_%s" % (interface_model,
                                                                                  interface_coupling_mode))

    config = Config(output_base=outputs_path)
    config.figures.SHOW_FLAG = True
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    config.figures.DEFAULT_SIZE= config.figures.NOTEBOOK_SIZE
    FIGSIZE = config.figures.DEFAULT_SIZE



    # This would run on TVB only before creating any multiscale cosimulation interface connections.

    from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

    # Reduced Wong-Wang model params

    model_params = {
        "G": np.array([2.0]),  # Global cloupling scaling
        "lamda": np.array([0.0]),  # Feedforward inhibition
        "w_p": np.array([1.4]),  # Feedback excitation
        "J_i": np.array([1.0]),  # Feedback inhibition
    }

    # Load and adjust connectivity
    from tvb.datatypes.connectivity import Connectivity

    # config.DEFAULT_CONNECTIVITY_ZIP = "/home/docker/packages/tvb_data/tvb_data/mouse/allen_2mm/ConnectivityAllen2mm.zip"
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)

    connectivity.configure()

    # -------------- Pick a minimal brain of only the first n_regions regions: ----------------
    if n_regions is not None:
        connectivity.number_of_regions = n_regions
        connectivity.region_labels = connectivity.region_labels[:n_regions]
        connectivity.centres = connectivity.centres[:n_regions]
        connectivity.areas = connectivity.areas[:n_regions]
        connectivity.orientations = connectivity.orientations[:n_regions]
        connectivity.hemispheres = connectivity.hemispheres[:n_regions]
        connectivity.cortical = connectivity.cortical[:n_regions]
        connectivity.weights = connectivity.weights[:n_regions][:, :n_regions]
        connectivity.tract_lengths = connectivity.tract_lengths[:n_regions][:, :n_regions]
    # -----------------------------------------------------------------------------------------

    # Remove diagonal self-connections:
    np.fill_diagonal(connectivity.weights, 0.0)

    # Normalize connectivity weights
    connectivity.weights /= np.percentile(connectivity.weights, 99)

    # -----------------------------------Or use the CoSimulator builder--------------------------------
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorSerialBuilder

    simulator_builder = CoSimulatorSerialBuilder()
    simulator_builder.config = config
    simulator_builder.model = ReducedWongWangExcIOInhI()
    simulator_builder.connectivity = connectivity
    simulator_builder.model_params = model_params
    simulator_builder.initial_conditions = np.array([0.0])

    simulator_builder.configure()
    simulator_builder.print_summary_info_details(recursive=1)

    simulator = simulator_builder.build()
    # -----

    simulator.configure()

    simulator.print_summary_info_details(recursive=1)

    # ------  Create basic configuration of NetPyNE network components -------------------
    from netpyne import specs

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






    # Build a NetPyNE network model with the corresponding builder
    from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import load_netpyne

    # from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    # sim_serial = load_pickled_dict(sim_serial_filepath)

    # Load NetPyNE
    netpyne = load_netpyne(config=config)


    # ------ Instantiating a non-opinionated NetPyNE network builder for this model, -----------------
    # ... and setting desired network description:


    from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder

    netpyne_model_builder = NetpyneNetworkBuilder(simulator, spiking_nodes_inds=netpyne_nodes_inds,
                                                  netpyne_instance=netpyne,
                                                  config=config)

    netpyne_model_builder.population_order = n_neurons
    netpyne_model_builder.tvb_to_spiking_dt_ratio = 2 # 2 NetPyNE integration steps for 1 TVB integration step
    netpyne_model_builder.monitor_period = 1.0

    # E and I population scale
    scale_e = 1.2
    scale_i = 0.4

    # Set populations:
    netpyne_model_builder.populations = [
        {
            "label": "E", # Population label, inherent to TVB mean-field model
            "model": cellModelLabel, # NetPyNE cell model
            "nodes": None,  # None means "all" -> building this population to all spiking_nodes_inds
            "params": {}, # custom parameters can go here
            "scale": scale_e # multiply netpyne_model_builder.population_order for the exact populations' size
        },
        {
            "label": "I",
            "model": cellModelLabel,
            "nodes": None,
            "scale": scale_i
        }
    ]

    # Set populations' connections within brain region nodes
    weight_ee = 0.015
    weight_ei = 0.01
    weight_ie = 0.01
    weight_ii = 0.01

    conn_spec_all_to_all = {"rule": "all_to_all"}
    conn_spec_prob_low = {"rule": {"prob": 0.01}}
    conn_spec_prob_high = {"rule": {"prob": 0.03}}

    netpyne_model_builder.populations_connections = [
        {
            "source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
            # ---------------- Possibly functions of spiking_nodes_inds --------------------------
            "conn_spec": conn_spec_prob_low,
            "weight": f'max(0.0, normal({weight_ee}, {weight_ee * 0.05}))', # NetPyNE connection weight
            "delay": 1,
            "receptor_type": receptor_type_E,
            # ---------------- Possibly functions of spiking_nodes_inds --------------------------
            "nodes": None}, # None means "all" -> performing this connection to all spiking_nodes_inds
        {
            "source": "E", "target": "I",  # E -> I
            "conn_spec": conn_spec_prob_low,
            "weight": weight_ei,
            "delay": 3,
            "receptor_type": receptor_type_E,
            "nodes": None},
        {
            "source": "I", "target": "E",  # I -> E
            "conn_spec": conn_spec_prob_high,
            "weight": weight_ie,
            "delay": 3,
            "receptor_type": receptor_type_I,
            "nodes": None},
        {
            "source": "I", "target": "I",  # I -> I
            "conn_spec": conn_spec_prob_low,
            "weight": weight_ii,
            "delay": 1,
            "receptor_type": receptor_type_I,
            "nodes": None}
    ]


    def tvb_weight_fun(source_node, target_node, lamda=None, sigma=0.1):
        weight = simulator.connectivity.weights[target_node, source_node]

        scale = netpyne_model_builder.global_coupling_scaling[0] * netpyne_model_builder.netpyne_synaptic_weight_scale
        if lamda:
            scale *= lamda
        weight *= scale
        return weight
        #return f"max(0.0, normal({weight}, {weight * sigma}))"

    def tvb_delay_fun(source_node, target_node, sigma=0.1):
        delay = simulator.connectivity.delays[target_node, source_node]
        return delay
        # low = (1-sigma) * delay
        # high = (1 + sigma) * delay
        # return f"uniform({low}, {high})"

    # lamda is the scale of E -> I connections between nodes
    lamda = simulator.model.lamda[0]

    netpyne_model_builder.nodes_connections = [
        {
            "source": "E", "target": "E",
            "conn_spec": conn_spec_prob_low,
            "weight": tvb_weight_fun,
            "delay": tvb_delay_fun,
            "receptor_type": receptor_type_E,
            "source_nodes": None, "target_nodes": None}  # None means "all"
        ]
    if lamda > 0:
        netpyne_model_builder.nodes_connections.append({
            "source": "E", "target": "I",
            "conn_spec": conn_spec_prob_low,
            # using lamda to scale connectivity weights (or alternatively, it can be used to downscale connection probability in 'conn_spec' above):
            "weight": lambda source_node, target_node: tvb_weight_fun(source_node, target_node, lamda),
            "delay": tvb_delay_fun,
            "receptor_type": receptor_type_E,
            "source_nodes": None, "target_nodes": None})

    # Set output recorder devices:
    from collections import OrderedDict
    connections = OrderedDict()
    connections["E"] = "E" # label <- target population
    connections["I"] = "I"
    params = config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
    # params["record_to"] = self.output_devices_record_to
    spike_recorder = {
        "model": "spike_recorder",
        "params": params,
        "connections": connections,
        "nodes": None}  # None means all here
    netpyne_model_builder.output_devices = [spike_recorder]


    # Set the simulation time:
    simulator.simulation_length = 350.0

    # This will be transferred to NetPyNE
    config.simulation_length = simulator.simulation_length

    netpyne_model_builder.configure(netParams, cfg, autoCreateSpikingNodes=True)

    def synaptic_weight_scale_func(is_coupling_mode_tvb):
        if is_coupling_mode_tvb: # "TVB"
            return 1e-2
        else: # "spikeNet"
            return 5

    netpyne_model_builder.global_coupling_scaling *= simulator.model.G
    netpyne_model_builder.netpyne_synaptic_weight_scale = synaptic_weight_scale_func(
        is_coupling_mode_tvb=interface_coupling_mode=="TVB")

    netpyne_network = netpyne_model_builder.build()

    # Configure NetpyneNetwork class:
    netpyne_network.configure()
    netpyne_network.print_summary_info_details(recursive=3, connectivity=True)







    # Build a TVB-NetPyNE interface with all the appropriate connections between the
    # TVB and NetPyNE modelled regions

    from tvb_multiscale.tvb_netpyne.interfaces.builders import TVBNetpyneInterfaceBuilder
    tvb_netpyne_interface_builder = TVBNetpyneInterfaceBuilder()  # non opinionated builder

    tvb_netpyne_interface_builder.config = config
    tvb_netpyne_interface_builder.tvb_cosimulator = simulator
    tvb_netpyne_interface_builder.spiking_network = netpyne_network
    # This can be used to set default tranformer and proxy models:
    tvb_netpyne_interface_builder.model = interface_model  # "RATE"
    tvb_netpyne_interface_builder.input_flag = True   # If True, NetPyNE->TVB update will be implemented
    tvb_netpyne_interface_builder.output_flag = True  # If True, TVB->NetPyNE coupling will be implemented
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / NetPyNE device for each spiking region,
    # "1-to-1" TVB->NetPyNE coupling.
    # If any other value, we need 1 "TVB proxy node" / NetPyNE device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in NetPyNE,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->NetPyNE coupling.
    tvb_netpyne_interface_builder.default_coupling_mode = interface_coupling_mode  # "spikeNet" # "TVB"

    # Number of neurons per population to be used to compute population mean instantaneous firing rates:
    tvb_netpyne_interface_builder.N_E = int(netpyne_model_builder.population_order * scale_e)
    tvb_netpyne_interface_builder.N_I = int(netpyne_model_builder.population_order * scale_i)

    tvb_netpyne_interface_builder.proxy_inds = netpyne_nodes_inds
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    tvb_netpyne_interface_builder.exclusive_nodes = True

    tvb_netpyne_interface_builder.synaptic_weight_scale_func = synaptic_weight_scale_func

    tvb_netpyne_interface_builder.output_interfaces = []
    tvb_netpyne_interface_builder.input_interfaces = []

    # # Using all default parameters for this example of an opinionated builder
    # tvb_netpyne_interface_builder.default_config()

    # or setting a nonopinionated builder:
    from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels

    if tvb_netpyne_interface_builder.default_coupling_mode == "TVB":
        proxy_inds = netpyne_nodes_inds
    else:
        proxy_inds = np.arange(simulator.connectivity.number_of_regions).astype('i')
        proxy_inds = np.delete(proxy_inds, netpyne_nodes_inds)


    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_netpyne_interface_builder.output_interfaces = [{
        'voi': np.array(["R_e"]),         # TVB state variable to get data from
        'populations': np.array(["E"]), # NetPyNE populations to couple to
        # --------------- Arguments that can default if not given by the user:------------------------------
        'model': tvb_netpyne_interface_builder.model, # This can be used to set default tranformer and proxy models
        'transformer_params': {'scale_factor': np.array([1.0])}, # due to the way Netpyne generates spikes, no scaling by population size is needed
        'proxy_params': {'number_of_neurons': tvb_netpyne_interface_builder.N_E},
        'receptor_type': receptor_type_E,
    }]

    if lamda > 0:
        tvb_netpyne_interface_builder.output_interfaces.append({
            'voi': np.array(["R_e"]),
            'populations': np.array(["I"]),
            # --------------- Arguments that can default if not given by the user:------------------------------
            'model': tvb_netpyne_interface_builder.model, # This can be used to set default tranformer and proxy models
            # due to the way Netpyne generates spikes, no scaling by population size is needed
            'transformer_params': {'scale_factor': np.array([1.0])},
            'proxy_params': {
                'number_of_neurons': tvb_netpyne_interface_builder.N_I,
                'lamda': lamda
            },
            'receptor_type': receptor_type_E,
        })

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for sVars, pop in zip([("S_e", "R_e"), ("S_i", "R_i")], ["E", "I"]):
        tvb_netpyne_interface_builder.input_interfaces.append(
            {'voi': sVars,
            'populations': np.array([pop]),
            'proxy_inds': netpyne_nodes_inds,
            # --------------- Arguments that can default if not given by the user:------------------------------
            # Set the enum entry or the corresponding label name for the "proxy_model",
            # options "SPIKES" (i.e., spikes per neuron), "SPIKES_MEAN", "SPIKES_TOTAL"
            # (the last two are identical for the moment returning all populations spikes together)
            'proxy_model': "SPIKES_MEAN",
            }
        )

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







    simulator = tvb_netpyne_interface_builder.build()

    # NetPyNE model is built in two steps. First need to create declarative-style specification for both spiking network itself and TVB-Netpyne proxy devides (interfaces).
    # Once it's done above using builders, network can be instantiated based on the specification:
    netpyne_network.netpyne_instance.instantiateNetwork()


    simulator.simulate_spiking_simulator = netpyne_network.netpyne_instance.run  # set the method to run NetPyNE

    # simulator.print_summary_info(recursive=3)
    # simulator.print_summary_info_details(recursive=3)

    print("\n\noutput (TVB->NetPyNE coupling) interfaces:\n")
    simulator.output_interfaces.print_summary_info_details(recursive=2)

    print("\n\ninput (NetPyNE->TVB update) interfaces:\n")
    simulator.input_interfaces.print_summary_info_details(recursive=2)




    # Configure the simulator with the TVB-NetPyNE interface...
    # ...and simulate!

    tic = time.time()

    print("Simulating TVB-NetPyNE...")
    simulator.configure()

    # Adjust simulation length to be an integer multiple of synchronization_time:
    if simulator.synchronization_time == 0:
        simulator.synchronization_time = 3.6
    simulator.simulation_length = \
                np.ceil(simulator.simulation_length / simulator.synchronization_time) * simulator.synchronization_time

    results = simulator.run()

    print("\nSimulated in %f secs!" % (time.time() - tic))

    return time.time() - tic

if __name__ == "__main__":
    runtime = wong_wang()
    print(runtime)