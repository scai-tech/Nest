# default network configs: Overwritten by input. 
num_accelerators = 1024
bandwidths_per_level = [900, 100, 400] #GBps
devices_per_level = [8, 4, 32]
latency_per_level = [0.0001,0.0002,0.0003] #s
setup_name = "default" #
bws = bandwidths_per_level.copy()
bws = [x*1024*1024*1024 for x in bandwidths_per_level]

baseline_configs = ["Manual", "Phaze", "MCMC"]
use_ilp = False

#H100s
# bandwidths_per_level = [900, 12.5, 50] #GBps
# devices_per_level = [8, 4, 32]

import os, json

def set_config(args):
    global num_accelerators, bandwidths_per_level, bws, devices_per_level, latency_per_level, setup_name, use_ilp
    if args.num_accelerators is not None:   
        num_accelerators = args.num_accelerators
    if args.bandwidths_per_level is not None:
        bandwidths_per_level = args.bandwidths_per_level
        bws = bandwidths_per_level.copy()
        bws = [x*1024*1024*1024 for x in bandwidths_per_level]
    if args.devices_per_level is not None:
        devices_per_level = args.devices_per_level
    if args.latency_per_level is not None:
        latency_per_level = args.latency_per_level
    if args.setup_name is not None:
        setup_name = args.setup_name
    if args.use_ilp is not None: 
        use_ilp = args.use_ilp 

    # check if produce of device per level is equal to total accelerators
    product = 1
    for d in devices_per_level: 
        product *= d
    assert product == num_accelerators, "Product of devices per level must be equal to total number of accelerators"
    # check length of bandwidths, device, and latency are the same
    assert len(bandwidths_per_level) == len(devices_per_level) == len(latency_per_level), "Number of levels of bandwidths, devices, and latency per level must be the same"

    print("Setup complete. Config set to: Name: " , setup_name, " num_accelerators: ", num_accelerators, 
        " Bandwidth: ", bandwidths_per_level,  " Devices: ", devices_per_level, " Latency: ", latency_per_level, "using ILP? ", use_ilp)

def set_baselines(args):
    global baseline_configs
    if args.baseline_configs is not None:
        baseline_configs = args.baseline_configs