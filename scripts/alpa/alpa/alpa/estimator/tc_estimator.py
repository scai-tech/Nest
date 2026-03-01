import sys

from alpa.estimator.sunstone.examples.simba_like.cnn_optimizer_wrapper import CNNOptimizer  # noqa

import time


def tensor_core_estimate(setupconfig, op, energy):

    start = time.time()

    memrs_flat = {
        "reg": setupconfig["L1_Buffer_TC"],
        "L1_ifmap": setupconfig["L2_Buffer"],
        "L1_w": setupconfig["L2_Buffer"],
        "L1_ofmap": setupconfig["L2_Buffer"],
        "L2": setupconfig["GLB_Buffer"],
    }

    # ******* this is the memory hierarchy format that our wrapper accepts
    #         (look into cnn_optimizer_wrapper)                            *******
    acc_memrs = {
        "reg": memrs_flat["reg"],
        "L1": (memrs_flat["L1_ifmap"], memrs_flat["L1_w"], memrs_flat["L1_ofmap"]),
        "L2": memrs_flat["L2"]
    }

    # ****** Access energies that can come from Accelergy
    #        (CACTI + ALADDIN) or any other source        ******
    acc_acs_enrgs = {
        # Register inside vector MAC
        "reg_ifmap": 0,
        "reg_w": 0,
        "reg_ofmap": 0,
        # PE SRAM
        "L1_ifmap":0,
        "L1_w": 0,
        "L1_ofmap": 0,
        "L1_NoC": 0,
        # Global SRAM
        "L2_ifmap": 0,
        "L2_w": 0,
        "L2_ofmap": 0,
        # Global SRAM
        "DR_ifmap": 0,
        "DR_w": 0,
        "DR_ofmap": 0,
    }

    # ******* dimensions of the vector MAC *******
    pe_spatial = {"X": setupconfig["PE_x"], "Y": setupconfig["PE_y"]}

    # ******* dimensions of the PE array *******
    cores_sptl = {"X": setupconfig["Core_x"], "Y": setupconfig["Core_y"]}

    # BW (words/cycle)
    smba_bws = [None, (64, 64), (256, 256), (128, 128)]

    # ******* this list indicate whether each operand has
    #         a dedicated buffer at each level & will be
    #         used to optimize for uneven mapping which Timeloop
    #         also supports. If uneven mapping will not be used,
    #         set this to None                                   *******
    sprt_strct = [(False, True, False), (True, True, True),
                  (False, False, False)]

    # ******* the tensor to keep inside vector MAC registers
    #         here we by-pass both IFMAP and OFMAP and only keep weights
    mem_bypass = {
        "reg": (True, False, True),
        "L1": (False, False, False),
        "L2": (False, False, False)}

    (
        b,  # batch size
        n, # output dim[0]
        m,  # output dim [1]
        c, # weight dim[0]
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    ) = op

    '''dim = {
        "N": n, #B
        "C": c, #N
        "M": m, #C
        "R": r, #r
        "S": s, #s
        "P": p, #p
        "Q": q, #q
    }'''

    # transfering to Conv notation
    dim = {
        "N": n * b,  # batch size
        "C": c,    # input channels
        "K": m,  #output channels
        "R": r,  # r
        "S": s,  # s
        "P": p,  # p
        "Q": q,  # q
    }

    opt = CNNOptimizer(
        dim,
        acc_memrs,
        acc_acs_enrgs,
        smba_bws,
        cores_sptl,
        pe_spatial,
        mem_bypass,
        sprt_strct,
        memrs_flat
    )

    try:
        fnl_res, full_cost = opt.solve(threads=8, optimize_uneven_mapping=False)
        tls, unevn_mp, ords, cost = fnl_res
        end = time.time()
        return (full_cost[1], full_cost[2], end - start)

    except:
        return None
