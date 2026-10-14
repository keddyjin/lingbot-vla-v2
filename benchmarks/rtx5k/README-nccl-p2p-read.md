# Optional NCCL_P2P_READ_ENABLE=1

Opt in from the repository root before running the existing SFT command:

```bash
source benchmarks/rtx5k/nccl_p2p_read.env.sh
# Run the original torchrun command with your existing YAML and paths.
```

The script only exports `NCCL_P2P_READ_ENABLE=1`. Explicit opt-in overrides an inherited value. It does not select devices, set model/data paths, alter batching or change training mathematics. To return to the NCCL default, use `unset NCCL_P2P_READ_ENABLE`. Existing NCCL settings, including MIN CTAs and NTHREADS, remain as set by the caller.

This option is specific to the measured single-node 8-GPU RTX Pro 5000 configuration with NCCL 2.27.3. Two independent short SFT pairs, including reversed execution order, measured throughput gains of **1.374527%** and **1.117855%** over genuinely unset READ with the five previously adopted controls fixed. Each run used 31 updates: warmup 1–10, observations 11–30, terminal update 31 excluded; micro batch 32, accumulation 1, global batch 256. Each of the four runs completed all 8 ranks with finite training scalars and no checkpoints. Two prior communication-operator runs passed 1,344 tensor comparisons across both settings. These checks are not a proof of long-term training convergence.

The environment setting requests the NCCL P2P read path where supported. Actual connection logs in these runs changed applicable P2P/CUMEM routes to P2P/CUMEM/read; SHM routes remained separate. This does not establish that every collective or every SendRecv message uses a read transport, or that all communication follows the same protocol or thread count. Other hardware, topology, NCCL versions and workloads need their own validation.
