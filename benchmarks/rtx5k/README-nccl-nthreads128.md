# Optional NCCL_NTHREADS=128

Opt in from the repository root before running the existing SFT command:

```bash
source benchmarks/rtx5k/nccl_nthreads128.env.sh
# Run the original torchrun training command with your existing YAML and paths.
```

The script only exports `NCCL_NTHREADS=128`; it does not select GPU devices, alter batch size, set model/data paths or modify training mathematics. To return to the NCCL default, use `unset NCCL_NTHREADS`. Explicit opt-in overrides an inherited NTHREADS setting.

This is a single-node scheduling experiment for the measured 8-GPU RTX Pro 5000 configuration. Effective kernel thread count depends on the NCCL algorithm/protocol/topology; the variable does not prove every message launches 128 threads. Other hardware and workloads require their own benchmark.
