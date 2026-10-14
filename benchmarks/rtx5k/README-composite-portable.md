# Optional normal / DataLoader / NUMA settings

The model change skips the unused MoGe normal output only when
`LINGBOT_MOGE_SKIP_UNUSED_NORMAL=1`. The default is `0`; depth, points, mask,
projection and the consumed alignment targets retain their original path.

In your existing training configuration, set `data.prefetch_factor: 2` to
reduce DataLoader read-ahead. The supplied fragment is one key to merge, not a
standalone training configuration. Keep the existing batch, accumulation,
worker, pin-memory, seed and model/data path settings.

The optional standard-library NUMA entry binds each worker before Torch imports
and then execs the original trainer. It preserves Python flags, trainer argv,
the working directory and environment. It does not bind memory, change thread
counts, initialize CUDA or write experiment logs.

```bash
# Run from the repository checkout in your prepared Python environment.
# Set CUDA_VISIBLE_DEVICES to the desired full GPU UUIDs in CUDA rank order.
# CFG points to your existing complete configuration with prefetch_factor=2.
export LINGBOT_MOGE_SKIP_UNUSED_NORMAL=1
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" \
  benchmarks/numa_node_entry.py --cpu-affinity node -- \
  tasks/vla/train_lingbotvla.py "$CFG"
```

Node mode requires complete, unique `GPU-...` UUIDs in CUDA_VISIBLE_DEVICES and
a matching LOCAL_WORLD_SIZE. It reads the driver's UUID/PCI mapping and Linux
sysfs NUMA CPU lists. The full selected node must be inside the worker's
original CPU authorization. Unknown NUMA nodes, ambiguous device identifiers,
numeric CUDA ordinals, partial-node CPU authorization and pre-existing Torch
imports/threads are rejected. There are no fixed GPU identities, node counts,
CPU IDs or model/checkpoint paths in the entry.

Use the same entry with `--cpu-affinity off`, normal flag `0` and prefetch
factor `4` for a fresh control. OFF never calls the affinity setter or reads GPU
topology. This is an explicit three-setting combination; individual gains must
not be added. Existing FSDP/optimizer/precision controls and their source-audit
requirements remain configured through their original training entry.

The entry is supported on Linux with the NVIDIA `/proc` and PCI/NUMA sysfs
interfaces. Portability of input parsing does not establish performance on an
untested machine. Numerical, process-inheritance and same-source performance
qualification for a new entry must be recorded separately from older scripts.
