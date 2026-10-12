# RTX PRO 5000 benchmark setup (functional)

The new experiment root is `/root/keddyj/LingBotVLA-Test` on
`root@10.6.142.11`. The timing-output guard is updated to that authorized root;
the previous H20 root is not used by these experiments. The guard still rejects
relative paths and resolved paths outside the experiment root. This change
does not change training mathematics and has no performance claim.

The branch is `keddyj-codex/sft-opt-rtx5k`. Use the supplied Distributed Muon
configuration with micro batch 32 and accumulation 1. Global batch is 128 on
four GPUs and 256 on eight GPUs. Establish separate performance anchors for
each GPU count and revalidate all H20 performance switches on Blackwell.

Short experiments enable `LINGBOT_BENCHMARK_SKIP_CHECKPOINT=1`, set
`enable_resume=false`, and omit `load_checkpoint_path` from YAML. The existing
short-benchmark guard disables periodic and final DCP/HF saves and restricts
each run to at most 1000 updates. Pretrained input and teacher weights are
retained. Each profiler run retains its source commit, actual diff, raw trace,
and analysis. Benchmark preparation/download time is tracked separately from
completed real training.

Set `LINGBOT_BENCHMARK_STOP_STEPS=31` for a short run while keeping the official
`train.max_steps=50000`. The stop override affects progress and termination
only; the LR scheduler still reads the configured production training horizon.
It requires checkpoint-skipping mode, must be positive and at most 1000, and
cannot exceed `train.max_steps`. With the override unset, existing training and
short-benchmark behavior is preserved. This is functional experiment control,
not an optimization or performance claim.
