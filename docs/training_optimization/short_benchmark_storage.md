# Short benchmark storage policy (functional change)

The test harness always uses `LINGBOT_BENCHMARK_SKIP_CHECKPOINT=1`,
`enable_resume=false`, `load_checkpoint_path=null`, `save_steps=0`,
`save_epochs=0`, `save_hf_weights=false`, and `async_save_hf_weights=false`.
Each short experiment starts from the same downloaded pretrained weights.

The training entry rejects resume or a distributed-checkpoint load path when
the benchmark flag is active. It disables periodic, epoch, terminal, DCP,
synchronous HF, and asynchronous HF saves, and skips creating the unused
checkpoint directory. The flag remains restricted to at most 1,000 updates.
The flag defaults off for ordinary training. This is a storage/measurement
change; it is not claimed as a steady compute speedup.

On 2026-10-12 the existing generated experiment outputs contained no saved
training-checkpoint payload. Sixteen remote and twelve local empty checkpoint
directories were removed. Downloaded student/teacher weights, raw profiling
files, configurations, training logs and timing evidence are retained.
