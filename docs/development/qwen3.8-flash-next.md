# Qwen3.8-Flash-Next development TODO

This document tracks the remaining cross-backend work for the `qwen4exp` implementation. The dense QSA and serial GDN paths must remain available until each accelerated backend has a verified replacement.

## Common

- [ ] Rebase the branch onto current upstream llama.cpp and resolve the overlapping Qwen graph, model, context, backend-property, and test changes.
- [ ] Make the generated Qwen4Exp regressions selectable by backend instead of running the specialized coverage on CPU only.
- [ ] Run streamed and resident synthetic models through identical prefill and continuation checks on every supported backend.
- [ ] Verify the detached MTP sidecar through both accepted and rejected speculative branches while the target uses expert streaming.
- [ ] Decide whether same-model MTP should share a streamed expert cache or continue requiring a detached sidecar. A streamed model currently permits only one context.
- [ ] Add a Qwen4Exp persistent-prefix test that saves, restarts, restores, forks two sequences, and verifies attention, indexer, recurrent, and PLE state.
- [ ] Decide how persistent prefix restoration and speculative decoding will coexist. The server currently rejects this combination.

## Apple Silicon and Metal

- [ ] Add forced-Metal regressions for PLE chunk history, unified QSA lanes, recurrent rollback, MTP, expert streaming, and prefix restoration.
- [ ] Implement masked `MUL_MAT_ID` in both Metal expert matrix-vector and matrix-matrix paths. Sentinel routes must produce exact zeroes.
- [ ] Cover all, some, and no masked routes below and above Metal's 32-token MMID kernel crossover with the quantization types used by the released model.
- [ ] Advertise masked MMID support only after the operator tests pass, then verify expert waves no longer clamp the physical ubatch.
- [ ] Adapt a packed scalar-gate GDN kernel for `Dk = 128` while retaining the serial kernel for unsupported shapes and rollback modes.
- [ ] Extend the optimized Metal GDN path to `K > 1` only after the K=4 rollback oracle passes.
- [ ] Implement Metal QSA index selection and indexed flash attention. Keep dense QSA as the fallback until both operators pass long-context tests.
- [ ] Profile macOS expert reads and cache uploads. Evaluate uncached reads and direct reads into shared Metal storage after correctness is established.
- [ ] Record M1 Pro operator and end-to-end baselines at short context and 32K, including peak unified-memory use.

## NVIDIA CUDA

- [ ] Build every CUDA translation unit with CUDA 12.8 or newer for SM 12.0 (`120a-real`) on the RTX 5060 Ti.
- [ ] Add 10-route, model-sized masked-MMID cases for `IQ3_XXS` gate/up and `IQ4_NL` down tensors, including all, some, and no routes masked.
- [ ] Run masked MMID and recurrent GDN rollback under Compute Sanitizer.
- [ ] Run the generated multiwave Qwen4Exp fixture with CUDA graphs enabled and disabled.
- [ ] Verify a real streamed model with partial layer offload and a bounded GPU expert cache before increasing context length.
- [ ] Exercise buffered and direct expert I/O with 1, 4, and 16 workers, split GGUF shards, cache thrashing, cancellation, and teardown.
- [ ] Replace pageable synchronous cache uploads with pinned asynchronous staging if profiling shows PCIe upload stalls.
- [ ] Port QSA index selection and indexed flash attention to NVIDIA. Keep dense QSA until the CUDA sparse path matches the CPU oracle.
- [ ] Run the CUDA sparse QSA operators under Compute Sanitizer before enabling their capability checks.
- [ ] Port and tune chunked GDN for SM120, including dynamic shared-memory limits, launch geometry, occupancy, and rollback-tail handling.
- [ ] Run the detached Q8 MTP sidecar through accept and reject paths, and tune draft-layer offload for the available VRAM.
- [ ] Record 2.5K, 32K, and the largest fitting long-context baselines with prefill speed, decode speed, GPU activity, peak host memory, peak VRAM, and expert I/O counters.

## Change gate

For each backend change:

1. Add the regression before the implementation.
2. Compare the accelerated result with the CPU or existing dense/serial oracle.
3. Run the focused operator test, `test-llama-archs -a qwen4exp`, and a real-model smoke test.
4. Keep a safe capability-gated fallback and an environment escape hatch for new performance kernels.
5. Commit only after the focused verifier accepts the test and implementation.
