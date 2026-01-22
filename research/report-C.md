
# Turning a Sequential Hash into a Throughput Engine

## Executive Summary

The optimization of your 6-stage, 18-operation hash function hinges on transforming the workload from a latency-bound sequential problem into a throughput-bound parallel one. While a single hash instance is constrained by its 18-cycle dependency chain, the independence of the 4096 hashes allows for massive parallelism. The optimal strategy is **Vector-Interleaved Scheduling**: using AVX-512 to process 16 hashes per instruction, combined with a 4-way interleave to keep 64 hashes in flight per core. This approach hides the dependency latency and saturates your hardware's 6 VALU slots.

### Key Strategic Insights
* **Vector-Interleaving Breaks the Latency Wall**: A single hash cannot run faster than its dependency chain (RecMII ≈ 18 cycles). By interleaving 4 independent AVX-512 vectors, you expose enough instruction-level parallelism (ILP) to fill the execution pipeline, effectively reducing the cost per hash to approximately 1.125 cycles, or even lower on wider cores.
* **AoSoA Layout Eliminates Gather Bottlenecks**: Standard Array-of-Structures (AoS) layouts force slow `gather` instructions. Transforming data to an Array-of-Structures-of-Arrays (AoSoA) layout enables efficient, contiguous `vmovdqa` loads, which is critical for feeding the vector units.
* **Shift Units Are the Hardware Choke Point**: Unlike logical operations, vector shifts often have limited execution ports (e.g., only 2 ports on AMD Zen 4 vs. 4 for adds). Scheduling must account for this asymmetry to avoid port contention.
* **AVX-512 Performance is Generation-Specific**: On older Skylake-SP cores, AVX-512 causes severe frequency throttling. On modern Ice Lake and Zen 4/5 architectures, penalties are negligible, making AVX-512 the clear performance winner.

## Problem Framing: 18-Op Chains vs. 4096 Independent Hashes

The fundamental challenge is the **Recurrence-constrained Minimum Initiation Interval (RecMII)**. Your hash function has a loop-carried dependency chain of 18 operations. If executed sequentially, the CPU must wait for the result of stage $N$ before starting stage $N+1$, leaving your 12 ALU and 6 VALU slots idle for the vast majority of cycles.

### The Hardware Budget: 12 ALU, 6 VALU
Your target hardware is exceptionally wide. To utilize 6 VALU slots per cycle with an 18-op chain, you theoretically need to issue instructions for $6 \times 18 = 108$ operations every 18 cycles. A single hash stream only provides 1 operation at a time.
* **The Solution**: You must have at least $6 \times \text{Latency}$ independent operations ready to issue.
* **The Lever**: The 4096 hashes are independent. By processing them in parallel, you decouple the throughput from the single-chain latency.

## Core Strategy: 16-Lane SIMD × 4-Way Interleaving

The most effective technique is to combine **SIMD parallelism** (spatial) with **Instruction Interleaving** (temporal).

### Scheduling Plan for Saturation
Using AVX-512, you process 16 hashes per vector. However, a single stream of dependent vector instructions will still stall due to latency. The solution is to interleave 4 independent vector streams within the same loop body.
* **Stream A**: Computes Stage 1 for hashes 0-15.
* **Stream B**: Computes Stage 1 for hashes 16-31.
* **Stream C**: Computes Stage 1 for hashes 32-47.
* **Stream D**: Computes Stage 1 for hashes 48-63.

By the time the CPU returns to Stream A for Stage 2, the latency for Stage 1 has elapsed, allowing execution without stalls. This keeps 64 hashes in flight simultaneously.

### Register Allocation to Avoid Spills
AVX-512 provides 32 ZMM registers (512-bit), which is a decisive advantage over AVX2's 16 registers. A 4-way interleaved kernel requires:
* **4 Registers**: Current state accumulators (Streams A, B, C, D).
* **1 Register**: Broadcasted `const1`.
* **8 Registers**: Temporaries for intermediate results (2 per stream).
* **Total**: ~13 Registers.
This fits comfortably within the 32 available ZMMs, ensuring zero register spills to memory.

### Performance Model
| Metric | Estimate | Basis |
| :--- | :--- | :--- |
| **Throughput** | ~0.375 cycles/hash | 3 vector ops/cycle throughput on modern cores |
| **Latency Hiding** | 100% | 4-way interleave covers >18 cycle chain depth |
| **Register Pressure** | Low | Uses ~13 of 32 ZMM registers |

## Data Layout Transformation: AoSoA

Naive Array-of-Structures (AoS) layouts kill SIMD performance because loading `a` requires gathering 32-bit values from non-contiguous memory.

### The AoSoA Advantage
You must restructure the 4096 hashes into an **Array of Structures of Arrays (AoSoA)**.
* **Structure**: Group hashes into blocks of 16 (matching the AVX-512 width).
* **Memory Access**: This allows a single `vmovdqa32` instruction to load 16 aligned, contiguous hash states.
* **Efficiency**: This layout ensures 100% cache line utilization and triggers hardware prefetchers effectively.

### Transposition Strategy
If the input data arrives in AoS format, perform a one-time transposition. Optimized AVX-512 transpose kernels (using `vshufi32x4`, `vpermi2d`) can be over 37x faster than scalar copying. The cost of this pre-processing is negligible compared to the gains in the main hash loop.

## Microarchitectural Scheduling: The Shift Bottleneck

While your machine has 6 VALU slots, they are likely not identical. Vector shift operations (`vpsll`, `vpsrl`) often run on fewer ports than logicals (`vpand`, `vpxor`) or adds (`vpadd`).

### Port Pressure Analysis
* **Intel Skylake**: Vector shifts are restricted to ports 0 and 5, while logicals can use ports 0, 1, and 5. Heavy shift usage can bottleneck port 5.
* **AMD Zen 4**: Has 4 pipes for logical/add operations but only 2 for vector shifts. This creates a 2:1 throughput disparity.

**Mitigation**: When interleaving, ensure that shift instructions from different streams are spaced out to avoid saturating the shift ports. Use performance counters (e.g., `RESOURCE_STALLS.ANY`) to verify if shifts are the limiting factor.

## ISA-Level Algebraic Fusion

You can reduce the instruction count by exploiting specific instruction set features that fuse operations.

### x86 LEA Optimization
The pattern `a + (a << k) + const` can often be fused into a single x86 `LEA` instruction: `LEA reg, [reg + reg*scale + const]`.
* **Constraints**: `k` must be 0, 1, 2, or 3 (scaling by 1, 2, 4, 8).
* **Latency**: On Ice Lake and Zen 4, this is a fast 1-cycle operation. On older Skylake, complex LEAs can take 3 cycles.

### AArch64 Shifted Operands
If targeting ARM, the instruction set supports "shifted register" operands.
* **Fusion**: `a + (a << k)` becomes `ADD wD, wN, wM, LSL #k`.
* **Flexibility**: Supports shift amounts `k` from 0 to 31, far more flexible than x86.

**Safety Note**: Only reassociate operations (e.g., `(a + b) + c` $\to$ `a + (b + c)`) if the operators are identical (all `+` or all `^`). Mixed operators (ARX) are non-linear and cannot be reordered without breaking bit-exactness.

## Bitslicing: A High-Throughput Alternative

Bitslicing (SWAR) is a radical alternative where you transpose the data to operate on 32 "bit-planes" of 4096 hashes simultaneously.

### Trade-offs
* **Pros**: Bitwise operations (XOR, AND, OR) are extremely fast. Constant shifts become "free" (just renaming registers).
* **Cons**: Addition is expensive. You must implement a software adder (ripple-carry or parallel-prefix), which takes 2-5x more instructions than a native `ADD`.
* **Verdict**: For your workload, bitslicing is viable if the hash is dominated by XORs and shifts. If additions are frequent, the overhead of software adders likely outweighs the benefits compared to AVX-512.

## Compiler & Autovectorization Playbook

Auto-vectorization is fragile for complex hash loops. You must explicitly guide the compiler.

### Key Flags and Pragmas
* **Clang/LLVM**: `-O3 -mavx512f -Rpass=loop-vectorize`. Use `#pragma clang loop vectorize(enable)`.
* **GCC**: `-O3 -march=native -fopt-info-vec`. Use `#pragma GCC ivdep`.
* **Intel ICX**: `-O3 -xCORE-AVX512 -qopt-zmm-usage=high`.
* **Code Structure**: Use `restrict` pointers to promise no aliasing. Ensure loop bounds are constant and known.

## GPU Implementation Strategy

For maximum raw throughput, a GPU is superior.
* **Mapping**: Assign 1 thread per hash. Launch a grid of 4096 threads (e.g., 16 blocks of 256 threads).
* **Memory**: Store `const1` and `shift_amount` in `__constant__` memory to broadcast them to all threads in a warp.
* **Throughput**: An NVIDIA A100 can execute ~9.7 Trillion Integer Ops/Sec, dwarfing any CPU.

## AVX-512 Frequency & Register Trade-offs

The decision to use AVX-512 depends on the CPU generation due to frequency throttling.

| Architecture | Throttling Risk | Recommendation |
| :--- | :--- | :--- |
| **Skylake-SP / Cascade Lake** | **High** (Level 1/2 licenses) | Use AVX2 unless isolated |
| **Ice Lake Server / Cooper Lake** | **Low** | Recommended |
| **Ice Lake Client / Rocket Lake** | **Negligible** | Strongly Recommended |
| **AMD Zen 4** | **None** (Double-pumped) | Benchmark (Register win) |
| **AMD Zen 5** | **None** (Native 512b) | Strongly Recommended |

**Register Advantage**: Even if frequency drops, AVX-512's 32 registers (vs. AVX2's 16) prevent spills in your complex interleaved kernel, often yielding a net win.

## Portable SIMD Framework: Google Highway

To write a single kernel that targets AVX2, AVX-512, and ARM NEON/SVE efficiently, **Google Highway** is the recommended framework.
* **Why**: It supports **dynamic SVE vector lengths**, which competitors like xsimd and EVE often lack (supporting only fixed sizes).
* **Performance**: It maps to native intrinsics and includes a runtime dispatcher to pick the best ISA at startup.

## Verification & Benchmarking

### Bit-Exact Verification
* **Golden Reference**: Write a scalar C implementation using `uint32_t` to enforce modulo $2^{32}$ arithmetic.
* **Shift Safety**: C++ shifts $\ge 32$ are Undefined Behavior (UB). x86 masks shifts to 5 bits; ARM/AVX might zero them. You **must** mask shift amounts (`shift & 31`) to ensure cross-platform identity.

### Metrics
* **Cycles/Hash**: Measure using `CPU_CLK_UNHALTED.THREAD`.
* **Hashes/Sec**: Measure wall-clock time for the batch.

## Synthesized Kernel Blueprint

Below is the blueprint for the optimized AVX-512 interleaved kernel.

```c
#include <immintrin.h>

// Assumes data is in AoSoA format: struct { uint32_t a[16]; } blocks[256];
// Assumes op1=ADD, op2=XOR, shift_op=left shift

void hash_kernel_avx512_interleaved(AoSoA_Block* data, uint32_t const1, int shift_amount) {
 const size_t num_total_blocks = 256;
 const int interleaving_factor = 4;

 // Pre-broadcast constant to a vector register
 __m512i zmm_c1 = _mm512_set1_epi32(const1);

 // Main loop processes 4 blocks (64 states) per iteration
 for (size_t i = 0; i < num_total_blocks; i += interleaving_factor) {
 // 1. Load states for 4 independent streams (AoSoA layout enables this)
 __m512i zmm_a0 = _mm512_load_si512(&data[i + 0].a);
 __m512i zmm_a1 = _mm512_load_si512(&data[i + 1].a);
 __m512i zmm_a2 = _mm512_load_si512(&data[i + 2].a);
 __m512i zmm_a3 = _mm512_load_si512(&data[i + 3].a);

 // 2. Unroll all 6 hash stages with interleaved instructions
 #pragma unroll
 for (int stage = 0; stage < 6; ++stage) {
 // Compute independent intermediate results for all 4 streams first
 // This breaks the dependency chain by filling the pipeline
 __m512i tmp_add0 = _mm512_add_epi32(zmm_a0, zmm_c1);
 __m512i tmp_shl0 = _mm512_slli_epi32(zmm_a0, shift_amount);

 __m512i tmp_add1 = _mm512_add_epi32(zmm_a1, zmm_c1);
 __m512i tmp_shl1 = _mm512_slli_epi32(zmm_a1, shift_amount);

 __m512i tmp_add2 = _mm512_add_epi32(zmm_a2, zmm_c1);
 __m512i tmp_shl2 = _mm512_slli_epi32(zmm_a2, shift_amount);

 __m512i tmp_add3 = _mm512_add_epi32(zmm_a3, zmm_c1);
 __m512i tmp_shl3 = _mm512_slli_epi32(zmm_a3, shift_amount);

 // Final XOR to update state registers
 zmm_a0 = _mm512_xor_si512(tmp_add0, tmp_shl0);
 zmm_a1 = _mm512_xor_si512(tmp_add1, tmp_shl1);
 zmm_a2 = _mm512_xor_si512(tmp_add2, tmp_shl2);
 zmm_a3 = _mm512_xor_si512(tmp_add3, tmp_shl3);
 }

 // 3. Store results back to memory
 _mm512_store_si512(&data[i + 0].a, zmm_a0);
 _mm512_store_si512(&data[i + 1].a, zmm_a1);
 _mm512_store_si512(&data[i + 2].a, zmm_a2);
 _mm512_store_si512(&data[i + 3].a, zmm_a3);
 }
}
```

## Risk Register & Failure Modes

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Register Spills** | High latency due to stack access | Limit interleave factor to 4; verify assembly for spills |
| **Gather Instructions** | Severe throughput penalty | Enforce AoSoA layout; forbid `vgather` instructions |
| **Shift UB** | Non-portable results | Mask shift counts `& 31` in scalar reference and SIMD |
| **AVX-512 Throttling** | Reduced clock speed on SKX | Use runtime dispatch to fallback to AVX2 on older Xeons |