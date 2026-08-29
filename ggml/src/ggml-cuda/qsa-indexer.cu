#include "qsa-indexer.cuh"

#if defined(GGML_USE_HIP)
#include <hipcub/block/block_radix_sort.hpp>
#endif

static constexpr int QSA_INDEXER_D           = 128;
static constexpr int QSA_INDEXER_HEADS       = 4;
static constexpr int QSA_INDEXER_BLOCK_SIZE  = 4;
static constexpr int QSA_INDEXER_TOKEN_TOP_K = 2048;
static constexpr int QSA_INDEXER_BLOCK_TOP_K = QSA_INDEXER_TOKEN_TOP_K/QSA_INDEXER_BLOCK_SIZE;
static constexpr int QSA_INDEXER_N_IDS       = QSA_INDEXER_TOKEN_TOP_K + QSA_INDEXER_BLOCK_SIZE - 1;
static constexpr int QSA_INDEXER_THREADS     = 256;
static constexpr int QSA_INDEXER_ITEMS       = 2;
static constexpr int QSA_INDEXER_TILE        = QSA_INDEXER_THREADS*QSA_INDEXER_ITEMS;

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ uint64_t qsa_indexer_key(float score, int block) {
    return (uint64_t(__float_as_uint(score)) << 32) | (0xffffffffu - uint32_t(block));
}

__launch_bounds__(QSA_INDEXER_THREADS, 1)
static __global__ void qsa_indexer_f32(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ block_cells,
        const char * __restrict__ visible,
        const char * __restrict__ tail,
        char       * __restrict__ dst,
        int n_blocks,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk3,
        size_t nbc3,
        size_t nbv1, size_t nbv3,
        size_t nbt1, size_t nbt3,
        size_t nbd1, size_t nbd3) {
    using block_sort = hipcub::BlockRadixSort<uint64_t, QSA_INDEXER_THREADS, QSA_INDEXER_ITEMS, int>;

    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int warp  = tid / WARP_SIZE;
    const int query = blockIdx.x;
    const int layer = blockIdx.y;

    __shared__ typename block_sort::TempStorage sort_storage;
    __shared__ float    q_shared[QSA_INDEXER_HEADS][QSA_INDEXER_D];
    __shared__ uint64_t top_keys[QSA_INDEXER_BLOCK_TOP_K];
    __shared__ int      top_ids [QSA_INDEXER_BLOCK_TOP_K];
    __shared__ uint64_t tile_keys[QSA_INDEXER_TILE];
    __shared__ int      tile_ids [QSA_INDEXER_TILE];
    __shared__ uint64_t merge_keys[QSA_INDEXER_BLOCK_TOP_K];
    __shared__ int      merge_ids [QSA_INDEXER_BLOCK_TOP_K];
    __shared__ int      n_visible;

    const char * q_row = q + layer*nbq3 + query*nbq2;
    for (int i = tid; i < QSA_INDEXER_HEADS*QSA_INDEXER_D; i += QSA_INDEXER_THREADS) {
        const int head = i/QSA_INDEXER_D;
        const int d    = i - head*QSA_INDEXER_D;
        q_shared[head][d] = *(const float *) (q_row + head*nbq1 + d*sizeof(float));
    }
    for (int i = tid; i < QSA_INDEXER_BLOCK_TOP_K; i += QSA_INDEXER_THREADS) {
        top_keys[i] = 0;
        top_ids [i] = -1;
    }
    if (tid == 0) {
        const int value = *(const int *) (visible + layer*nbv3 + query*nbv1);
        n_visible = max(0, min(value, n_blocks));
    }
    __syncthreads();

    constexpr int warps = QSA_INDEXER_THREADS/WARP_SIZE;
    for (int tile0 = 0; tile0 < n_visible; tile0 += QSA_INDEXER_TILE) {
        for (int i = warp; i < QSA_INDEXER_TILE; i += warps) {
            const int block = tile0 + i;
            float score = 0.0f;
            if (block < n_visible) {
                const float * k_row = (const float *) (k + layer*nbk3 + block*nbk1);
#pragma unroll
                for (int head = 0; head < QSA_INDEXER_HEADS; ++head) {
                    float qk = 0.0f;
#pragma unroll
                    for (int d = lane; d < QSA_INDEXER_D; d += WARP_SIZE) {
                        qk += q_shared[head][d]*k_row[d];
                    }
                    qk = warp_reduce_sum(qk);
                    score += qk > 0.0f ? qk : 0.0f;
                }
            }
            if (lane == 0) {
                tile_keys[i] = block < n_visible ? qsa_indexer_key(score, block) : 0;
                tile_ids [i] = block < n_visible ? block : -1;
            }
        }
        __syncthreads();

        uint64_t keys[QSA_INDEXER_ITEMS];
        int      ids [QSA_INDEXER_ITEMS];
#pragma unroll
        for (int i = 0; i < QSA_INDEXER_ITEMS; ++i) {
            const int item = tid*QSA_INDEXER_ITEMS + i;
            keys[i] = tile_keys[item];
            ids [i] = tile_ids [item];
        }
        block_sort(sort_storage).SortDescending(keys, ids);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < QSA_INDEXER_ITEMS; ++i) {
            const int item = tid*QSA_INDEXER_ITEMS + i;
            tile_keys[item] = keys[i];
            tile_ids [item] = ids [i];
        }
        __syncthreads();

        if (tid == 0) {
            int itop  = 0;
            int itile = 0;
            for (int i = 0; i < QSA_INDEXER_BLOCK_TOP_K; ++i) {
                const bool take_tile = itop == QSA_INDEXER_BLOCK_TOP_K ||
                    (itile < QSA_INDEXER_TILE && tile_keys[itile] > top_keys[itop]);
                if (take_tile) {
                    merge_keys[i] = tile_keys[itile];
                    merge_ids [i] = tile_ids [itile];
                    ++itile;
                } else {
                    merge_keys[i] = top_keys[itop];
                    merge_ids [i] = top_ids [itop];
                    ++itop;
                }
            }
        }
        __syncthreads();
        for (int i = tid; i < QSA_INDEXER_BLOCK_TOP_K; i += QSA_INDEXER_THREADS) {
            top_keys[i] = merge_keys[i];
            top_ids [i] = merge_ids [i];
        }
        __syncthreads();
    }

    int * out = (int *) (dst + layer*nbd3 + query*nbd1);
    const int * cells = (const int *) (block_cells + layer*nbc3);
    for (int i = tid; i < QSA_INDEXER_TOKEN_TOP_K; i += QSA_INDEXER_THREADS) {
        const int block  = top_ids[i/QSA_INDEXER_BLOCK_SIZE];
        const int member = i % QSA_INDEXER_BLOCK_SIZE;
        out[i] = block >= 0 ? cells[QSA_INDEXER_BLOCK_SIZE*block + member] : -1;
    }
    const int * tail_row = (const int *) (tail + layer*nbt3 + query*nbt1);
    if (tid < QSA_INDEXER_BLOCK_SIZE - 1) {
        out[QSA_INDEXER_TOKEN_TOP_K + tid] = tail_row[tid];
    }
}

#endif

bool ggml_cuda_qsa_indexer_supported(int device, const ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc)) {
        return false;
    }

    const ggml_tensor * q           = dst->src[0];
    const ggml_tensor * k           = dst->src[1];
    const ggml_tensor * block_cells = dst->src[2];
    const ggml_tensor * visible     = dst->src[3];
    const ggml_tensor * tail        = dst->src[4];
    if (!q || !k || !block_cells || !visible || !tail) {
        return false;
    }

    const int block_size = ggml_get_op_params_i32(dst, 0);
    const int token_top_k = ggml_get_op_params_i32(dst, 1);
    if (block_size != QSA_INDEXER_BLOCK_SIZE || token_top_k != QSA_INDEXER_TOKEN_TOP_K) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || block_cells->type != GGML_TYPE_I32 ||
            visible->type != GGML_TYPE_I32 || tail->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_I32) {
        return false;
    }
    if (q->ne[0] != QSA_INDEXER_D || q->ne[1] != QSA_INDEXER_HEADS || q->ne[2] <= 0 || q->ne[3] <= 0 ||
            k->ne[0] != QSA_INDEXER_D || k->ne[1] <= 0 || k->ne[1] > INT32_MAX/QSA_INDEXER_BLOCK_SIZE ||
            k->ne[2] != 1 || k->ne[3] != q->ne[3]) {
        return false;
    }
    if (block_cells->ne[0] != QSA_INDEXER_BLOCK_SIZE*k->ne[1] || block_cells->ne[1] != 1 ||
            block_cells->ne[2] != 1 || block_cells->ne[3] != q->ne[3]) {
        return false;
    }
    if (visible->ne[0] != 1 || visible->ne[1] != q->ne[2] || visible->ne[2] != 1 || visible->ne[3] != q->ne[3] ||
            tail->ne[0] != QSA_INDEXER_BLOCK_SIZE - 1 || tail->ne[1] != q->ne[2] ||
            tail->ne[2] != 1 || tail->ne[3] != q->ne[3]) {
        return false;
    }
    if (dst->ne[0] != QSA_INDEXER_N_IDS || dst->ne[1] != q->ne[2] || dst->ne[2] != 1 || dst->ne[3] != q->ne[3]) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(float) || block_cells->nb[0] != sizeof(int32_t) ||
            visible->nb[0] != sizeof(int32_t) || tail->nb[0] != sizeof(int32_t) || dst->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (q->ne[2] > INT32_MAX || q->ne[3] > INT32_MAX) {
        return false;
    }

    return true;
#endif
}

void ggml_cuda_qsa_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_qsa_indexer_supported(ctx.device, dst));
#if !defined(GGML_USE_HIP)
    GGML_ABORT("QSA indexer is not supported");
#else
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * q           = dst->src[0];
    const ggml_tensor * k           = dst->src[1];
    const ggml_tensor * block_cells = dst->src[2];
    const ggml_tensor * visible     = dst->src[3];
    const ggml_tensor * tail        = dst->src[4];

    const dim3 blocks(q->ne[2], q->ne[3], 1);
    qsa_indexer_f32<<<blocks, QSA_INDEXER_THREADS, 0, ctx.stream()>>>(
        (const char *) q->data,
        (const char *) k->data,
        (const char *) block_cells->data,
        (const char *) visible->data,
        (const char *) tail->data,
        (char *) dst->data,
        k->ne[1],
        q->nb[1], q->nb[2], q->nb[3],
        k->nb[1], k->nb[3],
        block_cells->nb[3],
        visible->nb[1], visible->nb[3],
        tail->nb[1], tail->nb[3],
        dst->nb[1], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
#endif
}
