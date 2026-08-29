#include "fattn.cuh"

#include "fattn-common.cuh"

#include <cfloat>

static constexpr int FATTN_INDEXED_D       = 256;
static constexpr int FATTN_INDEXED_GQA     = 12;
static constexpr int FATTN_INDEXED_ID_TILE = 16;
static constexpr int FATTN_INDEXED_Q8_SIZE = sizeof(block_q8_0)*(FATTN_INDEXED_D/QK8_0);

static_assert(FATTN_INDEXED_Q8_SIZE % sizeof(uint4) == 0, "Q8 row must be uint4 aligned");

template<int D, int GQA, int ID_TILE>
__launch_bounds__(WARP_SIZE*GQA, 1)
static __global__ void flash_attn_ext_indexed_q8_0(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const int  * __restrict__ ids,
        float      * __restrict__ dst,
        const float scale,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int32_t ne11, const int32_t ne12, const int32_t ne13,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int32_t ne22, const int32_t ne23,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int32_t ne30,
        const int64_t nb31, const int64_t nb33) {
    constexpr int q_i32_per_head = D/sizeof(int);
    constexpr int q_ds_per_head  = D/QK8_1;
    constexpr int q_iters        = q_i32_per_head/WARP_SIZE;
    constexpr int v_per_lane     = D/WARP_SIZE;
    constexpr int v_step         = 4;
    constexpr int row_u4         = sizeof(block_q8_0)*(D/QK8_0)/sizeof(uint4);

    const int lane        = threadIdx.x;
    const int head_in_gqa = threadIdx.y;
    const int tid         = head_in_gqa*WARP_SIZE + lane;
    const int query       = blockIdx.x;
    const int kv_head     = blockIdx.y;
    const int sequence    = blockIdx.z;
    const int head        = kv_head*GQA + head_in_gqa;

    const int k_sequence = sequence/(ne03/ne13);
    const int v_sequence = sequence/(ne03/ne23);

    __shared__ int    q_i32[GQA][q_i32_per_head];
    __shared__ float2 q_ds [GQA][q_ds_per_head];
    __shared__ int    ids_tile[ID_TILE];
    __shared__ uint4  k_tile[ID_TILE][row_u4];
    __shared__ uint4  v_tile[ID_TILE][row_u4];

    const float * q = (const float *) (Q + sequence*nb03 + head*nb02 + query*nb01);
    for (int i = 0; i < q_i32_per_head; i += WARP_SIZE) {
        quantize_q8_1_to_shared<float2, WARP_SIZE>(
            q + 4*i, scale, q_i32[head_in_gqa] + i, q_ds[head_in_gqa] + i/QI8_1);
    }
    __syncthreads();

    int q_i32_reg[q_iters];
    float2 q_ds_reg[q_iters];
#pragma unroll
    for (int i = 0; i < q_iters; ++i) {
        const int iq = i*WARP_SIZE + lane;
        q_i32_reg[i] = q_i32[head_in_gqa][iq];
        q_ds_reg[i]  = q_ds[head_in_gqa][iq/QI8_1];
    }

    float maximum = -FLT_MAX;
    float sum      = 0.0f;
    float acc[v_per_lane] = {0.0f};

    const char * ids_row = (const char *) ids + sequence*nb33 + query*nb31;
    constexpr int nthreads = WARP_SIZE*GQA;

    for (int i0 = 0; i0 < ne30; i0 += ID_TILE) {
        const int tile_size = min(ID_TILE, ne30 - i0);
        if (tid < ID_TILE) {
            ids_tile[tid] = tid < tile_size ? *(const int *) (ids_row + (i0 + tid)*sizeof(int)) : -1;
        }
        __syncthreads();

        constexpr int u4_per_id = 2*row_u4;
        for (int i = tid; i < ID_TILE*u4_per_id; i += nthreads) {
            const int iid   = i/u4_per_id;
            const int rem   = i - iid*u4_per_id;
            const int is_v  = rem >= row_u4;
            const int iu4   = rem - is_v*row_u4;
            const int token = ids_tile[iid];

            uint4 value = {};
            if (token >= 0 && token < ne11) {
                const char * row = is_v
                    ? V + v_sequence*nb23 + kv_head*nb22 + token*nb21
                    : K + k_sequence*nb13 + kv_head*nb12 + token*nb11;
                value = ((const uint4 *) row)[iu4];
            }
            if (is_v) {
                v_tile[iid][iu4] = value;
            } else {
                k_tile[iid][iu4] = value;
            }
        }
        __syncthreads();

#pragma unroll
        for (int iid = 0; iid < ID_TILE; ++iid) {
            const int token = ids_tile[iid];
            if (iid >= tile_size || token < 0 || token >= ne11) {
                continue;
            }

            float score = vec_dot_fattn_vec_KQ_q8_0<D, WARP_SIZE>(
                (const char *) k_tile[iid], nullptr, q_i32_reg, q_ds_reg);
            score = warp_reduce_sum(score);

            float weight = 1.0f;
            if (score > maximum) {
                const float rescale = maximum == -FLT_MAX ? 0.0f : expf(maximum - score);
#pragma unroll
                for (int i = 0; i < v_per_lane; ++i) {
                    acc[i] *= rescale;
                }
                sum = sum*rescale + 1.0f;
                maximum = score;
            } else {
                weight = expf(score - maximum);
                sum += weight;
            }

#pragma unroll
            for (int d0 = lane*v_step; d0 < D; d0 += WARP_SIZE*v_step) {
                float values[v_step];
                dequantize_V_q8_0<float, v_step>(v_tile[iid], values, d0);
                const int ia = (d0/(WARP_SIZE*v_step))*v_step;
#pragma unroll
                for (int d1 = 0; d1 < v_step; ++d1) {
                    acc[ia + d1] += weight*values[d1];
                }
            }
        }
        __syncthreads();
    }

    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f/sum;
    float * out = dst + ((sequence*ne01 + query)*ne02 + head)*D;
#pragma unroll
    for (int d0 = lane*v_step; d0 < D; d0 += WARP_SIZE*v_step) {
        const int ia = (d0/(WARP_SIZE*v_step))*v_step;
#pragma unroll
        for (int d1 = 0; d1 < v_step; ++d1) {
            out[d0 + d1] = acc[ia + d1]*inv_sum;
        }
    }

    GGML_UNUSED(ne12);
    GGML_UNUSED(ne22);
}

bool ggml_cuda_flash_attn_ext_indexed_supported(int device, const ggml_tensor * dst) {
    if (!GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc)) {
        return false;
    }

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * ids = dst->src[3];

    if (!q || !k || !v || !ids) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_0 || v->type != GGML_TYPE_Q8_0 ||
            ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->ne[0] != FATTN_INDEXED_D || k->ne[0] != FATTN_INDEXED_D || v->ne[0] != FATTN_INDEXED_D) {
        return false;
    }
    if (k->ne[2] <= 0 || q->ne[2] != FATTN_INDEXED_GQA*k->ne[2] || v->ne[2] != k->ne[2]) {
        return false;
    }
    if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (ids->ne[0] <= 0 || ids->ne[0] > 2051 || ids->ne[1] != q->ne[1] || ids->ne[2] != 1 || ids->ne[3] != q->ne[3]) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(block_q8_0) || v->nb[0] != sizeof(block_q8_0) ||
            ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (k->nb[1] % sizeof(uint4) != 0 || k->nb[2] % sizeof(uint4) != 0 || k->nb[3] % sizeof(uint4) != 0 ||
            v->nb[1] % sizeof(uint4) != 0 || v->nb[2] % sizeof(uint4) != 0 || v->nb[3] % sizeof(uint4) != 0) {
        return false;
    }
    if (k->view_offs % sizeof(uint4) != 0 || v->view_offs % sizeof(uint4) != 0) {
        return false;
    }
    if (k->ne[1] != v->ne[1] || k->ne[3] != v->ne[3]) {
        return false;
    }

    return true;
}

void ggml_cuda_flash_attn_ext_indexed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_indexed_supported(ctx.device, dst));
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * ids = dst->src[3];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    const dim3 blocks(q->ne[1], k->ne[2], q->ne[3]);
    const dim3 threads(WARP_SIZE, FATTN_INDEXED_GQA, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());

    ggml_cuda_kernel_launch(
        (flash_attn_ext_indexed_q8_0<FATTN_INDEXED_D, FATTN_INDEXED_GQA, FATTN_INDEXED_ID_TILE>),
        launch_params,
        (const char *) q->data,
        (const char *) k->data,
        (const char *) v->data,
        (const int  *) ids->data,
        (float      *) dst->data,
        scale,
        q->ne[1], q->ne[2], q->ne[3], q->nb[1], q->nb[2], q->nb[3],
        k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3],
        v->ne[2], v->ne[3], v->nb[1], v->nb[2], v->nb[3],
        ids->ne[0], ids->nb[1], ids->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
