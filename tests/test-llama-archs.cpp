#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-moe-stream.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v/--verbose] [-h/--help]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static constexpr uint32_t qwen4exp_ple_image_token_id = 126;

static void test_qwen4exp_ple_metadata_save() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    GGML_ASSERT(model);

    const std::string chat_template_key = LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const std::string chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}";
    model->gguf_kv.emplace(chat_template_key, chat_template);

    llama_hparams & hparams = model->hparams;
    hparams.n_layer_all = 1;
    hparams.ple_n_heads = 1;
    hparams.ple_ngram_size = 2;
    hparams.ple_heads_per_ngram = 1;
    hparams.ple_conv_kernel = 2;
    hparams.ple_eos_token_id = 127;
    hparams.ple_image_token_id = qwen4exp_ple_image_token_id;
    hparams.ple_head_dim = 1;
    hparams.is_ple_impl.set(0);
    hparams.ple_layer_multipliers[0] = 1;
    hparams.ple_layer_multipliers[1] = 2;
    hparams.ple_head_vocab_sizes[0] = 1;

    llama_model_saver saver(model.get());
    saver.add_kv_from_model();

    const std::string key = LLM_KV(model->arch)(LLM_KV_PLE_IMAGE_TOKEN_ID);
    const int64_t key_id = gguf_find_key(saver.gguf_ctx, key.c_str());
    GGML_ASSERT(key_id >= 0);
    GGML_ASSERT(gguf_get_val_u32(saver.gguf_ctx, key_id) == qwen4exp_ple_image_token_id);

    const int64_t chat_template_key_id = gguf_find_key(saver.gguf_ctx, chat_template_key.c_str());
    GGML_ASSERT(chat_template_key_id >= 0);
    GGML_ASSERT(gguf_get_kv_type(saver.gguf_ctx, chat_template_key_id) == GGUF_TYPE_STRING);
    GGML_ASSERT(chat_template == gguf_get_val_str(saver.gguf_ctx, chat_template_key_id));
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool ple = false) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    GGML_ASSERT(!ple || arch == LLM_ARCH_QWEN4EXP);

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        if (ple) {
            ms.add_kv(LLM_KV_PLE_LAYERS,            std::vector<uint32_t>{0});
            ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,        uint32_t(2));
            ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,   uint32_t(4));
            ms.add_kv(LLM_KV_PLE_CONV_KERNEL,       uint32_t(2));
            ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS, std::vector<uint64_t>{1, 2});
            ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,      std::vector<uint64_t>{0, 8, 16, 24});
            ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,  std::vector<uint64_t>{8, 8, 8, 8});
            ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,      uint32_t(127));
            ms.add_kv(LLM_KV_PLE_IMAGE_TOKEN_ID,    qwen4exp_ple_image_token_id);
            ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, uint32_t(64));

            ggml_tensor t = {};
            t.type  = GGML_TYPE_F16;
            t.ne[0] = 64;
            t.ne[1] = 32;
            t.ne[2] = 1;
            t.ne[3] = 1;
            t.nb[0] = ggml_type_size(t.type);
            t.nb[1] = ggml_row_size(t.type, t.ne[0]);
            t.nb[2] = t.nb[1] * t.ne[1];
            t.nb[3] = t.nb[2] * t.ne[2];
            const std::string name = LLM_TN(arch)(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
            ggml_set_name(&t, name.c_str());
            gguf_add_tensor(ms.gguf_ctx, &t);
        }
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false, uint32_t n_seq_max = 1,
        bool kv_unified = false, ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.kv_unified = kv_unified;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = cb_eval_user_data;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false,
        llama_seq_id seq_id = 0, llama_pos pos_offset = 0) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos_offset + pos, {seq_id}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static std::vector<float> get_qwen4exp_unified_logits(
        llama_model * model, llama_context * lctx,
        const std::vector<llama_token> & target,
        const std::vector<llama_token> & companion,
        llama_seq_id target_seq) {
    GGML_ASSERT(target.size() == companion.size());
    GGML_ASSERT(target_seq == 0 || target_seq == 1);

    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens = target.size();
    llama_batch batch = llama_batch_init(2*n_tokens, 0, 1);
    batch.n_tokens = 0;

    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        const auto & tokens = seq_id == target_seq ? target : companion;
        for (uint32_t pos = 0; pos < n_tokens; ++pos) {
            const int32_t i = batch.n_tokens++;
            batch.token[i] = tokens[pos];
            batch.pos[i] = pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = seq_id == target_seq && pos + 1 == n_tokens;
        }
    }

    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode unified QSA batch");
    }

    const int32_t output_idx = target_seq*n_tokens + n_tokens - 1;
    const float * logits = llama_get_logits_ith(lctx, output_idx);
    GGML_ASSERT(logits != nullptr);
    std::vector<float> result(logits, logits + n_vocab);

    llama_batch_free(batch);
    llama_memory_clear(llama_get_memory(lctx), true);
    return result;
}

static void test_qwen4exp_qsa_unified_sequences() {
    constexpr size_t   seed     = 3321213324;
    constexpr uint32_t n_prompt = 32;
    constexpr uint32_t n_vocab  = 128;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 2, true);

    const std::vector<llama_token> target      = get_tokens(n_prompt, n_vocab, seed);
    const std::vector<llama_token> companion_a = get_tokens(n_prompt, n_vocab, seed + 1);
    const std::vector<llama_token> companion_b = get_tokens(n_prompt, n_vocab, seed + 2);

    const auto seq0_a = get_qwen4exp_unified_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), target, companion_a, 0);
    const auto seq0_b = get_qwen4exp_unified_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), target, companion_b, 0);
    GGML_ASSERT(seq0_a == seq0_b);

    const auto seq1_a = get_qwen4exp_unified_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), target, companion_a, 1);
    const auto seq1_b = get_qwen4exp_unified_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), target, companion_b, 1);
    GGML_ASSERT(seq1_a == seq1_b);
}

static void test_qwen4exp_qsa_non_causal() {
    constexpr size_t   seed     = 3321213324;
    constexpr uint32_t n_prompt = 17;
    constexpr uint32_t n_vocab  = 128;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
    llama_set_causal_attn(model_and_ctx.second.get(), false);

    std::vector<llama_token> prompt_a = get_tokens(n_prompt, n_vocab, seed);
    std::vector<llama_token> prompt_b = prompt_a;
    prompt_b.back() = (prompt_b.back() + 1) % n_vocab;

    const std::vector<float> logits_a = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), prompt_a);
    llama_memory_clear(llama_get_memory(model_and_ctx.second.get()), true);
    const std::vector<float> logits_b = get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), prompt_b);

    bool changed = false;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        changed |= logits_a[i] != logits_b[i];
    }
    GGML_ASSERT(changed);
}

struct qwen4exp_qsa_capture {
    int64_t n_blocks = 0;
    int64_t query = -1;
    int64_t selected = -1;
    int64_t pooled_norm_shape[4] = {};
    std::vector<int32_t> block_pos;
    ggml_type pooled_type = GGML_TYPE_COUNT;
    std::vector<float> pooled_keys;
    std::vector<float> pooled_keys_f32;
};

static float qwen4exp_qsa_get_f32(const ggml_tensor * tensor, const std::vector<uint8_t> & data, size_t offset) {
    if (tensor->type == GGML_TYPE_F32) {
        float value;
        memcpy(&value, data.data() + offset, sizeof(value));
        return value;
    }

    GGML_ASSERT(tensor->type == GGML_TYPE_F16);
    ggml_fp16_t value;
    memcpy(&value, data.data() + offset, sizeof(value));
    return ggml_fp16_to_fp32(value);
}

static bool qwen4exp_qsa_capture_cb(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<qwen4exp_qsa_capture *>(user_data);

    const ggml_tensor * norm_src = tensor->src[0];
    bool pooled_norm = capture->pooled_norm_shape[0] == 0 && tensor->op == GGML_OP_RMS_NORM;
    for (int depth = 0; pooled_norm && norm_src != nullptr && depth < 4; ++depth, norm_src = norm_src->src[0]) {
        if (strncmp(norm_src->name, "indexer_k_pooled-", strlen("indexer_k_pooled-")) == 0) {
            break;
        }
    }
    pooled_norm &= norm_src != nullptr &&
        strncmp(norm_src->name, "indexer_k_pooled-", strlen("indexer_k_pooled-")) == 0;

    const bool block_pos = capture->block_pos.empty() && capture->n_blocks > 0 &&
        tensor->op == GGML_OP_ROPE && tensor->src[1] != nullptr &&
        ggml_nelements(tensor->src[1]) == 4*capture->n_blocks && tensor->ne[2] == capture->n_blocks;
    const bool old_selection = capture->selected < 0 && capture->query >= 0 &&
        tensor->op == GGML_OP_SET_ROWS && tensor->src[1] != nullptr &&
        strncmp(tensor->src[1]->name, "indexer_top_k-", strlen("indexer_top_k-")) == 0;
    const bool qsa_mask = capture->selected < 0 && capture->query >= 0 &&
        strncmp(tensor->name, "qsa_mask-", strlen("qsa_mask-")) == 0;
    const bool pooled_keys = capture->pooled_keys.empty() &&
        strncmp(tensor->name, "indexer_k_pooled-", strlen("indexer_k_pooled-")) == 0;

    if (ask) {
        return pooled_norm || block_pos || old_selection || qsa_mask || pooled_keys;
    }

    if (pooled_norm) {
        memcpy(capture->pooled_norm_shape, tensor->ne, sizeof(capture->pooled_norm_shape));
    }

    if (block_pos) {
        capture->block_pos.resize(4*capture->n_blocks);
        ggml_backend_tensor_get(tensor->src[1], capture->block_pos.data(), 0, capture->block_pos.size()*sizeof(int32_t));
    }

    if (old_selection || qsa_mask) {
        GGML_ASSERT(capture->query < tensor->ne[qsa_mask ? 1 : 2]);
        std::vector<uint8_t> data(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size());

        capture->selected = 0;
        for (int64_t j = 0; j < tensor->ne[qsa_mask ? 0 : 1]; ++j) {
            const size_t offset = qsa_mask
                ? j*tensor->nb[0] + capture->query*tensor->nb[1]
                : j*tensor->nb[1] + capture->query*tensor->nb[2];
            capture->selected += std::isfinite(qwen4exp_qsa_get_f32(tensor, data, offset));
        }
    }

    if (pooled_keys) {
        std::vector<uint8_t> data(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size());

        capture->pooled_type = tensor->type;
        capture->pooled_keys.resize(ggml_nelements(tensor));
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            capture->pooled_keys[i] = qwen4exp_qsa_get_f32(tensor, data, i*tensor->nb[0]);
        }

        const ggml_tensor * src = tensor->src[0];
        if (src != nullptr && src->type == GGML_TYPE_F32 && ggml_nelements(src) == ggml_nelements(tensor)) {
            capture->pooled_keys_f32.resize(ggml_nelements(src));
            ggml_backend_tensor_get(src, capture->pooled_keys_f32.data(), 0, ggml_nbytes(src));
        }
    }

    return true;
}

static void test_qwen4exp_qsa_norm_layout() {
    constexpr size_t seed = 3321213324;
    constexpr uint32_t n_tokens = 4;

    qwen4exp_qsa_capture capture;
    capture.n_blocks = 64;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 2, true,
            qwen4exp_qsa_capture_cb, &capture);

    const std::vector<llama_token> target = get_tokens(n_tokens, 128, seed);
    const std::vector<llama_token> companion = get_tokens(n_tokens, 128, seed + 1);
    get_qwen4exp_unified_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), target, companion, 0);

    if (capture.pooled_norm_shape[1] != capture.n_blocks || capture.pooled_norm_shape[2] != 2) {
        fprintf(stderr, "Qwen4Exp QSA regression: pooled key norm shape is [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                capture.pooled_norm_shape[0], capture.pooled_norm_shape[1],
                capture.pooled_norm_shape[2], capture.pooled_norm_shape[3]);
    }
    GGML_ASSERT(capture.pooled_norm_shape[1] == capture.n_blocks);
    GGML_ASSERT(capture.pooled_norm_shape[2] == 2);
}

static std::vector<int32_t> get_qwen4exp_qsa_block_pos() {
    constexpr size_t seed = 3321213324;
    constexpr int32_t n_tokens = 8;
    constexpr int32_t n_pos = 4;

    qwen4exp_qsa_capture capture;
    capture.n_blocks = 64;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 1, false,
            qwen4exp_qsa_capture_cb, &capture);

    const int32_t n_embd = llama_model_n_embd(model_and_ctx.first.get());
    std::vector<float> embd(n_tokens*n_embd, 0.0f);
    std::vector<llama_pos> pos(n_tokens*n_pos);
    std::vector<int32_t> n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id> seq_data(n_tokens, 0);
    std::vector<llama_seq_id *> seq_id(n_tokens);
    std::vector<int8_t> logits(n_tokens, false);

    for (int32_t i = 0; i < n_tokens; ++i) {
        pos[              i] = i < 4 ? 0 : 5;
        pos[  n_tokens + i] = i < 4 ? 0 : 1;
        pos[2*n_tokens + i] = i % 4;
        pos[3*n_tokens + i] = 0;
        seq_id[i] = &seq_data[i];
    }
    logits.back() = true;

    llama_batch batch = {
        n_tokens,
        nullptr,
        embd.data(),
        pos.data(),
        n_seq_id.data(),
        seq_id.data(),
        logits.data(),
    };
    GGML_ASSERT(llama_decode(model_and_ctx.second.get(), batch) == 0);
    return capture.block_pos;
}

static int64_t get_qwen4exp_qsa_selected_count() {
    constexpr size_t seed = 3321213324;
    constexpr uint32_t n_tokens = 14;

    qwen4exp_qsa_capture capture;
    capture.query = n_tokens - 1;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 1, false,
            qwen4exp_qsa_capture_cb, &capture);
    get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), get_tokens(n_tokens, 128, seed));
    return capture.selected;
}

static void test_qwen4exp_qsa_block_semantics() {
    const std::vector<int32_t> block_pos = get_qwen4exp_qsa_block_pos();
    const int64_t selected = get_qwen4exp_qsa_selected_count();
    const int64_t n_blocks = 64;

    const bool positions_ok = block_pos.size() == 4*n_blocks &&
        block_pos[0] == 0 && block_pos[1] == 5 &&
        block_pos[n_blocks] == 0 && block_pos[n_blocks + 1] == 1 &&
        block_pos[2*n_blocks] == 0 && block_pos[2*n_blocks + 1] == 0 &&
        block_pos[3*n_blocks] == 0 && block_pos[3*n_blocks + 1] == 0;

    if (!positions_ok || selected != 10) {
        fprintf(stderr, "Qwen4Exp QSA regression: block positions %s, selected=%" PRId64 "\n",
                positions_ok ? "OK" : "incorrect", selected);
    }
    GGML_ASSERT(positions_ok);
    GGML_ASSERT(selected == 10);
}

static void test_qwen4exp_qsa_pool_precision() {
    constexpr size_t seed = 3321213324;
    constexpr uint32_t n_tokens = 14;

    qwen4exp_qsa_capture capture;
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 1, false,
            qwen4exp_qsa_capture_cb, &capture);
    get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), get_tokens(n_tokens, 128, seed));

    GGML_ASSERT(!capture.pooled_keys.empty());
    int64_t unrounded = 0;
    for (const float value : capture.pooled_keys) {
        GGML_ASSERT(std::isfinite(value));
        unrounded += value != ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
    }
    if (capture.pooled_type != GGML_TYPE_F16 || unrounded != 0) {
        fprintf(stderr, "Qwen4Exp QSA regression: pooled keys are %s with %" PRId64 " values outside FP16 precision\n",
                ggml_type_name(capture.pooled_type), unrounded);
    }
    GGML_ASSERT(capture.pooled_type == GGML_TYPE_F16);
    GGML_ASSERT(unrounded == 0);

    GGML_ASSERT(capture.pooled_keys_f32.size() == capture.pooled_keys.size());
    int64_t changed = 0;
    for (size_t i = 0; i < capture.pooled_keys.size(); ++i) {
        const float rounded = ggml_fp16_to_fp32(ggml_fp32_to_fp16(capture.pooled_keys_f32[i]));
        GGML_ASSERT(capture.pooled_keys[i] == rounded);
        changed += capture.pooled_keys_f32[i] != rounded;
    }
    GGML_ASSERT(changed > 0);
}

struct qwen4exp_ple_capture {
    std::vector<int32_t> rows;
};

static bool qwen4exp_ple_capture_cb(ggml_tensor * tensor, bool ask, void * user_data) {
    const bool ple_embd = strncmp(tensor->name, "ple_embd-", strlen("ple_embd-")) == 0;
    if (ask) {
        return ple_embd;
    }

    if (ple_embd) {
        auto * capture = static_cast<qwen4exp_ple_capture *>(user_data);
        const ggml_tensor * gather = tensor;
        while (gather != nullptr && gather->op != GGML_OP_GET_ROWS) {
            gather = gather->src[0];
        }
        GGML_ASSERT(gather != nullptr);
        const ggml_tensor * rows = gather->src[1];
        GGML_ASSERT(rows != nullptr && rows->type == GGML_TYPE_I32);
        capture->rows.resize(ggml_nelements(rows));
        ggml_backend_tensor_get(rows, capture->rows.data(), 0, ggml_nbytes(rows));
    }

    return true;
}

static void decode_qwen4exp_image_chunk(llama_context * lctx, int32_t x0, int32_t n_tokens) {
    const int32_t n_embd = llama_model_n_embd(llama_get_model(lctx));
    std::vector<float> embd(n_tokens*n_embd, 0.0f);
    std::vector<llama_pos> pos(4*n_tokens, 0);
    std::vector<int32_t> n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id> seq_data(n_tokens, 0);
    std::vector<llama_seq_id *> seq_id(n_tokens);
    std::vector<int8_t> logits(n_tokens, false);

    for (int32_t i = 0; i < n_tokens; ++i) {
        pos[2*n_tokens + i] = x0 + i;
        seq_id[i] = &seq_data[i];
    }
    logits.back() = true;

    llama_batch batch = {
        n_tokens,
        nullptr,
        embd.data(),
        pos.data(),
        n_seq_id.data(),
        seq_id.data(),
        logits.data(),
    };
    GGML_ASSERT(llama_decode(lctx, batch) == 0);
}

static void test_qwen4exp_ple_split_mrope_history() {
    constexpr size_t seed = 3321213324;

    qwen4exp_ple_capture capture;
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 1, false,
            qwen4exp_ple_capture_cb, &capture);

    decode_qwen4exp_image_chunk(model_and_ctx.second.get(), 0, 65);

    const int32_t hash = (qwen4exp_ple_image_token_id ^ 2*qwen4exp_ple_image_token_id) % 8;
    const std::vector<int32_t> expected = { hash, hash + 8, hash + 16, hash + 24 };
    if (capture.rows != expected) {
        fprintf(stderr, "Qwen4Exp PLE regression: split image rows");
        for (const int32_t row : capture.rows) {
            fprintf(stderr, " %d", row);
        }
        fprintf(stderr, "\n");
    }
    GGML_ASSERT(capture.rows == expected);
}

static void test_qwen4exp_ple_shared_prefix() {
    constexpr size_t seed = 3321213324;

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 2, true);
    const std::vector<llama_token> tokens = get_tokens(4, 128, seed);

    llama_batch batch = llama_batch_init(tokens.size(), 0, 2);
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        common_batch_add(batch, tokens[pos], pos, {0, 1}, pos + 1 == tokens.size());
    }
    GGML_ASSERT(llama_decode(model_and_ctx.second.get(), batch) == 0);
    llama_batch_free(batch);
}

static void test_qwen4exp_indexer_seq_cp() {
    constexpr size_t   seed     = 3321213324;
    constexpr uint32_t n_prompt = 32;
    constexpr uint32_t n_vocab  = 128;

    gguf_context_ptr gguf_ctx_ref = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto ref = get_model_and_ctx(gguf_ctx_ref.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 2);

    gguf_context_ptr gguf_ctx_copy = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto copy = get_model_and_ctx(gguf_ctx_copy.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 2);

    const std::vector<llama_token> source = get_tokens(n_prompt, n_vocab, seed);
    const std::vector<llama_token> poison = get_tokens(n_prompt, n_vocab, seed + 1);
    const llama_token next = get_tokens(1, n_vocab, seed + 2)[0];

    get_logits(ref.first.get(), ref.second.get(), source, false, 0);
    get_logits(copy.first.get(), copy.second.get(), source, false, 0);
    get_logits(copy.first.get(), copy.second.get(), poison, false, 1);

    const std::vector<float> expected = get_logits(ref.first.get(), ref.second.get(), { next }, false, 0, n_prompt);

    llama_memory_seq_cp(llama_get_memory(copy.second.get()), 0, 1, -1, -1);
    const std::vector<float> actual = get_logits(copy.first.get(), copy.second.get(), { next }, false, 1, n_prompt);

    GGML_ASSERT(actual == expected);
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static void test_qwen4exp_moe_stream_exact() {
    constexpr size_t seed = 3321213324;

    for (const size_t alignment : { 1, 2, 512, 8192 }) {
        GGML_ASSERT(llama_moe_stream_alignment_supported(alignment));
    }
    for (const size_t alignment : { 0, 3, 6 }) {
        GGML_ASSERT(!llama_moe_stream_alignment_supported(alignment));
    }

    const auto range_512 = llama_moe_stream_align_range(700, 400, 512);
    GGML_ASSERT(range_512.offs == 512 && range_512.head == 188 && range_512.size == 1024);
    const auto range_8192 = llama_moe_stream_align_range(8190, 4, 8192);
    GGML_ASSERT(range_8192.offs == 0 && range_8192.head == 8190 && range_8192.size == 16384);
    GGML_ASSERT(llama_moe_stream_staging_size(400, 512) == 1424);
    GGML_ASSERT(llama_moe_stream_staging_size(400, 8192) == 16784);

    for (const char * name : { "CPU", "CUDA", "ROCm", "MTL", "Vulkan" }) {
        GGML_ASSERT(llama_moe_stream_backend_supported(name));
    }
    for (const char * name : { "OPENVINO", "MUSA", "RPC", "WebGPU" }) {
        GGML_ASSERT(!llama_moe_stream_backend_supported(name));
    }
    GGML_ASSERT(!llama_moe_stream_backend_supported(nullptr));

    const std::string path = (std::filesystem::temp_directory_path()/
            ("llama-qwen4exp-moe-stream-" + std::to_string(ggml_time_us()) + ".gguf")).string();
    struct cleanup_file {
        const std::string & path;
        ~cleanup_file() { std::remove(path.c_str()); }
    } cleanup { path };
    const std::string split_path_0 = path + ".split-0";
    const std::string split_path_1 = path + ".split-1";
    const std::string parallel_path = path + ".parallel";
    cleanup_file cleanup_split_0 { split_path_0 };
    cleanup_file cleanup_split_1 { split_path_1 };
    cleanup_file cleanup_parallel { parallel_path };

    {
        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        auto source = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
        llama_model_save_to_file(source.first.get(), path.c_str());

        llama_model_saver full(source.first.get());
        full.add_kv_from_model();
        full.add_tensors_from_model();
        gguf_context_ptr shard_0(gguf_init_empty());
        gguf_context_ptr shard_1(gguf_init_empty());
        gguf_set_kv(shard_0.get(), full.gguf_ctx);
        gguf_set_kv(shard_1.get(), full.gguf_ctx);

        const std::string split_no = "split.no";
        const std::string split_count = "split.count";
        const std::string split_tensors = "split.tensors.count";
        const int64_t n_tensors = gguf_get_n_tensors(full.gguf_ctx);
        GGML_ASSERT(n_tensors > 0 && n_tensors <= INT32_MAX);
        for (uint16_t i = 0; i < 2; ++i) {
            gguf_context * shard = i == 0 ? shard_0.get() : shard_1.get();
            gguf_set_val_u16(shard, split_no.c_str(), i);
            gguf_set_val_u16(shard, split_count.c_str(), 2);
            gguf_set_val_i32(shard, split_tensors.c_str(), (int32_t) n_tensors);
        }

        bool placed_expert_in_second_shard = false;
        for (const auto & [name, tensor] : llama_internal_get_tensor_map(source.first.get())) {
            const bool use_second_shard = !placed_expert_in_second_shard && name.find(".ffn_gate_up_exps.weight") != std::string::npos;
            gguf_add_tensor(use_second_shard ? shard_1.get() : shard_0.get(), tensor);
            placed_expert_in_second_shard = placed_expert_in_second_shard || use_second_shard;
        }
        GGML_ASSERT(placed_expert_in_second_shard);
        GGML_ASSERT(gguf_get_n_tensors(shard_0.get()) + gguf_get_n_tensors(shard_1.get()) == n_tensors);
        gguf_write_to_file(shard_0.get(), split_path_0.c_str(), false);
        gguf_write_to_file(shard_1.get(), split_path_1.c_str(), false);
    }

    {
        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        const std::string expert_count = LLM_KV(LLM_ARCH_QWEN4EXP)(LLM_KV_EXPERT_COUNT);
        const std::string expert_used_count = LLM_KV(LLM_ARCH_QWEN4EXP)(LLM_KV_EXPERT_USED_COUNT);
        GGML_ASSERT(gguf_find_key(gguf_ctx.get(), expert_count.c_str()) >= 0);
        GGML_ASSERT(gguf_find_key(gguf_ctx.get(), expert_used_count.c_str()) >= 0);
        gguf_set_val_u32(gguf_ctx.get(), expert_count.c_str(), 4);
        gguf_set_val_u32(gguf_ctx.get(), expert_used_count.c_str(), 2);
        auto source = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
        llama_model_save_to_file(source.first.get(), parallel_path.c_str());
    }

    const std::string empty_path = path + ".empty";
    cleanup_file cleanup_empty { empty_path };
    {
        FILE * file = fopen(empty_path.c_str(), "wb");
        GGML_ASSERT(file != nullptr);
        fclose(file);
    }
    {
        llama_file failed_file(empty_path.c_str(), "rb", false);
        llama_file fallback_file(path.c_str(), "rb", false);
        uint8_t read_buffer[4] = {};
        bool fallback_used = false;
        const uint8_t * data = llama_moe_stream_pread(failed_file, &fallback_file, read_buffer, sizeof(read_buffer), 0, true, 1, &fallback_used);
        GGML_ASSERT(fallback_used);
        GGML_ASSERT(data == read_buffer);
        GGML_ASSERT(memcmp(data, "GGUF", sizeof(read_buffer)) == 0);
    }

#ifdef __linux__
    llama_file direct_file(path.c_str(), "rb", true);
    if (direct_file.has_direct_io()) {
        llama_moe_stream direct_files(0, 1, 1, true);
        direct_files.open_files({ path });
        GGML_ASSERT(direct_files.use_direct_io);
        GGML_ASSERT(direct_files.io_alignment == direct_file.read_alignment());
        GGML_ASSERT(direct_files.buffered_files.size() == direct_files.files.size());
    }

    std::error_code proc_error;
    if (std::filesystem::exists("/proc/version", proc_error)) {
        llama_file buffered_file("/proc/version", "rb", true);
        if (direct_file.has_direct_io() && !buffered_file.has_direct_io()) {
            llama_moe_stream mixed_files(0, 1, 1, true);
            mixed_files.open_files({ path, "/proc/version" });
            GGML_ASSERT(!mixed_files.use_direct_io);
            for (const auto & file : mixed_files.files) {
                GGML_ASSERT(!file->has_direct_io());
            }
        }
    }
#endif

    auto make_context_params = []() {
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 64;
        context_params.n_batch = 8;
        context_params.n_ubatch = 1;
        context_params.n_threads = 4;
        context_params.n_threads_batch = 4;
        context_params.op_offload = false;
        return context_params;
    };

    auto make_model_params = [](bool stream) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.use_extra_bufts = false;
        model_params.moe_stream = stream;
        model_params.moe_stream_slots = stream ? 1 : 0;
        model_params.moe_stream_io_threads = stream ? 1 : 0;
        return model_params;
    };

    auto load = [&](bool stream) {
        llama_model_ptr model(llama_model_load_from_file(path.c_str(), make_model_params(stream)));
        GGML_ASSERT(model);

        llama_context_params context_params = make_context_params();
        if (stream) {
            context_params.n_ubatch = 8;
            context_params.op_offload = true;
        }
        llama_context_ptr context(llama_init_from_model(model.get(), context_params));
        GGML_ASSERT(context);
        return std::make_pair(std::move(model), std::move(context));
    };

    llama_model_params small_budget_params = make_model_params(true);
    small_budget_params.moe_stream_slots = 0;
    small_budget_params.moe_stream_budget = 1;
    llama_model_ptr small_budget(llama_model_load_from_file(path.c_str(), small_budget_params));
    GGML_ASSERT(small_budget == nullptr);

    llama_model_params mlock_params = make_model_params(true);
    mlock_params.load_mode = LLAMA_LOAD_MODE_MLOCK;
    llama_model_ptr mlock(llama_model_load_from_file(path.c_str(), mlock_params));
    GGML_ASSERT(mlock == nullptr);

    auto baseline = load(false);
    auto streamed = load(true);
    GGML_ASSERT(llama_n_ubatch(streamed.second.get()) == 8);
    GGML_ASSERT(!streamed.second->get_cparams().op_offload);

    llama_model_params retry_params = make_model_params(true);
    retry_params.moe_stream_direct = true;
    llama_model_ptr retry_model(llama_model_load_from_file(path.c_str(), retry_params));
    GGML_ASSERT(retry_model != nullptr);
    llama_context_ptr retry_context(llama_init_from_model(retry_model.get(), make_context_params()));
    GGML_ASSERT(retry_context != nullptr);
    llama_moe_stream * retry_stream = retry_model->moe_stream();
    GGML_ASSERT(retry_stream != nullptr);
    const bool retry_supported = retry_stream->use_direct_io;
    if (retry_supported) {
        GGML_ASSERT(retry_stream->files.size() == 1 && retry_stream->buffered_files.size() == 1);
        retry_stream->files[0].reset(new llama_file(empty_path.c_str(), "rb", false));
    }

    const char * split_paths[] = { split_path_0.c_str(), split_path_1.c_str() };
    llama_model_ptr split_model(llama_model_load_from_splits(split_paths, 2, make_model_params(true)));
    GGML_ASSERT(split_model != nullptr);
    llama_context_ptr split_context(llama_init_from_model(split_model.get(), make_context_params()));
    GGML_ASSERT(split_context != nullptr);
    llama_moe_stream * split_stream = split_model->moe_stream();
    GGML_ASSERT(split_stream != nullptr);

    std::vector<gguf_context_ptr> split_metadata;
    split_metadata.emplace_back(gguf_init_from_file(split_path_0.c_str(), { true, nullptr }));
    split_metadata.emplace_back(gguf_init_from_file(split_path_1.c_str(), { true, nullptr }));
    GGML_ASSERT(split_metadata[0] != nullptr && split_metadata[1] != nullptr);
    bool seen_shard[2] = {};
    const std::string cache_suffix = ".stream_cache";
    for (const auto & layer : split_stream->layers) {
        if (!layer) {
            continue;
        }
        for (const auto & weight : layer->weights) {
            GGML_ASSERT(weight.file_idx < 2);
            std::string name = weight.cache->name;
            GGML_ASSERT(name.size() > cache_suffix.size() && name.compare(name.size() - cache_suffix.size(), cache_suffix.size(), cache_suffix) == 0);
            name.resize(name.size() - cache_suffix.size());
            const int64_t tensor_id = gguf_find_tensor(split_metadata[weight.file_idx].get(), name.c_str());
            GGML_ASSERT(tensor_id >= 0);
            const size_t tensor_offs = gguf_get_tensor_offset(split_metadata[weight.file_idx].get(), tensor_id);
            GGML_ASSERT(weight.offs == gguf_get_data_offset(split_metadata[weight.file_idx].get()) + tensor_offs);
            if (weight.file_idx == 1) {
                GGML_ASSERT(tensor_offs == 0);
            }
            seen_shard[weight.file_idx] = true;
        }
    }
    GGML_ASSERT(seen_shard[0] && seen_shard[1]);

    for (const auto & layer : streamed.first->moe_stream()->layers) {
        if (!layer) {
            continue;
        }
        for (const auto & weight : layer->weights) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(weight.cache->buffer);
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (dev == nullptr) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            GGML_ASSERT(dev != nullptr && buft == ggml_backend_dev_buffer_type(dev));
            GGML_ASSERT(reg != nullptr && llama_moe_stream_backend_supported(ggml_backend_reg_name(reg)));
        }
    }

    auto assert_stream_memory = [](const llama_model * model, bool no_alloc) {
        llama_moe_stream * stream = model->moe_stream();
        GGML_ASSERT(stream != nullptr);

        std::map<ggml_backend_buffer_type_t, size_t> expected;
        if (no_alloc) {
            for (const auto & [buft, ctx] : stream->ctxs) {
                expected[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
            }
        } else {
            for (const auto & buffer : stream->bufs) {
                expected[ggml_backend_buffer_get_type(buffer.get())] += ggml_backend_buffer_get_size(buffer.get());
            }
        }

        const size_t staging_size = stream->max_nb_expert + 2*4096;
        expected[ggml_backend_cpu_buffer_type()] += (size_t) stream->n_io_threads*staging_size;
        GGML_ASSERT(stream->memory_breakdown(no_alloc) == expected);
    };
    auto assert_thread_delta = [](const llama_model * one, const llama_model * two) {
        llama_moe_stream * stream_one = one->moe_stream();
        llama_moe_stream * stream_two = two->moe_stream();
        GGML_ASSERT(stream_one != nullptr && stream_two != nullptr);
        GGML_ASSERT(stream_one->n_io_threads == 1 && stream_two->n_io_threads == 2);
        GGML_ASSERT(stream_one->max_nb_expert == stream_two->max_nb_expert);

        auto expected = one->memory_breakdown();
        expected[ggml_backend_cpu_buffer_type()] += stream_one->max_nb_expert + 2*4096;
        GGML_ASSERT(two->memory_breakdown() == expected);
    };
    assert_stream_memory(streamed.first.get(), false);

    llama_model_params two_thread_params = make_model_params(true);
    two_thread_params.moe_stream_io_threads = 2;
    llama_model_ptr two_thread(llama_model_load_from_file(path.c_str(), two_thread_params));
    GGML_ASSERT(two_thread != nullptr);
    assert_stream_memory(two_thread.get(), false);
    assert_thread_delta(streamed.first.get(), two_thread.get());

    llama_model_params no_alloc_params = make_model_params(true);
    no_alloc_params.load_mode = LLAMA_LOAD_MODE_NONE;
    no_alloc_params.no_alloc = true;
    llama_model_ptr no_alloc(llama_model_load_from_file(path.c_str(), no_alloc_params));
    GGML_ASSERT(no_alloc != nullptr);
    assert_stream_memory(no_alloc.get(), true);

    llama_model_params no_alloc_two_thread_params = no_alloc_params;
    no_alloc_two_thread_params.moe_stream_io_threads = 2;
    llama_model_ptr no_alloc_two_thread(llama_model_load_from_file(path.c_str(), no_alloc_two_thread_params));
    GGML_ASSERT(no_alloc_two_thread != nullptr);
    assert_stream_memory(no_alloc_two_thread.get(), true);
    assert_thread_delta(no_alloc.get(), no_alloc_two_thread.get());

    const std::string save_path = path + ".save";
    cleanup_file cleanup_save { save_path };
    constexpr char sentinel[] = "moe-stream-save-sentinel";
    {
        FILE * file = fopen(save_path.c_str(), "wb");
        GGML_ASSERT(file != nullptr);
        GGML_ASSERT(fwrite(sentinel, 1, sizeof(sentinel), file) == sizeof(sentinel));
        fclose(file);
    }
    llama_model_save_to_file(streamed.first.get(), save_path.c_str());
    {
        char contents[sizeof(sentinel)] = {};
        FILE * file = fopen(save_path.c_str(), "rb");
        GGML_ASSERT(file != nullptr);
        GGML_ASSERT(fread(contents, 1, sizeof(contents), file) == sizeof(contents));
        GGML_ASSERT(fgetc(file) == EOF);
        fclose(file);
        GGML_ASSERT(memcmp(contents, sentinel, sizeof(contents)) == 0);
    }

    llama_context_ptr concurrent(llama_init_from_model(streamed.first.get(), make_context_params()));
    GGML_ASSERT(concurrent == nullptr);
    streamed.second.reset();

    llama_context_params multi_seq_params = make_context_params();
    multi_seq_params.n_seq_max = 2;
    llama_context_ptr multi_seq(llama_init_from_model(streamed.first.get(), multi_seq_params));
    GGML_ASSERT(multi_seq != nullptr);
    multi_seq.reset();

    llama_context_ptr replacement(llama_init_from_model(streamed.first.get(), make_context_params()));
    GGML_ASSERT(replacement != nullptr);

    const std::vector<llama_token> tokens = get_tokens(8, 128, seed);
    const std::vector<float> expected = get_logits(baseline.first.get(), baseline.second.get(), tokens);
    replacement->set_warmup(true);
    const std::vector<float> actual = get_logits(streamed.first.get(), replacement.get(), tokens);
    replacement->set_warmup(false);
    GGML_ASSERT(expected == actual);
    if (retry_supported) {
        const std::vector<float> retry = get_logits(retry_model.get(), retry_context.get(), tokens);
        GGML_ASSERT(expected == retry);
        GGML_ASSERT(retry_stream->direct_io_failed.load(std::memory_order_relaxed));
    }
    const std::vector<float> split = get_logits(split_model.get(), split_context.get(), tokens);
    GGML_ASSERT(expected == split);

    llama_model_params parallel_baseline_params = make_model_params(false);
    parallel_baseline_params.n_gpu_layers = 99;
    llama_model_ptr parallel_baseline_model(llama_model_load_from_file(parallel_path.c_str(), parallel_baseline_params));
    GGML_ASSERT(parallel_baseline_model != nullptr);
    llama_context_params parallel_context_params = make_context_params();
    parallel_context_params.n_ctx = 128;
    parallel_context_params.n_batch = 64;
    parallel_context_params.n_ubatch = 64;
    llama_context_ptr parallel_baseline_context;

    llama_model_params parallel_params = make_model_params(true);
    parallel_params.n_gpu_layers = 99;
    parallel_params.moe_stream_slots = 2;
    parallel_params.moe_stream_io_threads = 2;
    llama_model_ptr parallel_model(llama_model_load_from_file(parallel_path.c_str(), parallel_params));
    GGML_ASSERT(parallel_model != nullptr);
    llama_moe_stream * parallel_stream = parallel_model->moe_stream();
    GGML_ASSERT(parallel_stream != nullptr);
    llama_context_ptr parallel_context(llama_init_from_model(parallel_model.get(), parallel_context_params));
    GGML_ASSERT(parallel_context != nullptr);
    const uint32_t expected_ubatch = parallel_stream->wave_supported ? 64 : 1;
    if (llama_n_ubatch(parallel_context.get()) != expected_ubatch) {
        throw std::runtime_error("MoE stream prefill set n_ubatch to " + std::to_string(llama_n_ubatch(parallel_context.get())) + ", expected " + std::to_string(expected_ubatch));
    }

    llama_context_params parallel_baseline_context_params = parallel_context_params;
    parallel_baseline_context_params.n_ubatch = expected_ubatch;
    parallel_baseline_context.reset(llama_init_from_model(parallel_baseline_model.get(), parallel_baseline_context_params));
    GGML_ASSERT(parallel_baseline_context != nullptr);

    const std::vector<llama_token> parallel_tokens = get_tokens(64, 128, seed + 1);
    const std::vector<float> parallel_expected = get_logits(parallel_baseline_model.get(), parallel_baseline_context.get(), parallel_tokens);
    const std::vector<float> parallel_actual = get_logits(parallel_model.get(), parallel_context.get(), parallel_tokens);

    bool has_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        has_gpu = has_gpu || ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    bool has_gpu_cache = false;
    for (const auto & layer : parallel_stream->layers) {
        if (!layer) {
            continue;
        }
        for (const auto & weight : layer->weights) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(weight.cache->buffer);
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (dev == nullptr) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            GGML_ASSERT(dev != nullptr && buft == ggml_backend_dev_buffer_type(dev));
            has_gpu_cache = has_gpu_cache || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        }
    }
    GGML_ASSERT(!has_gpu || has_gpu_cache);
    llama_file parallel_file(parallel_path.c_str(), "rb", false);
    for (const auto & layer : parallel_stream->layers) {
        if (!layer) {
            continue;
        }
        for (uint32_t slot = 0; slot < layer->n_slots; ++slot) {
            GGML_ASSERT(layer->slot_state[slot] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            const int32_t expert = layer->slot_expert[slot];
            GGML_ASSERT(expert >= 0 && (uint32_t) expert < layer->n_expert);
            for (const auto & weight : layer->weights) {
                std::vector<uint8_t> expected_data(weight.nb_expert);
                std::vector<uint8_t> actual_data(weight.nb_expert);
                parallel_file.seek(weight.offs + (size_t) expert*weight.nb_expert, SEEK_SET);
                parallel_file.read_raw(expected_data.data(), expected_data.size());
                ggml_backend_tensor_get(weight.cache, actual_data.data(), (size_t) slot*weight.nb_expert, actual_data.size());
                GGML_ASSERT(expected_data == actual_data);
            }
        }
    }
    if (parallel_expected != parallel_actual) {
        float max_abs_error = 0.0f;
        size_t mismatch_count = 0;
        size_t first_mismatch = parallel_expected.size();
        for (size_t i = 0; i < parallel_expected.size(); ++i) {
            const float abs_error = std::abs(parallel_expected[i] - parallel_actual[i]);
            max_abs_error = std::max(max_abs_error, abs_error);
            mismatch_count += parallel_expected[i] != parallel_actual[i];
            if (first_mismatch == parallel_expected.size() && parallel_expected[i] != parallel_actual[i]) {
                first_mismatch = i;
            }
        }
        fprintf(stderr, "parallel MoE stream mismatch: NMSE = %.8e, max abs = %.8e, values = %zu/%zu, first token = %zu, first logit = %zu, misses = %" PRId64 "\n", nmse(parallel_expected, parallel_actual), max_abs_error, mismatch_count, parallel_expected.size(), first_mismatch/128, first_mismatch%128, parallel_stream->stats.n_miss);
    }
    GGML_ASSERT(parallel_expected == parallel_actual);
    if (parallel_stream->wave_supported) {
        int64_t n_seen = 0;
        bool exceeded_slots = false;
        for (const auto & layer : parallel_stream->layers) {
            if (!layer) {
                continue;
            }
            const int64_t layer_seen = std::count(layer->seen.begin(), layer->seen.end(), 1);
            n_seen += layer_seen;
            exceeded_slots = exceeded_slots || layer_seen > layer->n_slots;
        }
        GGML_ASSERT(exceeded_slots);
        GGML_ASSERT(parallel_stream->stats.n_miss == parallel_stream->stats.n_miss_cold);
        GGML_ASSERT(parallel_stream->stats.n_miss_cold == n_seen);
    }

    const std::vector<llama_token> continuation = get_tokens(1, 128, seed + 2);
    const std::vector<float> continuation_expected = get_logits(parallel_baseline_model.get(), parallel_baseline_context.get(), continuation, false, 0, 64);
    const std::vector<float> continuation_actual = get_logits(parallel_model.get(), parallel_context.get(), continuation, false, 0, 64);
    GGML_ASSERT(continuation_expected == continuation_actual);
    GGML_ASSERT(parallel_stream->workers.size() == 2);
    GGML_ASSERT(parallel_stream->stats.n_miss > 2);

    parallel_baseline_context.reset();
    parallel_context.reset();
    parallel_baseline_context.reset(llama_init_from_model(parallel_baseline_model.get(), parallel_baseline_context_params));
    parallel_context.reset(llama_init_from_model(parallel_model.get(), parallel_context_params));
    GGML_ASSERT(parallel_baseline_context != nullptr && parallel_context != nullptr);

    const std::vector<llama_token> short_tokens = get_tokens(8, 128, seed + 3);
    const std::vector<float> short_expected = get_logits(parallel_baseline_model.get(), parallel_baseline_context.get(), short_tokens);
    const std::vector<float> short_actual = get_logits(parallel_model.get(), parallel_context.get(), short_tokens);
    GGML_ASSERT(short_expected == short_actual);

    llama_moe_stream * stream = streamed.first->moe_stream();
    GGML_ASSERT(stream != nullptr);
    GGML_ASSERT(stream->stats.n_calls > 0);
    GGML_ASSERT(stream->stats.n_hit > 0);
    GGML_ASSERT(stream->stats.n_miss > stream->stats.n_miss_cold);
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, arch == LLM_ARCH_QWEN4EXP);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    if (target_arch == LLM_ARCH_UNKNOWN || target_arch == LLM_ARCH_QWEN4EXP) {
        test_qwen4exp_ple_metadata_save();
        test_qwen4exp_ple_split_mrope_history();
        test_qwen4exp_ple_shared_prefix();
        test_qwen4exp_indexer_seq_cp();
        test_qwen4exp_qsa_unified_sequences();
        test_qwen4exp_qsa_non_causal();
        test_qwen4exp_qsa_norm_layout();
        test_qwen4exp_qsa_block_semantics();
        test_qwen4exp_qsa_pool_precision();
        test_qwen4exp_moe_stream_exact();
    }

    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
