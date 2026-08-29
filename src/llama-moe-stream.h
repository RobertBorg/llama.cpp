#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_moe_stream;
struct llama_moe_stream_layer;

struct llama_moe_stream_io_range {
    size_t offs = 0;
    size_t head = 0;
    size_t size = 0;
};

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1,
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr;

    uint16_t file_idx  = 0;
    size_t   offs      = 0;
    size_t   nb_expert = 0;
};

struct llama_moe_stream_wave {
    llama_moe_stream_layer * layer = nullptr;

    uint32_t expert_first = 0;
    uint32_t expert_last  = 0;
    uint32_t index        = 0;
    uint32_t count        = 0;
};

struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights;
    std::vector<llama_moe_stream_wave>   waves;

    std::vector<int32_t>                 slot_expert;
    std::vector<uint8_t>                 slot_state;
    std::vector<uint8_t>                 slot_claimed;
    std::vector<uint64_t>                slot_gen;
    std::vector<int64_t>                 slot_last_use;
    std::unordered_map<int32_t, int32_t> expert_slot;

    std::vector<uint32_t> route_hotness;
    std::vector<uint8_t>  seen;
    int64_t use_counter = 0;

    std::vector<int32_t> uniq;
    std::vector<uint8_t>  touched;
    std::vector<uint8_t>  keep;
    std::vector<int32_t> demand_slots;

    bool matches(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down, const ggml_tensor * gate_up) const;
};

struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    uint64_t gen    = 0;
};

struct llama_moe_stream {
    uint32_t n_slots      = 0;
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers;

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    ggml_tensor * create_cache_tensor(int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta, uint16_t file_idx, size_t offs);

    void alloc_bufs(bool no_alloc);
    void open_files(const std::vector<std::string> & paths);

    bool context_acquire();
    void context_release();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown(bool no_alloc) const;
    void print_stats() const;

    bool use_direct_io = false;
    size_t io_alignment = 0;

    llama_files files;
    llama_files buffered_files;
    std::atomic<bool> direct_io_failed { false };

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0;

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    mutable std::mutex      mtx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;

    std::deque<llama_moe_stream_work> q_demand;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;
    bool context_active  = false;
    bool debug           = false;
    bool wave_supported  = false;

    struct {
        int64_t n_calls     = 0;
        int64_t n_hit       = 0;
        int64_t n_miss      = 0;
        int64_t n_miss_cold = 0;
        int64_t t_stall_us  = 0;
    } stats;

    void start_workers_locked();
    void worker_loop();
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);
};

bool llama_moe_stream_backend_supported(const char * registry_name);
bool llama_moe_stream_alignment_supported(size_t alignment);
llama_moe_stream_io_range llama_moe_stream_align_range(size_t offs, size_t size, size_t alignment);
size_t llama_moe_stream_staging_size(size_t nb_expert, size_t alignment);
const uint8_t * llama_moe_stream_pread(llama_file & file, llama_file * fallback, uint8_t * staging, size_t size, size_t offs, bool direct, size_t alignment, bool * fallback_used);

void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata);
void llama_moe_stream_remap_wave(ggml_tensor * dst, const ggml_tensor * src, const ggml_tensor * barrier, int ith, int nth, void * userdata);
