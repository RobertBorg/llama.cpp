#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;
static const size_t   MOE_STREAM_IO_ALIGNMENT_DEFAULT = 4096;

bool llama_moe_stream_alignment_supported(size_t alignment) {
    return alignment > 0 && (alignment & (alignment - 1)) == 0;
}

llama_moe_stream_io_range llama_moe_stream_align_range(size_t offs, size_t size, size_t alignment) {
    GGML_ASSERT(llama_moe_stream_alignment_supported(alignment));
    const size_t aligned_offs = offs & ~(alignment - 1);
    const size_t head = offs - aligned_offs;
    GGML_ASSERT(size <= SIZE_MAX - head);
    const size_t used = head + size;
    GGML_ASSERT(used <= SIZE_MAX - (alignment - 1));
    return { aligned_offs, head, (used + alignment - 1) & ~(alignment - 1) };
}

size_t llama_moe_stream_staging_size(size_t nb_expert, size_t alignment) {
    GGML_ASSERT(llama_moe_stream_alignment_supported(alignment));
    GGML_ASSERT(alignment <= (SIZE_MAX - nb_expert)/2);
    return nb_expert + 2*alignment;
}

static void * moe_stream_aligned_alloc(size_t size, size_t alignment) {
    alignment = std::max(alignment, sizeof(void *));
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void * ptr = nullptr;
    return posix_memalign(&ptr, alignment, size) == 0 ? ptr : nullptr;
#endif
}

static void moe_stream_aligned_free(void * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

bool llama_moe_stream_backend_supported(const char * registry_name) {
    if (registry_name == nullptr) {
        return false;
    }
    for (const char * name : { "CPU", "CUDA", "ROCm", "MTL", "Vulkan" }) {
        if (strcmp(registry_name, name) == 0) {
            return true;
        }
    }
    return false;
}

static const uint8_t * moe_stream_pread_raw(llama_file & file, uint8_t * staging, size_t size, size_t offs, bool direct, size_t alignment) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    GGML_UNUSED(alignment);
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, size);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        const auto range = llama_moe_stream_align_range(offs, size, alignment);
        ssize_t result;
        do {
            result = pread(fd, staging, range.size, range.offs);
        } while (result < 0 && errno == EINTR);
        return result >= 0 && (size_t) result >= range.head + size ? staging + range.head : nullptr;
    }

    uint8_t * dst = staging;
    size_t left = size;
    while (left > 0) {
        const ssize_t result = pread(fd, dst, left, offs);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (result == 0) {
            return nullptr;
        }
        dst  += result;
        offs += (size_t) result;
        left -= (size_t) result;
    }
    return staging;
#endif
}

const uint8_t * llama_moe_stream_pread(llama_file & file, llama_file * fallback, uint8_t * staging, size_t size, size_t offs, bool direct, size_t alignment, bool * fallback_used) {
    if (fallback_used != nullptr) {
        *fallback_used = false;
    }
    const uint8_t * data = moe_stream_pread_raw(file, staging, size, offs, direct, alignment);
    if (data != nullptr || fallback == nullptr) {
        return data;
    }
    if (fallback_used != nullptr) {
        *fallback_used = true;
    }
    return moe_stream_pread_raw(*fallback, staging, size, offs, false, fallback->read_alignment());
}

bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * tensor) {
        for (const auto & weight : weights) {
            if (weight.cache == tensor) {
                return true;
            }
        }
        return false;
    };

    size_t count = 0;
    for (const ggml_tensor * tensor : { gate, up, down, gate_up }) {
        if (tensor == nullptr) {
            continue;
        }
        if (!is_cache(tensor)) {
            return false;
        }
        count++;
    }
    return count > 0 && count == weights.size();
}

llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);
    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);
    use_direct_io = direct;
    io_alignment = MOE_STREAM_IO_ALIGNMENT_DEFAULT;
    debug = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
}

llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
    }
    cv_work.notify_all();
    for (auto & worker : workers) {
        worker.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta, uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert = meta->ne[2];
    const size_t nb_expert = ggml_nbytes(meta)/n_expert;
    GGML_ASSERT(nb_expert*n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & [current_buft, current_ctx] : ctxs) {
        if (current_buft == buft) {
            ctx = current_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create MoE stream tensor context");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert*n_slots);

    auto & layer = layers[il];
    if (!layer) {
        layer = std::make_unique<llama_moe_stream_layer>();
        layer->mgr      = this;
        layer->il       = il;
        layer->n_expert = n_expert;
        layer->n_slots  = n_slots;
        layer->slot_expert.resize(n_slots, -1);
        layer->slot_state.resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        layer->slot_claimed.resize(n_slots, 0);
        layer->slot_gen.resize(n_slots, 0);
        layer->slot_last_use.resize(n_slots, 0);
        layer->route_hotness.resize(n_expert, 0);
        layer->seen.resize(n_expert, 0);
        layer->keep.resize(n_slots, 0);

        const uint32_t n_waves = (n_expert + n_slots - 1)/n_slots;
        layer->waves.resize(n_waves);
        for (uint32_t iw = 0; iw < n_waves; ++iw) {
            layer->waves[iw] = {
                layer.get(),
                iw*n_slots,
                std::min(n_expert, (iw + 1)*n_slots),
                iw,
                n_waves,
            };
        }
    }
    GGML_ASSERT(layer->n_expert == n_expert);

    layer->weights.push_back({ cache, file_idx, offs, nb_expert });
    max_nb_expert = std::max(max_nb_expert, nb_expert);
    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    wave_supported = !ctxs.empty();
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == nullptr) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        ggml_backend_dev_props props = {};
        if (dev == nullptr) {
            wave_supported = false;
        } else {
            ggml_backend_dev_get_props(dev, &props);
            wave_supported = wave_supported && props.caps.mul_mat_id_masked;
        }

        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buffer = nullptr;
        if (no_alloc) {
            buffer = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
                tensor->buffer = buffer;
            }
        } else {
            buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buffer == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buffer);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n", __func__, ggml_backend_buffer_name(buffer), ggml_backend_buffer_get_size(buffer)/1024.0/1024.0, n_slots);
    }

    if (!wave_supported) {
        LLAMA_LOG_WARN("%s: masked expert waves are not supported by the cache backend\n", __func__);
    }
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file path");
        }
    }

    direct_io_failed.store(false, std::memory_order_relaxed);
    buffered_files.clear();

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);
    if (use_direct_io) {
        bool supported = !files.empty();
        for (const auto & file : files) {
            if (!file->has_direct_io() || !llama_moe_stream_alignment_supported(file->read_alignment())) {
                supported = false;
                break;
            }
        }
        if (supported) {
            io_alignment = 1;
            for (const auto & file : files) {
                io_alignment = std::max(io_alignment, file->read_alignment());
            }
            uint8_t * probe = (uint8_t *) moe_stream_aligned_alloc(io_alignment, io_alignment);
            GGML_ASSERT(probe != nullptr);
            for (const auto & file : files) {
                const size_t alignment = file->read_alignment();
                if (llama_moe_stream_pread(*file, nullptr, probe, alignment, 0, true, alignment, nullptr) == nullptr) {
                    supported = false;
                    break;
                }
            }
            moe_stream_aligned_free(probe);
        }
        if (!supported) {
            LLAMA_LOG_WARN("%s: direct I/O unavailable, using buffered expert reads\n", __func__);
            use_direct_io = false;
            io_alignment = MOE_STREAM_IO_ALIGNMENT_DEFAULT;
            open_all(false);
        } else {
            for (const auto & path : paths) {
                buffered_files.emplace_back(new llama_file(path.c_str(), "rb", false));
            }
        }
    }

    int64_t n_streamed = 0;
    for (const auto & layer : layers) {
        n_streamed += layer != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS*n_streamed;
}

bool llama_moe_stream::context_acquire() {
    std::lock_guard<std::mutex> lock(mtx);
    if (context_active) {
        return false;
    }
    context_active = true;
    return true;
}

void llama_moe_stream::context_release() {
    std::lock_guard<std::mutex> lock(mtx);
    GGML_ASSERT(context_active);
    context_active = false;
}

void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; ++i) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

void llama_moe_stream::worker_loop() {
    uint8_t * staging = (uint8_t *) moe_stream_aligned_alloc(llama_moe_stream_staging_size(max_nb_expert, io_alignment), io_alignment);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lock(mtx);
    while (true) {
        cv_work.wait(lock, [&]() { return shutting_down || !q_demand.empty(); });
        if (shutting_down) {
            break;
        }

        const llama_moe_stream_work work = q_demand.front();
        q_demand.pop_front();
        auto & layer = *work.sl;
        if (work.gen != layer.slot_gen[work.slot] || layer.slot_state[work.slot] != LLAMA_MOE_STREAM_SLOT_LOADING || layer.slot_expert[work.slot] != work.expert || layer.slot_claimed[work.slot]) {
            continue;
        }
        layer.slot_claimed[work.slot] = 1;

        lock.unlock();
        bool ok = true;
        for (const auto & weight : layer.weights) {
            const bool direct = use_direct_io && !direct_io_failed.load(std::memory_order_relaxed);
            llama_file * file = direct || !use_direct_io ? files[weight.file_idx].get() : buffered_files[weight.file_idx].get();
            llama_file * fallback = direct ? buffered_files[weight.file_idx].get() : nullptr;
            bool fallback_used = false;
            const uint8_t * data = llama_moe_stream_pread(*file, fallback, staging, weight.nb_expert, weight.offs + (size_t) work.expert*weight.nb_expert, direct, file->read_alignment(), &fallback_used);
            if (fallback_used && !direct_io_failed.exchange(true, std::memory_order_relaxed)) {
                LLAMA_LOG_WARN("%s: direct I/O read failed, using buffered expert reads\n", __func__);
            }
            if (data == nullptr) {
                ok = false;
                break;
            }
            ggml_backend_tensor_set(weight.cache, data, (size_t) work.slot*weight.nb_expert, weight.nb_expert);
        }
        lock.lock();

        layer.slot_claimed[work.slot] = 0;
        if (ok) {
            layer.slot_state[work.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        } else {
            load_failed = true;
        }
        cv_done.notify_all();
    }
    lock.unlock();
    moe_stream_aligned_free(staging);
}

int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & layer, const uint8_t * keep) const {
    int32_t victim = -1;
    for (uint32_t slot = 0; slot < layer.n_slots; ++slot) {
        if ((keep && keep[slot]) || layer.slot_state[slot] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (layer.slot_state[slot] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return slot;
        }
        if (victim < 0) {
            victim = slot;
            continue;
        }
        const uint32_t slot_hotness = layer.route_hotness[layer.slot_expert[slot]];
        const uint32_t victim_hotness = layer.route_hotness[layer.slot_expert[victim]];
        if (slot_hotness < victim_hotness || (slot_hotness == victim_hotness && layer.slot_last_use[slot] < layer.slot_last_use[victim])) {
            victim = slot;
        }
    }
    return victim;
}

void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & layer, int32_t expert, int32_t slot) {
    if (layer.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, layer.il, layer.slot_expert[slot], slot);
        }
        layer.expert_slot.erase(layer.slot_expert[slot]);
    }
    layer.slot_expert[slot] = expert;
    layer.slot_state[slot] = LLAMA_MOE_STREAM_SLOT_LOADING;
    layer.slot_gen[slot]++;
    layer.slot_last_use[slot] = ++layer.use_counter;
    layer.expert_slot[expert] = slot;
    layer.seen[expert] = 1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_stream::memory_breakdown(bool no_alloc) const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    GGML_ASSERT(ctxs.size() == bufs.size());
    for (size_t i = 0; i < ctxs.size(); ++i) {
        const auto buft = ctxs[i].first;
        result[buft] += no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctxs[i].second.get(), buft) : ggml_backend_buffer_get_size(bufs[i].get());
    }
    const size_t staging_size = llama_moe_stream_staging_size(max_nb_expert, io_alignment);
    GGML_ASSERT((size_t) n_io_threads <= SIZE_MAX/staging_size);
    result[ggml_backend_cpu_buffer_type()] += (size_t) n_io_threads*staging_size;
    return result;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);
    const int64_t total = stats.n_hit + stats.n_miss;
    LLAMA_LOG_INFO("%s: MoE stream calls = %" PRId64 ", hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n", __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold, total > 0 ? 100.0*stats.n_hit/total : 0.0);
    LLAMA_LOG_INFO("%s: MoE stream load stall = %.2f ms total (%.3f ms per call)\n", __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
}

static void llama_moe_stream_remap_impl(
        llama_moe_stream_layer * layer,
        ggml_tensor * dst,
        const ggml_tensor * src,
        uint32_t expert_first,
        uint32_t expert_last,
        int32_t sentinel,
        bool first_wave,
        bool last_wave) {
    auto * mgr = layer->mgr;
    GGML_ASSERT(src->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    GGML_ASSERT(expert_first < expert_last && expert_last <= layer->n_expert);

    const int64_t count = ggml_nelements(src);
    const int32_t * ids = (const int32_t *) src->data;
    int32_t * slots = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lock(mgr->mtx);
    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed");
    }
    if (first_wave) {
        mgr->stats.n_calls++;
    }
    mgr->start_workers_locked();

    layer->touched.assign(layer->n_expert, 0);
    layer->uniq.clear();
    for (int64_t i = 0; i < count; ++i) {
        const int32_t expert = ids[i];
        GGML_ASSERT(expert >= 0 && (uint32_t) expert < layer->n_expert);
        if ((uint32_t) expert >= expert_first && (uint32_t) expert < expert_last && !layer->touched[expert]) {
            layer->touched[expert] = 1;
            layer->uniq.push_back(expert);
        }
    }
    std::sort(layer->uniq.begin(), layer->uniq.end());
    if (layer->uniq.size() > layer->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu experts but has %u slots", layer->il, layer->uniq.size(), layer->n_slots);
    }

    for (const int32_t expert : layer->uniq) {
        if (layer->route_hotness[expert] < UINT32_MAX - 1) {
            layer->route_hotness[expert]++;
        }
    }
    if (last_wave && mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & current_layer : mgr->layers) {
            if (current_layer) {
                for (auto & hotness : current_layer->route_hotness) {
                    hotness >>= 1;
                }
            }
        }
    }

    std::fill(layer->keep.begin(), layer->keep.end(), 0);
    layer->demand_slots.clear();
    bool waited = false;
    for (const int32_t expert : layer->uniq) {
        const auto found = layer->expert_slot.find(expert);
        if (found != layer->expert_slot.end()) {
            const int32_t slot = found->second;
            if (layer->slot_state[slot] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->q_demand.push_back({ layer, expert, slot, layer->slot_gen[slot] });
                mgr->cv_work.notify_one();
                waited = true;
            }
            mgr->stats.n_hit++;
            layer->keep[slot] = 1;
            layer->demand_slots.push_back(slot);
            continue;
        }

        int32_t victim = -1;
        while ((victim = mgr->pick_victim_locked(*layer, layer->keep.data())) < 0) {
            mgr->cv_done.wait(lock);
            if (mgr->load_failed) {
                GGML_ABORT("MoE expert streaming: expert load failed");
            }
        }
        if (!layer->seen[expert]) {
            mgr->stats.n_miss_cold++;
        }
        mgr->reserve_slot_locked(*layer, expert, victim);
        mgr->q_demand.push_back({ layer, expert, victim, layer->slot_gen[victim] });
        mgr->cv_work.notify_one();
        mgr->stats.n_miss++;
        waited = true;
        layer->keep[victim] = 1;
        layer->demand_slots.push_back(victim);
    }

    if (waited) {
        const int64_t start = ggml_time_us();
        mgr->cv_done.wait(lock, [&]() {
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t slot : layer->demand_slots) {
                if (layer->slot_state[slot] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed");
        }
        mgr->stats.t_stall_us += ggml_time_us() - start;
    }

    for (int64_t i = 0; i < count; ++i) {
        const int32_t expert = ids[i];
        if ((uint32_t) expert < expert_first || (uint32_t) expert >= expert_last) {
            GGML_ASSERT(sentinel >= 0);
            slots[i] = sentinel;
            continue;
        }
        const int32_t slot = layer->expert_slot.at(expert);
        layer->slot_last_use[slot] = ++layer->use_counter;
        slots[i] = slot;
    }
}

void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * layer = (llama_moe_stream_layer *) userdata;
    llama_moe_stream_remap_impl(layer, dst, src, 0, layer->n_expert, -1, true, true);
}

void llama_moe_stream_remap_wave(
        ggml_tensor * dst,
        const ggml_tensor * src,
        const ggml_tensor * barrier,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(barrier);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * wave = (llama_moe_stream_wave *) userdata;
    llama_moe_stream_remap_impl(
            wave->layer,
            dst,
            src,
            wave->expert_first,
            wave->expert_last,
            wave->layer->n_slots,
            wave->index == 0,
            wave->index + 1 == wave->count);
}
