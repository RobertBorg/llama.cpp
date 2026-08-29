#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct server_prefix_cache_plan {
    // Lengths include the replay token.
    size_t capture_len = 0;
    size_t hit_len     = 0;
};

class server_prefix_cache {
public:
    server_prefix_cache();
    ~server_prefix_cache();

    server_prefix_cache(const server_prefix_cache &) = delete;
    server_prefix_cache & operator=(const server_prefix_cache &) = delete;

    server_prefix_cache(server_prefix_cache &&) noexcept;
    server_prefix_cache & operator=(server_prefix_cache &&) noexcept;

    bool init(
            const std::string & dir,
                         size_t chunk_tokens,
                         size_t max_bytes,
            const std::string & namespace_id,
                llama_context * ctx,
                  llama_seq_id hidden_seq_id) noexcept;

    server_prefix_cache_plan observe(const llama_tokens & tokens, uint64_t request_id) noexcept;

    // On success, processed is the restored prefix without its replay token.
    bool restore(const llama_tokens & tokens, llama_seq_id dest_seq_id, llama_tokens & processed) noexcept;

    // The source must contain exactly prefix_len - 1 decoded prompt tokens.
    bool capture(const llama_tokens & tokens, size_t prefix_len, llama_seq_id source_seq_id) noexcept;

    bool evict_resident() noexcept;

    bool enabled() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
