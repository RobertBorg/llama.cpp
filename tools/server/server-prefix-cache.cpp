#include "server-prefix-cache.h"

#include "json.h"
#include "log.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

namespace fs = std::filesystem;

constexpr size_t SERVER_PREFIX_CACHE_MAX_TOKENS    = 65536;
constexpr size_t SERVER_PREFIX_CACHE_MIN_CHUNK     = 16;
constexpr size_t SERVER_PREFIX_CACHE_MAX_GHOSTS    = 4096;
constexpr size_t SERVER_PREFIX_CACHE_MAX_MANIFEST  = 8 * 1024 * 1024;
constexpr unsigned int SERVER_PREFIX_CACHE_VERSION = 2;

const char * SERVER_PREFIX_CACHE_FORMAT = "llama.cpp/server-prefix-cache";
const char * SERVER_PREFIX_CACHE_STATE_FORMAT = "llama.cpp/server-prefix-cache-state";

struct prefix_cache_entry {
    std::string namespace_id;
    std::string key;
    llama_tokens tokens;
    size_t processed_tokens = 0;
    llama_token replay_token = LLAMA_TOKEN_NULL;

    std::string blob_name;
    std::string blob_sha256;
    size_t blob_size = 0;
    fs::path blob_path;

    uint64_t references     = 0;
    uint64_t last_access_ms = 0;
    uint64_t last_request   = std::numeric_limits<uint64_t>::max();
    size_t saved_tokens     = 0;
    double gdsf_score       = 0.0;

    fs::path manifest_path;
};

struct prefix_cache_ghost {
    uint64_t last_request = 0;
    uint64_t last_touch   = 0;
    unsigned int hits     = 0;
};

enum class anchor_load_result {
    loaded,
    unavailable,
    corrupt,
};

enum class blob_hash_result {
    valid,
    unavailable,
    corrupt,
};

enum class file_read_result {
    valid,
    unavailable,
    corrupt,
};

enum class manifest_parse_result {
    valid,
    unavailable,
    unsupported,
    corrupt,
};

static std::string path_to_utf8(const fs::path & path) {
    const auto value = path.generic_u8string();
    return std::string(value.begin(), value.end());
}

static uint64_t now_ms() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static bool is_sha256(const std::string & value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

static bool is_decimal(const std::string & value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
        return c >= '0' && c <= '9';
    });
}

static bool is_managed_temp(const std::string & filename) {
    const size_t marker = filename.rfind(".tmp.");
    if (marker == std::string::npos) {
        return false;
    }

    const std::string target = filename.substr(0, marker);
    const std::string suffix = filename.substr(marker + 5);
    const size_t separator = suffix.find('.');
    if (separator == std::string::npos || !is_decimal(suffix.substr(0, separator)) ||
            !is_decimal(suffix.substr(separator + 1))) {
        return false;
    }

    if (target == ".prefix-cache-state") {
        return true;
    }
    const fs::path path = fs::u8path(target);
    const std::string stem = path_to_utf8(path.stem());
    return is_sha256(stem) && (path.extension() == ".json" || path.extension() == ".ggsq");
}

static void hash_u64_le(sha256_t & hash, uint64_t value) {
    unsigned char data[8];
    for (unsigned int i = 0; i < 8; ++i) {
        data[i] = (unsigned char) (value >> (i * 8));
    }
    sha256_update(&hash, data, sizeof(data));
}

static void hash_u32_le(sha256_t & hash, uint32_t value) {
    unsigned char data[4];
    for (unsigned int i = 0; i < 4; ++i) {
        data[i] = (unsigned char) (value >> (i * 8));
    }
    sha256_update(&hash, data, sizeof(data));
}

static sha256_t prefix_hash_begin(const std::string & namespace_id) {
    static const unsigned char domain[] = "llama-server-prefix-cache-v2";

    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, domain, sizeof(domain));
    hash_u64_le(hash, namespace_id.size());
    sha256_update(&hash, reinterpret_cast<const unsigned char *>(namespace_id.data()), namespace_id.size());
    return hash;
}

static std::string prefix_hash_finish(const sha256_t & prefix_hash, size_t prefix_len) {
    static const char hex[] = "0123456789abcdef";

    sha256_t hash = prefix_hash;
    hash_u64_le(hash, prefix_len);
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);

    std::string result;
    result.reserve(SHA256_DIGEST_SIZE * 2);
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        result.push_back(hex[digest[i] >> 4]);
        result.push_back(hex[digest[i] & 0x0f]);
    }
    return result;
}

static std::string prefix_key(
        const std::string & namespace_id,
        const llama_tokens & tokens,
                     size_t prefix_len) {
    sha256_t hash = prefix_hash_begin(namespace_id);
    for (size_t i = 0; i < prefix_len; ++i) {
        hash_u32_le(hash, (uint32_t) tokens[i]);
    }
    return prefix_hash_finish(hash, prefix_len);
}

static double retention_score(const prefix_cache_entry & entry) {
    if (entry.blob_size == 0) {
        return 0.0;
    }
    const long double value = (long double) entry.references * (long double) entry.saved_tokens / (long double) entry.blob_size;
    return value >= std::numeric_limits<double>::max() ? std::numeric_limits<double>::max() : (double) value;
}

static double gdsf_priority(const prefix_cache_entry & entry, double inflation) {
    const long double value = (long double) inflation + (long double) retention_score(entry);
    const double result = value >= std::numeric_limits<double>::max() ? std::numeric_limits<double>::max() : (double) value;
    return std::max(entry.gdsf_score, result);
}

static bool prefix_matches(const prefix_cache_entry & entry, const llama_tokens & tokens) {
    return entry.tokens.size() <= tokens.size() &&
        std::equal(entry.tokens.begin(), entry.tokens.end(), tokens.begin());
}

static file_read_result read_text_file(const fs::path & path, std::string & data) {
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) {
        return file_read_result::unavailable;
    }
    if (size > SERVER_PREFIX_CACHE_MAX_MANIFEST) {
        return file_read_result::corrupt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return file_read_result::unavailable;
    }

    data.resize((size_t) size);
    if (size != 0) {
        file.read(&data[0], (std::streamsize) size);
    }
    if ((uintmax_t) file.gcount() == size) {
        return file_read_result::valid;
    }
    return file.bad() ? file_read_result::unavailable : file_read_result::corrupt;
}

static std::string digest_to_hex(const unsigned char * digest) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(SHA256_DIGEST_SIZE * 2);
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        result.push_back(hex[digest[i] >> 4]);
        result.push_back(hex[digest[i] & 0x0f]);
    }
    return result;
}

static blob_hash_result hash_blob_file_checked(const fs::path & path, size_t expected_size, std::string & checksum) {
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) {
        return ec == std::errc::no_such_file_or_directory ? blob_hash_result::corrupt : blob_hash_result::unavailable;
    }
    if (size != expected_size) {
        return blob_hash_result::corrupt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return blob_hash_result::unavailable;
    }

    sha256_t hash;
    sha256_init(&hash);

    std::array<unsigned char, 64 * 1024> buffer;
    size_t offset = 0;
    while (offset < expected_size) {
        const size_t count = std::min(expected_size - offset, buffer.size());
        file.read(reinterpret_cast<char *>(buffer.data()), (std::streamsize) count);
        if ((size_t) file.gcount() != count) {
            return file.bad() ? blob_hash_result::unavailable : blob_hash_result::corrupt;
        }
        sha256_update(&hash, buffer.data(), count);
        offset += count;
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    checksum = digest_to_hex(digest);
    return blob_hash_result::valid;
}

static blob_hash_result probe_blob_file(const fs::path & path, size_t expected_size) {
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) {
        return ec == std::errc::no_such_file_or_directory ? blob_hash_result::corrupt : blob_hash_result::unavailable;
    }
    if (size != expected_size) {
        return blob_hash_result::corrupt;
    }
    std::ifstream file(path, std::ios::binary);
    return file ? blob_hash_result::valid : blob_hash_result::unavailable;
}

static bool hash_blob_file(const fs::path & path, size_t expected_size, std::string & checksum) {
    return hash_blob_file_checked(path, expected_size, checksum) == blob_hash_result::valid;
}

static fs::path temp_path(const fs::path & path) {
    static std::atomic<uint64_t> sequence { 0 };
    fs::path result = path;
    result += ".tmp." + std::to_string(now_ms()) + "." + std::to_string(sequence.fetch_add(1));
    return result;
}

static bool rename_replace(const fs::path & source, const fs::path & destination, std::error_code & ec) {
#ifdef _WIN32
    if (MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        ec.clear();
        return true;
    }
    ec = std::error_code((int) GetLastError(), std::system_category());
    return false;
#else
    fs::rename(source, destination, ec);
    return !ec;
#endif
}

static bool write_text_atomic(const fs::path & path, const std::string & data) {
    const fs::path tmp = temp_path(path);
    {
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file) {
            return false;
        }
        file.write(data.data(), (std::streamsize) data.size());
        file.flush();
        if (!file) {
            std::error_code remove_ec;
            fs::remove(tmp, remove_ec);
            return false;
        }
    }

    std::error_code ec;
    if (!rename_replace(tmp, path, ec)) {
        std::error_code remove_ec;
        fs::remove(tmp, remove_ec);
        return false;
    }
    return true;
}

static common_json entry_manifest(const prefix_cache_entry & entry) {
    return common_json {
        {"format",           SERVER_PREFIX_CACHE_FORMAT},
        {"version",          SERVER_PREFIX_CACHE_VERSION},
        {"namespace",        entry.namespace_id},
        {"key",              entry.key},
        {"tokens",           entry.tokens},
        {"processed_tokens", (unsigned long long) entry.processed_tokens},
        {"replay_token",     (int) entry.replay_token},
        {"blob", common_json {
            {"file",   entry.blob_name},
            {"size",   (unsigned long long) entry.blob_size},
            {"sha256", entry.blob_sha256},
        }},
        {"retention", common_json {
            {"references",     (unsigned long long) entry.references},
            {"last_access_ms", (unsigned long long) entry.last_access_ms},
            {"saved_tokens",   (unsigned long long) entry.saved_tokens},
            {"gdsf_score",     entry.gdsf_score},
        }},
    };
}

static void log_warning(const std::string & message) {
    LOG_WRN("prefix cache: %s\n", message.c_str());
}

} // namespace

struct server_prefix_cache::impl {
    fs::path dir;
    size_t chunk_tokens = 0;
    size_t max_bytes = 0;
    std::string namespace_id;
    llama_context * ctx = nullptr;
    llama_seq_id hidden_seq_id = -1;
    bool ready = false;

    std::unordered_map<std::string, prefix_cache_entry> entries;
    std::unordered_map<std::string, prefix_cache_ghost> ghosts;
    uint64_t ghost_clock = 0;
    double gdsf_inflation = 0.0;
    std::string resident_key;

    fs::path state_path() const {
        return dir / ".prefix-cache-state";
    }

    bool load_state() {
        std::error_code ec;
        if (!fs::exists(state_path(), ec)) {
            return !ec;
        }

        try {
            std::string text;
            const file_read_result read_result = read_text_file(state_path(), text);
            if (read_result == file_read_result::unavailable) {
                return false;
            }
            if (read_result == file_read_result::corrupt) {
                fs::remove(state_path(), ec);
                return !ec;
            }
            const common_json root = common_json::parse(text);
            if (!root.is_object() || root.at("format").get<std::string>() != SERVER_PREFIX_CACHE_STATE_FORMAT ||
                    root.at("version").get<unsigned int>() != SERVER_PREFIX_CACHE_VERSION) {
                return false;
            }
            const double inflation = root.at("inflation").get<double>();
            if (!std::isfinite(inflation) || inflation < 0.0) {
                fs::remove(state_path(), ec);
                return !ec;
            }
            gdsf_inflation = inflation;
            return true;
        } catch (...) {
            fs::remove(state_path(), ec);
            return !ec;
        }
    }

    bool write_state(double inflation) const {
        return write_text_atomic(state_path(), common_json {
            {"format", SERVER_PREFIX_CACHE_STATE_FORMAT},
            {"version", SERVER_PREFIX_CACHE_VERSION},
            {"inflation", inflation},
        }.dump(2));
    }

    manifest_parse_result parse_manifest(const fs::path & path, prefix_cache_entry & entry, std::string & error) const {
        try {
            std::string text;
            const file_read_result read_result = read_text_file(path, text);
            if (read_result == file_read_result::unavailable) {
                error = "cannot read manifest";
                return manifest_parse_result::unavailable;
            }
            if (read_result == file_read_result::corrupt) {
                error = "manifest is too large or incomplete";
                return manifest_parse_result::corrupt;
            }

            const common_json root = common_json::parse(text);
            if (!root.is_object()) {
                error = "manifest root is not an object";
                return manifest_parse_result::corrupt;
            }
            if (root.at("format").get<std::string>() != SERVER_PREFIX_CACHE_FORMAT ||
                    root.at("version").get<unsigned int>() != SERVER_PREFIX_CACHE_VERSION) {
                error = "unknown manifest format";
                return manifest_parse_result::unsupported;
            }
            entry.namespace_id = root.at("namespace").get<std::string>();
            if (entry.namespace_id.empty()) {
                error = "empty namespace";
                return manifest_parse_result::corrupt;
            }

            entry.key    = root.at("key").get<std::string>();
            entry.tokens = root.at("tokens").get<llama_tokens>();
            if (!is_sha256(entry.key) || entry.tokens.size() < 2 || entry.tokens.size() > SERVER_PREFIX_CACHE_MAX_TOKENS) {
                error = "invalid key or token count";
                return manifest_parse_result::corrupt;
            }
            if (path.filename() != fs::u8path(entry.key + ".json") ||
                    prefix_key(entry.namespace_id, entry.tokens, entry.tokens.size()) != entry.key) {
                error = "manifest key mismatch";
                return manifest_parse_result::corrupt;
            }

            const unsigned long long processed = root.at("processed_tokens").get<unsigned long long>();
            entry.replay_token = (llama_token) root.at("replay_token").get<int>();
            if (processed != entry.tokens.size() - 1 || entry.replay_token != entry.tokens.back()) {
                error = "invalid replay boundary";
                return manifest_parse_result::corrupt;
            }
            entry.processed_tokens = (size_t) processed;

            const common_json & blob = root.at("blob");
            entry.blob_name   = blob.at("file").get<std::string>();
            entry.blob_sha256 = blob.at("sha256").get<std::string>();
            const unsigned long long blob_size = blob.at("size").get<unsigned long long>();
            if (!is_sha256(entry.blob_sha256) || entry.blob_name != entry.blob_sha256 + ".ggsq" ||
                    blob_size == 0 || blob_size > std::numeric_limits<size_t>::max()) {
                error = "invalid blob metadata";
                return manifest_parse_result::corrupt;
            }
            entry.blob_size = (size_t) blob_size;

            const common_json & retention = root.at("retention");
            entry.references     = retention.at("references").get<unsigned long long>();
            entry.last_access_ms = retention.at("last_access_ms").get<unsigned long long>();
            const unsigned long long saved_tokens = retention.at("saved_tokens").get<unsigned long long>();
            const double persisted_score = retention.at("gdsf_score").get<double>();
            if (entry.references == 0 || saved_tokens != entry.processed_tokens ||
                    !std::isfinite(persisted_score) || persisted_score < 0.0) {
                error = "invalid retention metadata";
                return manifest_parse_result::corrupt;
            }
            entry.saved_tokens = (size_t) saved_tokens;
            entry.gdsf_score = persisted_score;
            if (entry.gdsf_score + std::numeric_limits<double>::epsilon() < retention_score(entry)) {
                error = "invalid retention score";
                return manifest_parse_result::corrupt;
            }
            entry.manifest_path = path;

            entry.blob_path = path.parent_path() / fs::u8path(entry.blob_name);
            return manifest_parse_result::valid;
        } catch (const std::exception & exception) {
            error = exception.what();
            return manifest_parse_result::corrupt;
        } catch (...) {
            error = "unknown manifest error";
            return manifest_parse_result::corrupt;
        }
    }

    bool initialize(
            const std::string & directory,
                         size_t chunk,
                         size_t budget_bytes,
            const std::string & compatibility_namespace,
                llama_context * context,
                  llama_seq_id hidden_id) {
        if (directory.empty() || chunk < SERVER_PREFIX_CACHE_MIN_CHUNK || chunk > SERVER_PREFIX_CACHE_MAX_TOKENS ||
                compatibility_namespace.empty() || context == nullptr || hidden_id < 0) {
            log_warning("invalid configuration; persistent cache is disabled");
            return false;
        }

        dir           = fs::u8path(directory);
        chunk_tokens  = chunk;
        max_bytes     = budget_bytes;
        namespace_id  = compatibility_namespace;
        ctx            = context;
        hidden_seq_id  = hidden_id;

        std::error_code ec;
        const bool directory_existed = fs::exists(dir, ec);
        if (ec) {
            log_warning("cannot inspect cache directory " + directory);
            return false;
        }
        fs::create_directories(dir, ec);
        if (ec || !fs::is_directory(dir, ec) || ec) {
            log_warning("cannot create cache directory " + directory);
            return false;
        }
#ifndef _WIN32
        if (!directory_existed) {
            fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
        }
        const fs::perms permissions = fs::status(dir, ec).permissions();
        if (ec || (permissions & (fs::perms::group_all | fs::perms::others_all)) != fs::perms::none) {
            log_warning("cache directory must only be accessible by its owner");
            return false;
        }
#endif
        if (!load_state()) {
            log_warning("cannot read the prefix cache retention state");
            return false;
        }

        bool scan_safe = true;
        fs::directory_iterator current(dir, ec);
        const fs::directory_iterator end;
        while (!ec && current != end) {
            const fs::directory_entry item = *current;
            std::error_code item_ec;
            if (item.is_symlink(item_ec) && !item_ec) {
                scan_safe = false;
                log_warning("symbolic links are not supported in the prefix cache directory");
            } else if (item.is_regular_file(item_ec) && !item_ec && item.path().extension() == ".json" &&
                    is_sha256(path_to_utf8(item.path().stem()))) {
                prefix_cache_entry entry;
                std::string error;
                const manifest_parse_result result = parse_manifest(item.path(), entry, error);
                if (result == manifest_parse_result::valid) {
                    const auto inserted = entries.emplace(entry.key, std::move(entry));
                    if (!inserted.second) {
                        fs::remove(item.path(), item_ec);
                        scan_safe = scan_safe && !item_ec;
                        log_warning("duplicate manifest " + path_to_utf8(item.path().filename()));
                    }
                } else if (result == manifest_parse_result::corrupt) {
                    fs::remove(item.path(), item_ec);
                    scan_safe = scan_safe && !item_ec;
                    log_warning("removed corrupt manifest " + path_to_utf8(item.path().filename()) + ": " + error);
                } else {
                    scan_safe = false;
                    log_warning("cannot use manifest " + path_to_utf8(item.path().filename()) + ": " + error);
                }
            } else if (item_ec) {
                scan_safe = false;
            }
            current.increment(ec);
        }
        if (ec) {
            log_warning("directory scan stopped: " + ec.message());
        }
        if (ec || !scan_safe) {
            log_warning("persistent cache is disabled because the manifest scan was incomplete");
            return false;
        }
        if (!cleanup_orphan_blobs()) {
            return false;
        }
        if (!enforce_budget()) {
            return false;
        }

        llama_memory_seq_rm(llama_get_memory(ctx), hidden_seq_id, -1, -1);
        ready = true;
        const size_t compatible = std::count_if(entries.begin(), entries.end(), [&](const auto & item) {
            return item.second.namespace_id == namespace_id;
        });
        LOG_INF("prefix cache: loaded %zu compatible checkpoint(s), %zu total from %s\n",
            compatible, entries.size(), path_to_utf8(dir).c_str());
        return true;
    }

    size_t longest_hit(const llama_tokens & tokens) const {
        size_t best = 0;
        for (const auto & item : entries) {
            const prefix_cache_entry & entry = item.second;
            if (entry.namespace_id == namespace_id && entry.tokens.size() > best && prefix_matches(entry, tokens)) {
                best = entry.tokens.size();
            }
        }
        return best;
    }

    static void add_bytes(size_t & total, uintmax_t value) {
        if (value > std::numeric_limits<size_t>::max() - total) {
            total = std::numeric_limits<size_t>::max();
        } else {
            total += (size_t) value;
        }
    }

    bool disk_usage(size_t & total) const {
        total = 0;
        std::error_code ec;
        fs::recursive_directory_iterator current(dir, ec);
        const fs::recursive_directory_iterator end;
        while (!ec && current != end) {
            const fs::directory_entry item = *current;
            std::error_code item_ec;
            if (item.is_regular_file(item_ec) && !item_ec) {
                add_bytes(total, item.file_size(item_ec));
                if (item_ec) {
                    return false;
                }
            } else if (item_ec) {
                return false;
            }
            current.increment(ec);
        }
        return !ec;
    }

    bool remove_blob_if_unreferenced(const fs::path & blob_path) const {
        const bool referenced = std::any_of(entries.begin(), entries.end(), [&](const auto & item) {
            return item.second.blob_path == blob_path;
        });
        if (!referenced) {
            std::error_code ec;
            fs::remove(blob_path, ec);
            if (ec) {
                log_warning("cannot remove unreferenced blob " + path_to_utf8(blob_path));
                return false;
            }
        }
        return true;
    }

    bool cleanup_orphan_blobs() const {
        std::vector<fs::path> blob_paths;
        std::error_code ec;
        fs::directory_iterator current(dir, ec);
        const fs::directory_iterator end;
        while (!ec && current != end) {
            const fs::directory_entry item = *current;
            std::error_code item_ec;
            const std::string filename = path_to_utf8(item.path().filename());
            const std::string stem = path_to_utf8(item.path().stem());
            const bool regular = item.is_regular_file(item_ec);
            if (regular && !item_ec && is_managed_temp(filename)) {
                const auto mtime = item.last_write_time(item_ec);
                if (!item_ec && mtime < fs::file_time_type::clock::now() - std::chrono::hours(1)) {
                    fs::remove(item.path(), item_ec);
                    if (item_ec) {
                        log_warning("cannot remove stale temporary file " + path_to_utf8(item.path()));
                    }
                }
            } else if (regular && !item_ec && item.path().extension() == ".ggsq" && is_sha256(stem)) {
                blob_paths.push_back(item.path());
            }
            current.increment(ec);
        }
        if (ec) {
            log_warning("orphan blob scan stopped: " + ec.message());
            return false;
        }
        bool success = true;
        for (const fs::path & blob_path : blob_paths) {
            success = remove_blob_if_unreferenced(blob_path) && success;
        }
        return success;
    }

    bool remove_entry(const std::string & key) {
        auto found = entries.find(key);
        if (found == entries.end()) {
            return true;
        }

        const fs::path blob_path = found->second.blob_path;
        const fs::path manifest_path = found->second.manifest_path;
        std::error_code ec;
        if (fs::exists(manifest_path, ec) && !ec && !fs::remove(manifest_path, ec)) {
            return false;
        }
        if (ec) {
            return false;
        }

        if (resident_key == key) {
            clear_anchor();
        }
        entries.erase(found);

        return remove_blob_if_unreferenced(blob_path);
    }

    bool enforce_budget() {
        if (max_bytes == 0) {
            return true;
        }

        size_t usage = 0;
        if (!disk_usage(usage)) {
            log_warning("cannot measure the prefix cache byte usage");
            return false;
        }

        std::unordered_set<std::string> failed;
        while (!entries.empty() && usage > max_bytes) {
            auto victim = entries.end();
            for (auto current = entries.begin(); current != entries.end(); ++current) {
                if (failed.count(current->first) != 0) {
                    continue;
                }
                if (victim == entries.end() || current->second.gdsf_score < victim->second.gdsf_score ||
                        (current->second.gdsf_score == victim->second.gdsf_score &&
                         (current->second.last_access_ms < victim->second.last_access_ms ||
                          (current->second.last_access_ms == victim->second.last_access_ms && current->first < victim->first)))) {
                    victim = current;
                }
            }
            if (victim == entries.end()) {
                log_warning("cannot enforce the prefix cache byte budget");
                return false;
            }

            const std::string key = victim->first;
            const double score = victim->second.gdsf_score;
            const uint64_t references = victim->second.references;
            const size_t saved_tokens = victim->second.saved_tokens;
            const size_t blob_size = victim->second.blob_size;
            const double next_inflation = std::max(gdsf_inflation, score);
            if (next_inflation != gdsf_inflation && !write_state(next_inflation)) {
                log_warning("cannot persist the prefix cache retention state");
                return false;
            }
            gdsf_inflation = next_inflation;
            if (!remove_entry(key)) {
                failed.insert(key);
            }
            LOG_INF("prefix cache: evicted %s (score %.9g, references %llu, saved tokens %zu, blob bytes %zu)\n",
                key.c_str(), score, (unsigned long long) references, saved_tokens, blob_size);
            if (!disk_usage(usage)) {
                log_warning("cannot measure the prefix cache byte usage");
                return false;
            }
        }
        return usage <= max_bytes;
    }

    void evict_ghost_if_needed() {
        if (ghosts.size() < SERVER_PREFIX_CACHE_MAX_GHOSTS) {
            return;
        }
        auto victim = ghosts.begin();
        for (auto current = ghosts.begin(); current != ghosts.end(); ++current) {
            if (current->second.last_touch < victim->second.last_touch) {
                victim = current;
            }
        }
        ghosts.erase(victim);
    }

    server_prefix_cache_plan observe(const llama_tokens & tokens, uint64_t request_id) {
        server_prefix_cache_plan plan;
        if (!ready) {
            return plan;
        }

        plan.hit_len = longest_hit(tokens);
        bool touched = false;
        bool removed_corrupt = false;
        if (plan.hit_len > 0) {
            for (auto item = entries.begin(); item != entries.end(); ++item) {
                auto & entry = item->second;
                if (entry.namespace_id == namespace_id && entry.tokens.size() == plan.hit_len &&
                        prefix_matches(entry, tokens) && entry.last_request != request_id) {
                    const blob_hash_result probe_result = probe_blob_file(entry.blob_path, entry.blob_size);
                    if (probe_result == blob_hash_result::corrupt) {
                        const std::string key = item->first;
                        if (!remove_entry(key)) {
                            log_warning("cannot remove corrupt checkpoint " + key);
                        }
                        removed_corrupt = true;
                        break;
                    }
                    if (probe_result == blob_hash_result::unavailable) {
                        break;
                    }
                    entry.last_request = request_id;
                    if (entry.references != std::numeric_limits<uint64_t>::max()) {
                        ++entry.references;
                    }
                    entry.last_access_ms = now_ms();
                    entry.gdsf_score = gdsf_priority(entry, gdsf_inflation);
                    try {
                        if (!write_manifest(entry)) {
                            log_warning("cannot update manifest " + entry.key);
                        }
                    } catch (...) {
                        log_warning("cannot update manifest " + entry.key);
                    }
                    touched = true;
                    break;
                }
            }
        }
        if (touched) {
            enforce_budget();
            plan.hit_len = longest_hit(tokens);
        } else if (removed_corrupt) {
            plan.hit_len = longest_hit(tokens);
        }
        const size_t limit = std::min(tokens.size(), SERVER_PREFIX_CACHE_MAX_TOKENS);
        sha256_t hash = prefix_hash_begin(namespace_id);
        for (size_t i = 0; i < limit; ++i) {
            hash_u32_le(hash, (uint32_t) tokens[i]);
            const size_t prefix_len = i + 1;
            if (prefix_len % chunk_tokens != 0 || prefix_len <= plan.hit_len) {
                continue;
            }

            const std::string key = prefix_hash_finish(hash, prefix_len);
            const auto existing = entries.find(key);
            if (existing != entries.end() && existing->second.namespace_id == namespace_id &&
                    prefix_matches(existing->second, tokens)) {
                ghosts.erase(key);
                continue;
            }

            auto ghost = ghosts.find(key);
            if (ghost == ghosts.end()) {
                evict_ghost_if_needed();
                ghost = ghosts.emplace(key, prefix_cache_ghost { request_id, ++ghost_clock, 1 }).first;
            } else {
                ghost->second.last_touch = ++ghost_clock;
                if (ghost->second.last_request != request_id) {
                    ghost->second.last_request = request_id;
                    ghost->second.hits = std::min(2u, ghost->second.hits + 1);
                }
            }
            if (ghost->second.hits >= 2) {
                plan.capture_len = prefix_len;
            }
        }
        return plan;
    }

    bool write_manifest(prefix_cache_entry & entry) {
        const std::string contents = entry_manifest(entry).dump(2);
        return write_text_atomic(entry.manifest_path, contents);
    }

    void clear_anchor() {
        if (ctx != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx), hidden_seq_id, -1, -1);
        }
        resident_key.clear();
    }

    anchor_load_result load_anchor(const prefix_cache_entry & entry) {
        if (resident_key == entry.key) {
            return anchor_load_result::loaded;
        }
        clear_anchor();
        std::string checksum;
        const blob_hash_result hash_result = hash_blob_file_checked(entry.blob_path, entry.blob_size, checksum);
        if (hash_result == blob_hash_result::unavailable) {
            log_warning("blob is unavailable for " + entry.key);
            return anchor_load_result::unavailable;
        }
        if (hash_result == blob_hash_result::corrupt || checksum != entry.blob_sha256) {
            log_warning("blob checksum failed for " + entry.key);
            return anchor_load_result::corrupt;
        }

        const std::string blob_path_str = path_to_utf8(entry.blob_path);
        llama_tokens loaded(entry.processed_tokens);
        size_t token_count = 0;
        const size_t nread = llama_state_seq_load_file(
            ctx, blob_path_str.c_str(), hidden_seq_id, loaded.data(), loaded.size(), &token_count);
        if (nread == 0) {
            clear_anchor();
            std::string retry_checksum;
            const blob_hash_result retry = hash_blob_file_checked(entry.blob_path, entry.blob_size, retry_checksum);
            if (retry == blob_hash_result::corrupt ||
                    (retry == blob_hash_result::valid && retry_checksum != entry.blob_sha256)) {
                log_warning("GGSQ state became corrupt for " + entry.key);
                return anchor_load_result::corrupt;
            }
            log_warning("GGSQ state load is unavailable for " + entry.key);
            return anchor_load_result::unavailable;
        }
        if (token_count != entry.processed_tokens) {
            clear_anchor();
            log_warning("GGSQ token count failed for " + entry.key);
            return anchor_load_result::corrupt;
        }
        loaded.resize(token_count);
        const bool tokens_match = loaded.size() == entry.processed_tokens &&
            std::equal(loaded.begin(), loaded.end(), entry.tokens.begin());
        if (nread != entry.blob_size || !tokens_match) {
            clear_anchor();
            log_warning("GGSQ payload failed for " + entry.key);
            return anchor_load_result::corrupt;
        }

        resident_key = entry.key;
        return anchor_load_result::loaded;
    }

    bool restore(const llama_tokens & tokens, llama_seq_id destination, llama_tokens & processed) {
        processed.clear();
        if (!ready || destination < 0 || destination == hidden_seq_id) {
            return false;
        }

        std::vector<std::pair<size_t, std::string>> candidates;
        for (const auto & item : entries) {
            if (item.second.namespace_id == namespace_id && prefix_matches(item.second, tokens)) {
                candidates.emplace_back(item.second.tokens.size(), item.first);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto & left, const auto & right) {
            return left.first > right.first;
        });

        for (const auto & candidate : candidates) {
            auto found = entries.find(candidate.second);
            if (found == entries.end()) {
                continue;
            }
            prefix_cache_entry & entry = found->second;
            const anchor_load_result load_result = load_anchor(entry);
            if (load_result != anchor_load_result::loaded) {
                if (load_result == anchor_load_result::corrupt && !remove_entry(candidate.second)) {
                    log_warning("cannot remove corrupt checkpoint " + candidate.second);
                }
                continue;
            }

            llama_tokens restored(entry.tokens.begin(), entry.tokens.end() - 1);
            llama_memory_t memory = llama_get_memory(ctx);
            if (!llama_memory_seq_rm(memory, destination, -1, -1)) {
                log_warning("cannot clear destination sequence");
                return false;
            }
            llama_memory_seq_cp(memory, hidden_seq_id, destination, -1, -1);
            processed = std::move(restored);

            return true;
        }
        return false;
    }

    bool publish_blob(const fs::path & tmp, const fs::path & final_path, size_t size, const std::string & checksum) {
        std::error_code ec;
        if (fs::exists(final_path, ec) && !ec) {
            std::string existing_checksum;
            if (hash_blob_file(final_path, size, existing_checksum) && existing_checksum == checksum) {
                fs::remove(tmp, ec);
                return true;
            }
            log_warning("replacing invalid content-addressed blob " + path_to_utf8(final_path.filename()));
        }
        ec.clear();
        if (!rename_replace(tmp, final_path, ec)) {
            if (fs::exists(final_path, ec) && !ec) {
                std::string existing_checksum;
                if (hash_blob_file(final_path, size, existing_checksum) && existing_checksum == checksum) {
                    fs::remove(tmp, ec);
                    return true;
                }
            }
            fs::remove(tmp, ec);
            return false;
        }
        return true;
    }

    bool capture(const llama_tokens & tokens, size_t prefix_len, llama_seq_id source) {
        if (!ready || source < 0 || source == hidden_seq_id || prefix_len < 2 ||
                prefix_len > tokens.size() || prefix_len > SERVER_PREFIX_CACHE_MAX_TOKENS || prefix_len % chunk_tokens != 0) {
            return false;
        }

        const std::string key = prefix_key(namespace_id, tokens, prefix_len);
        auto present = entries.find(key);
        if (present != entries.end() && prefix_matches(present->second, tokens)) {
            ghosts.erase(key);
            return true;
        }
        const auto ghost = ghosts.find(key);
        if (ghost == ghosts.end() || ghost->second.hits < 2) {
            log_warning("capture was not admitted by two distinct requests");
            return false;
        }

        const size_t processed_count = prefix_len - 1;
        const llama_pos expected_pos = (llama_pos) processed_count - 1;
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), source) != expected_pos) {
            log_warning("capture source is not at the requested prefix boundary");
            return false;
        }

        llama_tokens processed(tokens.begin(), tokens.begin() + processed_count);
        const fs::path state_tmp = temp_path(dir / fs::u8path(key + ".ggsq"));
        const std::string state_tmp_str = path_to_utf8(state_tmp);
        const size_t state_size = llama_state_seq_save_file(
            ctx, state_tmp_str.c_str(), source, processed.data(), processed.size());
        if (state_size == 0) {
            std::error_code ec;
            fs::remove(state_tmp, ec);
            log_warning("cannot save GGSQ state for " + key);
            return false;
        }

        std::string checksum;
        if (!hash_blob_file(state_tmp, state_size, checksum)) {
            std::error_code ec;
            fs::remove(state_tmp, ec);
            log_warning("cannot read saved GGSQ state for " + key);
            return false;
        }
        const std::string blob_name = checksum + ".ggsq";
        if (!publish_blob(state_tmp, dir / fs::u8path(blob_name), state_size, checksum)) {
            log_warning("cannot publish GGSQ state for " + key);
            return false;
        }

        prefix_cache_entry entry;
        entry.namespace_id     = namespace_id;
        entry.key              = key;
        entry.tokens.assign(tokens.begin(), tokens.begin() + prefix_len);
        entry.processed_tokens = processed_count;
        entry.replay_token     = tokens[prefix_len - 1];
        entry.blob_name        = blob_name;
        entry.blob_sha256      = checksum;
        entry.blob_size        = state_size;
        entry.references       = 2;
        entry.last_access_ms   = now_ms();
        entry.saved_tokens     = processed_count;
        entry.gdsf_score       = gdsf_priority(entry, gdsf_inflation);
        entry.manifest_path    = dir / fs::u8path(key + ".json");
        entry.blob_path        = dir / fs::u8path(blob_name);
        if (!write_manifest(entry)) {
            remove_blob_if_unreferenced(entry.blob_path);
            log_warning("cannot publish manifest for " + key);
            return false;
        }

        entries[key] = std::move(entry);
        ghosts.erase(key);
        if (!enforce_budget()) {
            if (!remove_entry(key)) {
                log_warning("cannot remove checkpoint after budget enforcement failed " + key);
            }
            return false;
        }
        if (entries.find(key) == entries.end()) {
            return false;
        }

        llama_memory_t memory = llama_get_memory(ctx);
        llama_memory_seq_rm(memory, hidden_seq_id, -1, -1);
        llama_memory_seq_cp(memory, source, hidden_seq_id, -1, -1);
        resident_key = key;
        return true;
    }

    bool evict_resident() {
        if (!ready || resident_key.empty()) {
            return false;
        }
        clear_anchor();
        return true;
    }
};

server_prefix_cache::server_prefix_cache() : impl_(new impl) {
}

server_prefix_cache::~server_prefix_cache() = default;

server_prefix_cache::server_prefix_cache(server_prefix_cache &&) noexcept = default;

server_prefix_cache & server_prefix_cache::operator=(server_prefix_cache &&) noexcept = default;

bool server_prefix_cache::init(
        const std::string & dir,
                     size_t chunk_tokens,
                     size_t max_bytes,
        const std::string & namespace_id,
            llama_context * ctx,
              llama_seq_id hidden_seq_id) noexcept {
    try {
        std::unique_ptr<impl> next(new impl);
        if (!next->initialize(dir, chunk_tokens, max_bytes, namespace_id, ctx, hidden_seq_id)) {
            return false;
        }
        impl_ = std::move(next);
        return true;
    } catch (const std::exception & exception) {
        log_warning(std::string("initialization failed: ") + exception.what());
    } catch (...) {
        log_warning("initialization failed with an unknown error");
    }
    return false;
}

server_prefix_cache_plan server_prefix_cache::observe(const llama_tokens & tokens, uint64_t request_id) noexcept {
    try {
        return impl_ ? impl_->observe(tokens, request_id) : server_prefix_cache_plan {};
    } catch (const std::exception & exception) {
        log_warning(std::string("observation failed: ") + exception.what());
    } catch (...) {
        log_warning("observation failed with an unknown error");
    }
    return {};
}

bool server_prefix_cache::restore(
        const llama_tokens & tokens,
              llama_seq_id dest_seq_id,
              llama_tokens & processed) noexcept {
    try {
        return impl_ && impl_->restore(tokens, dest_seq_id, processed);
    } catch (const std::exception & exception) {
        processed.clear();
        log_warning(std::string("restore failed: ") + exception.what());
    } catch (...) {
        processed.clear();
        log_warning("restore failed with an unknown error");
    }
    return false;
}

bool server_prefix_cache::capture(
        const llama_tokens & tokens,
                     size_t prefix_len,
               llama_seq_id source_seq_id) noexcept {
    try {
        return impl_ && impl_->capture(tokens, prefix_len, source_seq_id);
    } catch (const std::exception & exception) {
        log_warning(std::string("capture failed: ") + exception.what());
    } catch (...) {
        log_warning("capture failed with an unknown error");
    }
    return false;
}

bool server_prefix_cache::evict_resident() noexcept {
    try {
        return impl_ && impl_->evict_resident();
    } catch (const std::exception & exception) {
        log_warning(std::string("resident eviction failed: ") + exception.what());
    } catch (...) {
        log_warning("resident eviction failed with an unknown error");
    }
    return false;
}

bool server_prefix_cache::enabled() const noexcept {
    return impl_ && impl_->ready;
}
