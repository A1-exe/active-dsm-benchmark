#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/timer.h"
#include "hermes_shm/util/compress/compress_factory.h"
#include "hermes_shm/util/random.h"
#include <utility>
#include <cassert>
#include <iterator>

template <typename T>
void display_vector(std::vector<T> &v)
{
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<T>(std::cout, " "));
}

class Benchmark {
    public:
    hshm::Timer compress_timer_;
    hshm::Timer decompress_timer_;
    size_t seed = 0;
    size_t shape = 0;

    Benchmark() {}

    Benchmark(double seed) {
        this->seed = seed;
    }

    Benchmark(double seed, double shape) {
        this->seed = seed;
        this->shape = shape;
    }

    /// Benchmarking The Impact of Compression Algorithms on Memory Performance
    /// @param comp The compression algorithm to use
    /// @param dist The distribution of the dataset
    /// @param cmpr_size The compression size
    /// @param raw_size The size of the dataset (Remember doubles are 8 bytes)
    /// @returns The time taken to compress and decompress a dataset
    void benchmark_compression(hshm::Compressor &comp, hshm::Distribution &dist, size_t cmpr_size, size_t raw_size) {
        // Pre-allocate memory for the dataset
        std::vector<char> raw(raw_size);
        std::vector<char> compressed(raw_size);
        std::vector<char> decompressed(raw_size);

        if (seed != 0) {
            dist.Seed(seed);
        } else {
            dist.Seed();
        }

        // if (shape != 0) {
        //     dist.Shape(shape);
        // }

        // Generate random data
        for (size_t i = 0; i < raw_size; ++i) {
            raw[i] = (char)dist.GetInt();
        }
        
        // Reset timers
        compress_timer_.Reset();
        decompress_timer_.Reset();

        // Time compression
        compress_timer_.Resume();
        comp.Compress(compressed.data(), cmpr_size, raw.data(), raw.size());
        compress_timer_.Pause();

        HIPRINT("Compression took {} msec\n", compress_timer_.GetMsecFromStart());

        // Time Decompression
        decompress_timer_.Resume();
        comp.Decompress(decompressed.data(), raw_size, compressed.data(), cmpr_size);
        decompress_timer_.Pause();

        HIPRINT("Decompression took {} msec\n", decompress_timer_.GetMsecFromStart());

        // Ensure data integrity
        assert(raw == decompressed);
    }
};

int main() {
    /* 
        == Hermes Supported Compression Algorithms ==
        - Bzip2, LZO, Zstd, LZ4, Zlib, Lzma
        - Brotli, Snappy, Blosc2

        == Hermes Supported Random Distributions ==
        - Uniform Random, Normal, Gamma, Exponential
    */

    /* 
        == Objective(s) ==
        - Benchmark each library for a variety of datasets
        -- Data Type: int, size_t, double
        -- Data Distribution: Uniform Random, Normal, Gamma, Exponential
    */

    Benchmark bench;

    // Distributions
    hshm::UniformDistribution uniform;
    hshm::NormalDistribution normal;
    hshm::GammaDistribution gamma;
    hshm::ExponentialDistribution exponential;

    // Compression Algorithms
    hshm::Bzip2 bzip2;
    hshm::Lzo lzo;
    hshm::Zstd zstd;
    hshm::Lz4 lz4;
    hshm::Zlib zlib;
    hshm::Lzma lzma;
    hshm::Brotli brotli;
    hshm::Snappy snappy;
    hshm::Blosc blosc2;
    
    // Tests
    HIPRINT("=== TESTING BZIP2 ===\n");
    bench.benchmark_compression(bzip2, uniform, 1024, 1024);
    HIPRINT("\n");

    // vSegfaults
    // HIPRINT("=== TESTING LZO ===\n");
    // bench.benchmark_compression(lzo, uniform, 1024, 1024);
    // HIPRINT("\n");

    HIPRINT("=== TESTING ZSTD ===\n");
    bench.benchmark_compression(zstd, uniform, 1024, 1024);
    HIPRINT("\n");

    HIPRINT("=== TESTING LZ4 ===\n");
    bench.benchmark_compression(lz4, uniform, 1024, 1024);
    HIPRINT("\n");

    HIPRINT("=== TESTING ZLIB ===\n");
    bench.benchmark_compression(zlib, uniform, 1024, 1024);
    HIPRINT("\n");

    HIPRINT("=== TESTING LZMA ===\n");
    bench.benchmark_compression(lzma, uniform, 1024, 1024);
    HIPRINT("\n");

    HIPRINT("=== TESTING BROTLI ===\n");
    bench.benchmark_compression(brotli, uniform, 1024, 1024);
    HIPRINT("\n");

    HIPRINT("=== TESTING SNAPPY ===\n");
    bench.benchmark_compression(snappy, uniform, 1024, 1024);
    HIPRINT("\n");

    HIPRINT("=== TESTING BLOSC2 ===\n");
    bench.benchmark_compression(blosc2, uniform, 1024, 1024);
    HIPRINT("\n");

    

    HIPRINT("Compress/Decompress test passed\n");
}