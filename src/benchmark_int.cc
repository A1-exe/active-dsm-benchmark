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
    void benchmark_compression(hshm::Compressor &comp, hshm::Distribution &dist, size_t raw_size) {
        // Pre-allocate memory for the dataset
        std::vector<int> raw(raw_size);
        std::vector<int> compressed(raw_size);
        std::vector<int> decompressed(raw_size);
        size_t raw_size_bytes = raw_size * sizeof(int);
        size_t cmpr_size_bytes = raw_size_bytes;

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
            raw[i] = dist.GetInt();
        }
        
        // Reset timers
        compress_timer_.Reset();
        decompress_timer_.Reset();

        // Time compression
        compress_timer_.Resume();
        comp.Compress(compressed.data(), cmpr_size_bytes, raw.data(), raw_size_bytes);
        compress_timer_.Pause();

        HIPRINT("Compression took {} msec\n", compress_timer_.GetMsecFromStart());

        // Time Decompression
        decompress_timer_.Resume();
        comp.Decompress(decompressed.data(), raw_size_bytes, compressed.data(), cmpr_size_bytes);
        decompress_timer_.Pause();

        HIPRINT("Decompression took {} msec\n", decompress_timer_.GetMsecFromStart());

        // Ensure data integrity
        printf("Raw bytes: %zu\n", raw_size_bytes);
        printf("Compressed bytes: %zu\n", cmpr_size_bytes);
        assert(raw == decompressed);
    }
};

int main(int argc, char** argv) {
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

    Benchmark* bench;
    hshm::Compressor* bench_comp;
    hshm::Distribution* bench_dist;

    // Usage:
    // ./benchmark_int count algo dist size [seed] [shape]
    // count: number of iterations
    // algo: compression algorithm
    // dist: distribution
    // size: size of the dataset
    // seed: random seed
    // shape: shape parameter for the distribution

    // Parse command line arguments
    if (argc < 5) {
        HIPRINT("Usage: ./benchmark_int count algo dist size [seed] [shape]\n");
        return 1;
    }

    int count = std::stoi(argv[1]);
    std::string algo = argv[2];
    std::string dist = argv[3];
    size_t size = std::stoi(argv[4]);
    double seed = -1;
    double shape = -1;

    if (argc > 5) {
        seed = std::stod(argv[5]);
    }

    if (argc > 6) {
        shape = std::stod(argv[6]);
    }

    // Create compressor from arguments
    if (algo == "bzip2") {
        bench_comp = new hshm::Bzip2();
    } else if (algo == "lzo") {
        bench_comp = new hshm::Lzo();
    } else if (algo == "zstd") {
        bench_comp = new hshm::Zstd();
    } else if (algo == "lz4") {
        bench_comp = new hshm::Lz4();
    } else if (algo == "zlib") {
        bench_comp = new hshm::Zlib();
    } else if (algo == "lzma") {
        bench_comp = new hshm::Lzma();
    } else if (algo == "brotli") {
        bench_comp = new hshm::Brotli();
    } else if (algo == "snappy") {
        bench_comp = new hshm::Snappy();
    } else if (algo == "blosc2") {
        bench_comp = new hshm::Blosc();
    } else {
        HIPRINT("Invalid compression algorithm\n");
        return 1;
    }

    // Create distribution from arguments
    if (dist == "uniform") {
        bench_dist = new hshm::UniformDistribution();
    } else if (dist == "normal") {
        bench_dist = new hshm::NormalDistribution();
    } else if (dist == "gamma") {
        bench_dist = new hshm::GammaDistribution();
    } else if (dist == "exponential") {
        bench_dist = new hshm::ExponentialDistribution();
    } else {
        HIPRINT("Invalid distribution\n");
        return 1;
    }
    
    if (seed != -1) {
        if (shape != -1) {
            bench = new Benchmark(seed, shape);
        } else {
            bench = new Benchmark(seed);
        }
    } else {
        bench = new Benchmark();
    }

    for (int i = 0; i < count; ++i) {
        bench->benchmark_compression(*bench_comp, *bench_dist, size);
    }
}