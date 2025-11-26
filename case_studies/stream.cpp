// Simple STREAM Triad benchmark:
//   a[i] = b[i] + s * c[i]
//
// Usage:
//   g++ -O3 -std=c++17 -march=native stream.cpp -o stream
//   ./stream --size=50000000 --iterations=5

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t n = 50'000'000; // number of elements in a vector array 
    int iterations = 5;         // number of timed repetitions
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--size=N] [--iterations=K]\n";
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq_pos = arg.find('=');
        std::string key = (eq_pos == std::string::npos) ? arg : arg.substr(0, eq_pos);
        std::string val = (eq_pos == std::string::npos) ? ""  : arg.substr(eq_pos + 1);

        if (key == "--size") {
            opt.n = static_cast<std::size_t>(std::stoull(val));
        } else if (key == "--iterations") {
            opt.iterations = std::stoi(val);
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return opt;
}

double run_stream_triad(const Options& opt) {
    const std::size_t n = opt.n;
    const int iterations = opt.iterations;
    const double scalar = 3.0;

    std::cout << "=== STREAM Triad ===\n";
    std::cout << "N = " << n << ", iterations = " << iterations << "\n";

    std::vector<double> a(n), b(n), c(n);

    // Initialize data
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 2.0;
    }

    // Warm-up (unmeasured)
    {
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = b[i] + scalar * c[i];
        }
    }

    // >>> If needed, place energy measurement start here (PMT/Variorum/Alumet) <<<
    auto t0 = Clock::now();

    for (int it = 0; it < iterations; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = b[i] + scalar * c[i];
        }
    }

    auto t1 = Clock::now();
    // >>> Place energy measurement stop here (PMT/Variorum/Alumet) <<<

    std::chrono::duration<double> elapsed = t1 - t0;
    double seconds = elapsed.count();

    // Prevent the compiler from optimizing away the loop
    double checksum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        checksum += a[i];
    }

    // STREAM triad: per element: 2 loads + 1 store of 8 bytes = 24 bytes
    // and 2 FLOPs (mul + add).
    double bytes_moved = static_cast<double>(n) * 24.0 * iterations;
    double flops       = static_cast<double>(n) * 2.0  * iterations;

    double bandwidth_gb_s = bytes_moved / seconds / 1e9;
    double gflops         = flops       / seconds / 1e9;

    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "Time: " << seconds << " s\n";
    std::cout << "Approx. theoretical bandwidth: " << bandwidth_gb_s << " GB/s\n";
    std::cout << "Approx. theoretical compute:   " << gflops         << " GFLOP/s\n";

    return seconds;
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    run_stream_triad(opt);
    return 0;
}
