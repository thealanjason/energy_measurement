// gemm.cpp
//
// Naive dense matrix multiplication benchmark:
//   C = A * B  for N-by-N square matrices
//
// Usage:
//   g++ -O3 -std=c++17 -march=native gemm.cpp -o gemm
//   ./gemm --size=1024 --iterations=3

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t n = 1024; // matrix dimension (N x N)
    int iterations = 3;   // number of timed multiplications
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

double run_gemm(const Options& opt) {
    const std::size_t n = opt.n;
    const int iterations = opt.iterations;

    std::cout << "=== Dense GEMM (naive) ===\n";
    std::cout << "N = " << n << " (matrix is " << n << " x " << n << ")\n";
    std::cout << "iterations = " << iterations << "\n";

    std::size_t total_elems = n * n;
    std::vector<double> A(total_elems);
    std::vector<double> B(total_elems);
    std::vector<double> C(total_elems);

    // Deterministic initialization of the matrices A, B and C
    for (std::size_t i = 0; i < total_elems; ++i) {
        A[i] = static_cast<double>((i % 1000) + 1) / 1000.0;
        B[i] = static_cast<double>(((i * 7) % 1000) + 1) / 1000.0;
        C[i] = 0.0;
    }

    // >>> If needed, place energy measurement start here (PMT/Variorum/Alumet) <<<
    auto t0 = Clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        std::fill(C.begin(), C.end(), 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                double aik = A[i * n + k];
                for (std::size_t j = 0; j < n; ++j) {
                    C[i * n + j] += aik * B[k * n + j];
                }
            }
        }
    }

    auto t1 = Clock::now();
    // >>> And stop your tools after this point

    std::chrono::duration<double> elapsed = t1 - t0;
    double seconds = elapsed.count();

    // Prevent optimization: checksum over C
    double checksum = 0.0;
    for (std::size_t i = 0; i < total_elems; ++i) {
        checksum += C[i];
    }

    // FLOP count for naive GEMM: 2 * N^3 per multiply
    double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n)
                         * static_cast<double>(n) * static_cast<double>(iterations);
    double gflops = flops / seconds / 1e9;

    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "Time: " << seconds << " s\n";
    std::cout << "Approx. theoretical compute: " << gflops << " GFLOP/s\n";

    return seconds;
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    run_gemm(opt);
    return 0;
}
