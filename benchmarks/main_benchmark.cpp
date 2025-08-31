#include "learned_index/benchmark_framework.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

using namespace rocksdb::learned_index::benchmark;

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --workload <type>     Workload type (sequential, random, range, mixed)\n";
    std::cout << "  --operations <num>    Number of operations (default: 100000)\n";
    std::cout << "  --keys <num>          Number of keys (default: 1000000)\n";
    std::cout << "  --key-size <bytes>    Key size in bytes (default: 16)\n";
    std::cout << "  --value-size <bytes>  Value size in bytes (default: 100)\n";
    std::cout << "  --threads <num>       Number of threads (default: 1)\n";
    std::cout << "  --range-size <num>    Range query size (default: 100)\n";
    std::cout << "  --confidence <val>    Confidence threshold (default: 0.8)\n";
    std::cout << "  --output <file>       Output file prefix (default: benchmark)\n";
    std::cout << "  --help               Show this help message\n";
}

BenchmarkConfig ParseArguments(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "--workload" && i + 1 < argc) {
            std::string workload = argv[++i];
            if (workload == "sequential") {
                config.workload_type = WorkloadType::SEQUENTIAL_READ;
            } else if (workload == "random") {
                config.workload_type = WorkloadType::RANDOM_READ;
            } else if (workload == "range") {
                config.workload_type = WorkloadType::RANGE_QUERY;
            } else if (workload == "mixed") {
                config.workload_type = WorkloadType::MIXED_WORKLOAD;
            } else {
                std::cerr << "Unknown workload type: " << workload << std::endl;
                exit(1);
            }
        } else if (arg == "--operations" && i + 1 < argc) {
            config.num_operations = std::stoull(argv[++i]);
        } else if (arg == "--keys" && i + 1 < argc) {
            config.num_keys = std::stoull(argv[++i]);
        } else if (arg == "--key-size" && i + 1 < argc) {
            config.key_size = std::stoull(argv[++i]);
        } else if (arg == "--value-size" && i + 1 < argc) {
            config.value_size = std::stoull(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoull(argv[++i]);
        } else if (arg == "--range-size" && i + 1 < argc) {
            config.range_size = std::stoull(argv[++i]);
        } else if (arg == "--confidence" && i + 1 < argc) {
            config.learned_index_options.confidence_threshold = std::stod(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

std::string GetWorkloadName(WorkloadType type) {
    switch (type) {
        case WorkloadType::SEQUENTIAL_READ: return "Sequential Read";
        case WorkloadType::RANDOM_READ: return "Random Read";
        case WorkloadType::RANGE_QUERY: return "Range Query";
        case WorkloadType::MIXED_WORKLOAD: return "Mixed Workload";
        case WorkloadType::READ_HEAVY: return "Read Heavy";
        case WorkloadType::WRITE_HEAVY: return "Write Heavy";
        case WorkloadType::COMPACTION_HEAVY: return "Compaction Heavy";
        default: return "Unknown";
    }
}

void RunComprehensiveBenchmark() {
    std::cout << "Running Comprehensive Learned Index Benchmark Suite\n";
    std::cout << "=================================================\n\n";
    
    BenchmarkComparison comparison;
    std::vector<BenchmarkConfig> configs;
    
    // Configuration parameters
    std::vector<WorkloadType> workloads = {
        WorkloadType::SEQUENTIAL_READ,
        WorkloadType::RANDOM_READ,
        WorkloadType::RANGE_QUERY,
        WorkloadType::MIXED_WORKLOAD
    };
    
    std::vector<size_t> dataset_sizes = {10000, 100000, 1000000};
    std::vector<size_t> operation_counts = {10000, 50000, 100000};
    
    // Create configurations for comprehensive testing
    for (auto workload : workloads) {
        for (auto dataset_size : dataset_sizes) {
            for (auto op_count : operation_counts) {
                // Traditional configuration
                BenchmarkConfig traditional_config;
                traditional_config.workload_type = workload;
                traditional_config.num_keys = dataset_size;
                traditional_config.num_operations = op_count;
                traditional_config.enable_learned_index = false;
                traditional_config.output_file = "traditional";
                configs.push_back(traditional_config);
                
                // Learned index configuration
                BenchmarkConfig learned_config = traditional_config;
                learned_config.enable_learned_index = true;
                learned_config.learned_index_options.model_type = ModelType::LINEAR;
                learned_config.learned_index_options.confidence_threshold = 0.8;
                learned_config.learned_index_options.max_prediction_error_bytes = 4096;
                learned_config.output_file = "learned";
                configs.push_back(learned_config);
            }
        }
    }
    
    std::cout << "Total configurations to test: " << configs.size() << "\n\n";
    
    // Run benchmarks
    size_t completed = 0;
    for (const auto& config : configs) {
        std::string config_name = (config.enable_learned_index ? "LearnedIndex_" : "Traditional_") +
                                 GetWorkloadName(config.workload_type) + "_" +
                                 std::to_string(config.num_keys) + "keys_" +
                                 std::to_string(config.num_operations) + "ops";
        
        std::cout << "Running: " << config_name << " (" << ++completed << "/" << configs.size() << ")\n";
        
        BenchmarkRunner runner(config);
        if (runner.SetupBenchmark()) {
            auto start_time = std::chrono::high_resolution_clock::now();
            runner.RunBenchmark();
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            auto metrics = runner.AnalyzeResults();
            comparison.AddResult(config_name, metrics);
            
            std::cout << "  Completed in " << duration << "ms - "
                     << "Avg Latency: " << std::fixed << std::setprecision(2)
                     << metrics.avg_latency_ns / 1000.0 << "μs, "
                     << "Throughput: " << std::setprecision(0)
                     << metrics.operations_per_second << " ops/sec\n";
            
            runner.CleanupBenchmark();
        } else {
            std::cerr << "  Failed to setup benchmark for " << config_name << "\n";
        }
    }
    
    std::cout << "\nGenerating comprehensive performance reports...\n";
    
    // Generate all report formats
    comparison.GenerateTextReport("benchmarks/results/comprehensive_report.txt");
    comparison.GenerateCSVReport("benchmarks/results/comprehensive_report.csv");
    comparison.GenerateHTMLReport("benchmarks/results/comprehensive_report.html");
    comparison.GenerateJSONReport("benchmarks/results/comprehensive_report.json");
    
    std::cout << "\nBenchmark Summary:\n";
    comparison.PrintSummary();
    
    std::cout << "\nReports generated:\n";
    std::cout << "  - benchmarks/results/comprehensive_report.txt\n";
    std::cout << "  - benchmarks/results/comprehensive_report.csv\n";
    std::cout << "  - benchmarks/results/comprehensive_report.html\n";
    std::cout << "  - benchmarks/results/comprehensive_report.json\n";
}

void RunSingleBenchmark(const BenchmarkConfig& config) {
    std::cout << "Running Single Benchmark\n";
    std::cout << "========================\n";
    std::cout << "Workload: " << GetWorkloadName(config.workload_type) << "\n";
    std::cout << "Operations: " << config.num_operations << "\n";
    std::cout << "Keys: " << config.num_keys << "\n";
    std::cout << "Learned Index: " << (config.enable_learned_index ? "Enabled" : "Disabled") << "\n\n";
    
    BenchmarkComparison comparison;
    
    // Run traditional benchmark
    BenchmarkConfig traditional_config = config;
    traditional_config.enable_learned_index = false;
    
    std::cout << "Running traditional implementation...\n";
    BenchmarkRunner traditional_runner(traditional_config);
    if (traditional_runner.SetupBenchmark()) {
        traditional_runner.RunBenchmark();
        auto traditional_metrics = traditional_runner.AnalyzeResults();
        comparison.AddResult("Traditional", traditional_metrics);
        traditional_runner.CleanupBenchmark();
        std::cout << "Traditional benchmark completed.\n";
    }
    
    // Run learned index benchmark
    BenchmarkConfig learned_config = config;
    learned_config.enable_learned_index = true;
    
    std::cout << "Running learned index implementation...\n";
    BenchmarkRunner learned_runner(learned_config);
    if (learned_runner.SetupBenchmark()) {
        learned_runner.RunBenchmark();
        auto learned_metrics = learned_runner.AnalyzeResults();
        comparison.AddResult("LearnedIndex", learned_metrics);
        learned_runner.CleanupBenchmark();
        std::cout << "Learned index benchmark completed.\n";
    }
    
    // Generate reports
    std::string output_prefix = config.output_file.empty() ? "benchmark" : config.output_file;
    
    comparison.GenerateTextReport("benchmarks/results/" + output_prefix + "_report.txt");
    comparison.GenerateCSVReport("benchmarks/results/" + output_prefix + "_report.csv");
    comparison.GenerateHTMLReport("benchmarks/results/" + output_prefix + "_report.html");
    comparison.GenerateJSONReport("benchmarks/results/" + output_prefix + "_report.json");
    
    std::cout << "\nBenchmark Results:\n";
    comparison.PrintSummary();
}

int main(int argc, char* argv[]) {
    std::cout << "Learned Index RocksDB - Performance Benchmark Suite\n";
    std::cout << "===================================================\n\n";
    
    // Parse command line arguments
    BenchmarkConfig config = ParseArguments(argc, argv);
    
    // Create output directories
    system("mkdir -p benchmarks/results benchmarks/reports");
    
    if (argc == 1) {
        // No arguments provided - run comprehensive benchmark
        RunComprehensiveBenchmark();
    } else {
        // Arguments provided - run single benchmark with specified config
        RunSingleBenchmark(config);
    }
    
    return 0;
}