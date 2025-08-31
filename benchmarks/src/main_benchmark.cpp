#include "../include/benchmark_framework.h"
#include "../include/btree_index.h"
#include "../include/learned_index_adapter.h"
#include <iostream>
#include <memory>

using namespace benchmark;

void RunComprehensiveBenchmark() {
    BenchmarkRunner runner;
    
    // Add different index implementations to compare
    runner.AddIndex(std::make_unique<LearnedIndexAdapter>());
    runner.AddIndex(std::make_unique<SortedArrayIndex>());
    runner.AddIndex(std::make_unique<BTreeIndex>());
    runner.AddIndex(std::make_unique<HashIndex>());
    
    std::cout << "=== Comprehensive Learned Index Benchmark ===" << std::endl;
    std::cout << "Comparing Learned Index vs Traditional Indexes" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Define different workload configurations
    std::vector<WorkloadConfig> workloads;
    
    // Sequential workload
    WorkloadConfig sequential_config;
    sequential_config.type = WorkloadType::Sequential;
    sequential_config.dataset_size = 10000;
    sequential_config.num_queries = 5000;
    sequential_config.key_range_min = 1000;
    sequential_config.key_range_max = 50000;
    sequential_config.seed = 42;
    workloads.push_back(sequential_config);
    
    // Random workload
    WorkloadConfig random_config;
    random_config.type = WorkloadType::Random;
    random_config.dataset_size = 10000;
    random_config.num_queries = 5000;
    random_config.key_range_min = 1000;
    random_config.key_range_max = 100000;
    random_config.seed = 123;
    workloads.push_back(random_config);
    
    // Mixed workload (80% sequential, 20% random)
    WorkloadConfig mixed_config;
    mixed_config.type = WorkloadType::Mixed;
    mixed_config.dataset_size = 10000;
    mixed_config.num_queries = 5000;
    mixed_config.key_range_min = 1000;
    mixed_config.key_range_max = 80000;
    mixed_config.sequential_ratio = 0.8;
    mixed_config.seed = 456;
    workloads.push_back(mixed_config);
    
    // Zipfian workload (skewed access pattern)
    WorkloadConfig zipfian_config;
    zipfian_config.type = WorkloadType::Zipfian;
    zipfian_config.dataset_size = 10000;
    zipfian_config.num_queries = 5000;
    zipfian_config.key_range_min = 1000;
    zipfian_config.key_range_max = 60000;
    zipfian_config.zipfian_theta = 0.99;
    zipfian_config.seed = 789;
    workloads.push_back(zipfian_config);
    
    // Temporal workload (time-series like data)
    WorkloadConfig temporal_config;
    temporal_config.type = WorkloadType::Temporal;
    temporal_config.dataset_size = 10000;
    temporal_config.num_queries = 5000;
    temporal_config.seed = 101112;
    workloads.push_back(temporal_config);
    
    // Run benchmarks for each workload
    for (const auto& workload : workloads) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        runner.RunBenchmark(workload);
    }
    
    // Print comprehensive results
    runner.PrintResults();
    
    // Save results to CSV
    runner.SaveResults("results/benchmark_results.csv");
    
    // Generate visualization charts
    runner.GenerateCharts("results");
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    std::cout << "Results saved to: results/" << std::endl;
    std::cout << "Charts generated as Python scripts in results/" << std::endl;
    std::cout << "Run the .py files to generate PNG charts." << std::endl;
}

void RunScalabilityBenchmark() {
    std::cout << "\n=== Scalability Benchmark ===" << std::endl;
    
    BenchmarkRunner runner;
    runner.AddIndex(std::make_unique<LearnedIndexAdapter>());
    runner.AddIndex(std::make_unique<SortedArrayIndex>());
    
    // Test with different dataset sizes
    std::vector<size_t> dataset_sizes = {1000, 5000, 10000, 25000, 50000};
    
    for (size_t dataset_size : dataset_sizes) {
        WorkloadConfig config;
        config.type = WorkloadType::Mixed;
        config.dataset_size = dataset_size;
        config.num_queries = dataset_size / 2; // 50% query ratio
        config.key_range_min = 1000;
        config.key_range_max = dataset_size * 10;
        config.sequential_ratio = 0.7;
        config.seed = 42;
        
        std::cout << "\nTesting dataset size: " << dataset_size << std::endl;
        runner.RunBenchmark(config);
    }
    
    runner.SaveResults("results/scalability_results.csv");
}

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [option]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --comprehensive  Run comprehensive benchmark with all workload types" << std::endl;
    std::cout << "  --scalability    Run scalability benchmark with different dataset sizes" << std::endl;
    std::cout << "  --help           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Default: Run comprehensive benchmark" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        std::string option = (argc > 1) ? argv[1] : "--comprehensive";
        
        if (option == "--help" || option == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else if (option == "--comprehensive") {
            RunComprehensiveBenchmark();
        } else if (option == "--scalability") {
            RunScalabilityBenchmark();
        } else {
            std::cerr << "Unknown option: " << option << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}