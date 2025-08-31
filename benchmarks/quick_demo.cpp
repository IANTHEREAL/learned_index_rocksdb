#include "learned_index/benchmark_framework.h"
#include "workloads/ycsb_workloads.h"
#include <iostream>
#include <iomanip>

using namespace rocksdb::learned_index::benchmark;

void RunQuickDemo() {
    std::cout << "Learned Index RocksDB - Quick Performance Demo\n";
    std::cout << "=============================================\n\n";
    
    BenchmarkComparison comparison;
    
    // Test configuration - smaller scale for quick demo
    std::vector<std::pair<std::string, BenchmarkConfig>> test_configs = {
        {"Sequential Read", [](){
            BenchmarkConfig config;
            config.workload_type = WorkloadType::SEQUENTIAL_READ;
            config.num_keys = 50000;
            config.num_operations = 25000;
            config.key_size = 16;
            config.value_size = 100;
            return config;
        }()},
        {"Random Read", [](){
            BenchmarkConfig config;
            config.workload_type = WorkloadType::RANDOM_READ;
            config.num_keys = 50000;
            config.num_operations = 25000;
            config.key_size = 16;
            config.value_size = 100;
            return config;
        }()},
        {"Range Query", [](){
            BenchmarkConfig config;
            config.workload_type = WorkloadType::RANGE_QUERY;
            config.num_keys = 50000;
            config.num_operations = 5000;
            config.range_size = 50;
            config.key_size = 16;
            config.value_size = 100;
            return config;
        }()}
    };
    
    for (const auto& test_config : test_configs) {
        std::cout << "Testing " << test_config.first << " workload...\n";
        std::cout << "Keys: " << test_config.second.num_keys 
                  << ", Operations: " << test_config.second.num_operations << "\n";
        
        // Run traditional version
        BenchmarkConfig traditional_config = test_config.second;
        traditional_config.enable_learned_index = false;
        
        std::cout << "  Running traditional implementation...";
        std::cout.flush();
        
        BenchmarkRunner traditional_runner(traditional_config);
        if (traditional_runner.SetupBenchmark()) {
            traditional_runner.RunBenchmark();
            auto traditional_metrics = traditional_runner.AnalyzeResults();
            comparison.AddResult("Traditional_" + test_config.first, traditional_metrics);
            traditional_runner.CleanupBenchmark();
        }
        std::cout << " ✓\n";
        
        // Run learned index version
        BenchmarkConfig learned_config = test_config.second;
        learned_config.enable_learned_index = true;
        learned_config.learned_index_options.model_type = ModelType::LINEAR;
        learned_config.learned_index_options.confidence_threshold = 0.8;
        learned_config.learned_index_options.max_prediction_error_bytes = 4096;
        
        std::cout << "  Running learned index implementation...";
        std::cout.flush();
        
        BenchmarkRunner learned_runner(learned_config);
        if (learned_runner.SetupBenchmark()) {
            learned_runner.RunBenchmark();
            auto learned_metrics = learned_runner.AnalyzeResults();
            comparison.AddResult("LearnedIndex_" + test_config.first, learned_metrics);
            learned_runner.CleanupBenchmark();
        }
        std::cout << " ✓\n\n";
    }
    
    std::cout << "Demo Results Summary:\n";
    std::cout << "====================\n";
    comparison.PrintSummary();
    
    // Generate quick demo report
    comparison.GenerateTextReport("benchmarks/results/quick_demo_report.txt");
    comparison.GenerateHTMLReport("benchmarks/results/quick_demo_report.html");
    
    std::cout << "\nDetailed reports saved to:\n";
    std::cout << "  - benchmarks/results/quick_demo_report.txt\n";
    std::cout << "  - benchmarks/results/quick_demo_report.html\n";
}

void DemonstrateWorkloadGenerators() {
    std::cout << "\nWorkload Generator Demonstration\n";
    std::cout << "===============================\n";
    
    BenchmarkConfig config;
    config.num_operations = 1000;
    config.num_keys = 10000;
    config.range_size = 10;
    
    std::vector<std::unique_ptr<WorkloadGenerator>> generators;
    generators.push_back(std::make_unique<SequentialWorkloadGenerator>());
    generators.push_back(std::make_unique<RandomWorkloadGenerator>(42));
    generators.push_back(std::make_unique<ZipfianWorkloadGenerator>(1.0, 42));
    
    for (const auto& generator : generators) {
        std::cout << "\n" << generator->GetName() << " Generator:\n";
        std::cout << "Description: " << generator->GetDescription() << "\n";
        
        auto keys = generator->GenerateKeys(config);
        
        // Show first 20 keys as sample
        std::cout << "Sample keys: ";
        for (size_t i = 0; i < std::min(size_t(20), keys.size()); ++i) {
            std::cout << keys[i];
            if (i < std::min(size_t(19), keys.size() - 1)) std::cout << ", ";
        }
        std::cout << "\n";
        
        // Calculate basic statistics
        if (!keys.empty()) {
            auto minmax = std::minmax_element(keys.begin(), keys.end());
            double avg = std::accumulate(keys.begin(), keys.end(), 0.0) / keys.size();
            
            std::cout << "Statistics: Min=" << *minmax.first 
                      << ", Max=" << *minmax.second
                      << ", Avg=" << std::fixed << std::setprecision(2) << avg << "\n";
        }
    }
}

void ShowPerformanceMetricsExample() {
    std::cout << "\nPerformance Metrics Example\n";
    std::cout << "==========================\n";
    
    // Create a simple benchmark to demonstrate metrics
    BenchmarkConfig config;
    config.workload_type = WorkloadType::SEQUENTIAL_READ;
    config.num_keys = 10000;
    config.num_operations = 5000;
    config.enable_learned_index = true;
    config.learned_index_options.confidence_threshold = 0.8;
    
    BenchmarkRunner runner(config);
    if (runner.SetupBenchmark()) {
        std::cout << "Running example benchmark...\n";
        runner.RunBenchmark();
        
        auto metrics = runner.AnalyzeResults();
        
        std::cout << "\nDetailed Performance Metrics:\n";
        std::cout << "----------------------------\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total Operations: " << metrics.total_operations << "\n";
        std::cout << "Successful Operations: " << metrics.successful_operations << "\n";
        std::cout << "Average Latency: " << metrics.avg_latency_ns / 1000.0 << " μs\n";
        std::cout << "Median (P50) Latency: " << metrics.p50_latency_ns / 1000.0 << " μs\n";
        std::cout << "95th Percentile Latency: " << metrics.p95_latency_ns / 1000.0 << " μs\n";
        std::cout << "99th Percentile Latency: " << metrics.p99_latency_ns / 1000.0 << " μs\n";
        std::cout << "Max Latency: " << metrics.max_latency_ns / 1000.0 << " μs\n";
        std::cout << "Min Latency: " << metrics.min_latency_ns / 1000.0 << " μs\n";
        std::cout << std::setprecision(0);
        std::cout << "Throughput: " << metrics.operations_per_second << " ops/sec\n";
        std::cout << std::setprecision(2);
        std::cout << "Bandwidth: " << metrics.mb_per_second << " MB/sec\n";
        std::cout << "Memory Usage: " << metrics.memory_usage_bytes / 1024 << " KB\n";
        std::cout << std::setprecision(1);
        std::cout << "Prediction Accuracy: " << metrics.prediction_accuracy * 100.0 << "%\n";
        std::cout << "Cache Hit Rate: " << metrics.cache_hit_rate * 100.0 << "%\n";
        std::cout << "Fallback Rate: " << metrics.fallback_rate * 100.0 << "%\n";
        
        runner.CleanupBenchmark();
    }
}

int main() {
    system("mkdir -p benchmarks/results");
    
    std::cout << "Learned Index RocksDB - Benchmark Framework Demo\n";
    std::cout << "================================================\n";
    
    try {
        // Run quick performance demonstration
        RunQuickDemo();
        
        // Demonstrate different workload generators
        DemonstrateWorkloadGenerators();
        
        // Show detailed performance metrics
        ShowPerformanceMetricsExample();
        
        std::cout << "\n🎉 Benchmark demonstration completed successfully!\n";
        std::cout << "\nNext Steps:\n";
        std::cout << "1. Run full benchmark suite: ./benchmarks/run_performance_analysis.sh\n";
        std::cout << "2. View detailed HTML report: benchmarks/results/quick_demo_report.html\n";
        std::cout << "3. Customize benchmarks using the main benchmark executable\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmark demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}