#include "learned_index/benchmark_framework.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

namespace rocksdb {
namespace learned_index {
namespace benchmark {

// MemoryTracker implementation
MemoryTracker::MemoryTracker() : baseline_memory_(0), peak_memory_(0) {}

void MemoryTracker::RecordBaseline() {
    baseline_memory_ = GetCurrentUsage();
    peak_memory_ = baseline_memory_;
}

void MemoryTracker::UpdatePeakUsage() {
    size_t current = GetCurrentUsage();
    if (current > peak_memory_) {
        peak_memory_ = current;
    }
}

size_t MemoryTracker::GetCurrentUsage() const {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024; // Convert KB to bytes on Linux
}

size_t MemoryTracker::GetPeakUsage() const {
    return peak_memory_;
}

size_t MemoryTracker::GetAdditionalUsage() const {
    return peak_memory_ > baseline_memory_ ? peak_memory_ - baseline_memory_ : 0;
}

// SequentialWorkloadGenerator implementation
std::vector<uint64_t> SequentialWorkloadGenerator::GenerateKeys(const BenchmarkConfig& config) {
    std::vector<uint64_t> keys;
    keys.reserve(config.num_operations);
    
    for (size_t i = 0; i < config.num_operations; ++i) {
        keys.push_back(i % config.num_keys);
    }
    
    return keys;
}

// RandomWorkloadGenerator implementation
std::vector<uint64_t> RandomWorkloadGenerator::GenerateKeys(const BenchmarkConfig& config) {
    std::vector<uint64_t> keys;
    keys.reserve(config.num_operations);
    
    std::mt19937_64 gen(seed_);
    std::uniform_int_distribution<uint64_t> dist(0, config.num_keys - 1);
    
    for (size_t i = 0; i < config.num_operations; ++i) {
        keys.push_back(dist(gen));
    }
    
    return keys;
}

// ZipfianWorkloadGenerator implementation
std::vector<uint64_t> ZipfianWorkloadGenerator::GenerateKeys(const BenchmarkConfig& config) {
    std::vector<uint64_t> keys;
    keys.reserve(config.num_operations);
    
    // Pre-compute Zipfian probabilities
    std::vector<double> probabilities(config.num_keys);
    double sum = 0.0;
    
    for (size_t i = 0; i < config.num_keys; ++i) {
        probabilities[i] = 1.0 / std::pow(i + 1, alpha_);
        sum += probabilities[i];
    }
    
    // Normalize probabilities
    for (double& prob : probabilities) {
        prob /= sum;
    }
    
    // Create cumulative distribution
    std::vector<double> cumulative(config.num_keys);
    std::partial_sum(probabilities.begin(), probabilities.end(), cumulative.begin());
    
    // Generate keys
    std::mt19937_64 gen(seed_);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < config.num_operations; ++i) {
        double rand_val = dist(gen);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rand_val);
        size_t key_index = std::distance(cumulative.begin(), it);
        keys.push_back(key_index);
    }
    
    return keys;
}

// MockSST implementation
MockSST::MockSST(size_t block_size) 
    : block_size_(block_size), learned_index_enabled_(false) {}

void MockSST::AddKey(uint64_t key, const std::vector<uint8_t>& value) {
    data_[key] = {0, value}; // Block index will be set in Finalize()
}

void MockSST::AddKeys(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>>& keys) {
    for (const auto& kv : keys) {
        AddKey(kv.first, kv.second);
    }
}

void MockSST::Finalize() {
    OrganizeIntoBlocks();
    if (learned_index_enabled_) {
        TrainLearnedIndex();
    }
}

OperationResult MockSST::Get(uint64_t key) {
    OperationResult result;
    result.start_time = std::chrono::high_resolution_clock::now();
    
    auto it = data_.find(key);
    if (it == data_.end()) {
        result.success = false;
        result.end_time = std::chrono::high_resolution_clock::now();
        return result;
    }
    
    uint32_t actual_block = it->second.first;
    uint32_t predicted_block = actual_block;
    bool used_learned_index = false;
    bool cache_hit = false;
    
    if (learned_index_enabled_ && learned_index_manager_) {
        double confidence;
        predicted_block = learned_index_manager_->PredictBlock(key, &confidence);
        used_learned_index = (confidence >= learned_index_manager_->GetStats().total_queries > 0 ? 
                             learned_index_manager_->GetStats().GetSuccessRate() : 0.8);
        
        // Simulate cache hit based on manager's cache
        cache_hit = (learned_index_manager_->GetCacheSize() > 0);
        
        // Update manager statistics
        learned_index_manager_->UpdateStats(key, actual_block, predicted_block, !used_learned_index);
    }
    
    result.success = true;
    result.used_learned_index = used_learned_index;
    result.cache_hit = cache_hit;
    result.predicted_block = predicted_block;
    result.actual_block = actual_block;
    result.bytes_read = it->second.second.size();
    result.end_time = std::chrono::high_resolution_clock::now();
    
    return result;
}

std::vector<OperationResult> MockSST::RangeQuery(uint64_t start_key, uint64_t end_key) {
    std::vector<OperationResult> results;
    
    for (uint64_t key = start_key; key <= end_key; ++key) {
        if (data_.find(key) != data_.end()) {
            results.push_back(Get(key));
        }
    }
    
    return results;
}

void MockSST::EnableLearnedIndex(const SSTLearnedIndexOptions& options) {
    learned_index_enabled_ = true;
    learned_index_manager_ = std::make_unique<SSTLearnedIndexManager>(options);
    learned_index_manager_->Initialize("benchmark.sst", GetDataSize());
}

void MockSST::DisableLearnedIndex() {
    learned_index_enabled_ = false;
    learned_index_manager_.reset();
}

size_t MockSST::GetDataSize() const {
    size_t total_size = 0;
    for (const auto& kv : data_) {
        total_size += sizeof(uint64_t) + kv.second.second.size();
    }
    return total_size;
}

uint32_t MockSST::FindBlockTraditional(uint64_t key) const {
    // Simple linear search simulation
    for (size_t block_idx = 0; block_idx < blocks_.size(); ++block_idx) {
        const auto& block = blocks_[block_idx];
        if (!block.empty() && key >= block.front() && key <= block.back()) {
            return static_cast<uint32_t>(block_idx);
        }
    }
    return 0;
}

void MockSST::OrganizeIntoBlocks() {
    // Sort keys for block organization
    std::vector<uint64_t> sorted_keys;
    sorted_keys.reserve(data_.size());
    
    for (const auto& kv : data_) {
        sorted_keys.push_back(kv.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());
    
    // Organize into blocks
    blocks_.clear();
    size_t keys_per_block = std::max(size_t(1), 
        static_cast<size_t>(block_size_ / (sizeof(uint64_t) + 100))); // Assume 100 byte average value
    
    for (size_t i = 0; i < sorted_keys.size(); i += keys_per_block) {
        std::vector<uint64_t> block;
        size_t end = std::min(i + keys_per_block, sorted_keys.size());
        
        for (size_t j = i; j < end; ++j) {
            block.push_back(sorted_keys[j]);
            data_[sorted_keys[j]].first = static_cast<uint32_t>(blocks_.size());
        }
        
        blocks_.push_back(block);
    }
}

void MockSST::TrainLearnedIndex() {
    if (!learned_index_manager_) {
        return;
    }
    
    // Create key ranges for training
    std::vector<KeyRange> key_ranges;
    key_ranges.reserve(blocks_.size());
    
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& block = blocks_[i];
        if (!block.empty()) {
            key_ranges.emplace_back(
                block.front(),                    // start_key
                block.back(),                     // end_key
                static_cast<uint32_t>(i),        // block_index
                block.size()                      // key_count
            );
        }
    }
    
    learned_index_manager_->TrainModel(key_ranges);
}

// BenchmarkRunner implementation
BenchmarkRunner::BenchmarkRunner(const BenchmarkConfig& config) : config_(config) {
    // Create appropriate workload generator
    switch (config_.workload_type) {
        case WorkloadType::SEQUENTIAL_READ:
            workload_generator_ = std::make_unique<SequentialWorkloadGenerator>();
            break;
        case WorkloadType::RANDOM_READ:
            workload_generator_ = std::make_unique<RandomWorkloadGenerator>();
            break;
        case WorkloadType::RANGE_QUERY:
        case WorkloadType::MIXED_WORKLOAD:
        case WorkloadType::READ_HEAVY:
        case WorkloadType::WRITE_HEAVY:
        case WorkloadType::COMPACTION_HEAVY:
        default:
            workload_generator_ = std::make_unique<RandomWorkloadGenerator>();
            break;
    }
    
    sst_ = std::make_unique<MockSST>();
}

bool BenchmarkRunner::SetupBenchmark() {
    memory_tracker_.RecordBaseline();
    
    // Generate test data
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> test_data;
    test_data.reserve(config_.num_keys);
    
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<uint8_t> value_dist(0, 255);
    
    for (size_t i = 0; i < config_.num_keys; ++i) {
        std::vector<uint8_t> value(config_.value_size);
        for (size_t j = 0; j < config_.value_size; ++j) {
            value[j] = value_dist(gen);
        }
        test_data.emplace_back(i, std::move(value));
    }
    
    // Add data to SST
    sst_->AddKeys(test_data);
    
    // Configure learned index if enabled
    if (config_.enable_learned_index) {
        sst_->EnableLearnedIndex(config_.learned_index_options);
    }
    
    // Finalize SST (organize blocks and train model)
    sst_->Finalize();
    
    memory_tracker_.UpdatePeakUsage();
    return true;
}

void BenchmarkRunner::CleanupBenchmark() {
    results_.clear();
    sst_.reset();
}

void BenchmarkRunner::RunBenchmark() {
    results_.clear();
    results_.reserve(config_.num_operations);
    
    switch (config_.workload_type) {
        case WorkloadType::SEQUENTIAL_READ:
            RunSequentialReads();
            break;
        case WorkloadType::RANDOM_READ:
        case WorkloadType::READ_HEAVY:
            RunRandomReads();
            break;
        case WorkloadType::RANGE_QUERY:
            RunRangeQueries();
            break;
        case WorkloadType::MIXED_WORKLOAD:
        case WorkloadType::WRITE_HEAVY:
        case WorkloadType::COMPACTION_HEAVY:
            RunMixedWorkload();
            break;
    }
    
    memory_tracker_.UpdatePeakUsage();
}

void BenchmarkRunner::RunSequentialReads() {
    auto keys = workload_generator_->GenerateKeys(config_);
    
    for (uint64_t key : keys) {
        auto result = sst_->Get(key);
        results_.push_back(result);
    }
}

void BenchmarkRunner::RunRandomReads() {
    auto keys = workload_generator_->GenerateKeys(config_);
    
    for (uint64_t key : keys) {
        auto result = sst_->Get(key);
        results_.push_back(result);
    }
}

void BenchmarkRunner::RunRangeQueries() {
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<uint64_t> start_dist(0, config_.num_keys - config_.range_size);
    
    for (size_t i = 0; i < config_.num_operations; ++i) {
        uint64_t start_key = start_dist(gen);
        uint64_t end_key = start_key + config_.range_size;
        
        auto range_results = sst_->RangeQuery(start_key, end_key);
        results_.insert(results_.end(), range_results.begin(), range_results.end());
    }
}

void BenchmarkRunner::RunMixedWorkload() {
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    auto keys = workload_generator_->GenerateKeys(config_);
    
    for (size_t i = 0; i < config_.num_operations; ++i) {
        double op_type = op_dist(gen);
        
        if (op_type < config_.read_ratio) {
            // Read operation
            uint64_t key = keys[i % keys.size()];
            auto result = sst_->Get(key);
            results_.push_back(result);
        } else {
            // Simulate write operation (just record timing)
            OperationResult result;
            result.start_time = std::chrono::high_resolution_clock::now();
            // Simulate write latency
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            result.end_time = std::chrono::high_resolution_clock::now();
            result.success = true;
            result.used_learned_index = false;
            results_.push_back(result);
        }
    }
}

PerformanceMetrics BenchmarkRunner::AnalyzeResults() const {
    PerformanceMetrics metrics;
    
    if (results_.empty()) {
        return metrics;
    }
    
    // Collect latencies
    std::vector<double> latencies;
    latencies.reserve(results_.size());
    
    uint64_t successful_ops = 0;
    uint64_t failed_ops = 0;
    uint64_t learned_index_ops = 0;
    uint64_t cache_hits = 0;
    uint64_t total_bytes = 0;
    double total_latency = 0.0;
    
    for (const auto& result : results_) {
        double latency_ns = result.GetLatencyNs();
        latencies.push_back(latency_ns);
        total_latency += latency_ns;
        
        if (result.success) {
            successful_ops++;
            total_bytes += result.bytes_read;
        } else {
            failed_ops++;
        }
        
        if (result.used_learned_index) {
            learned_index_ops++;
        }
        
        if (result.cache_hit) {
            cache_hits++;
        }
    }
    
    // Sort for percentile calculations
    std::sort(latencies.begin(), latencies.end());
    
    // Calculate metrics
    metrics.total_operations = results_.size();
    metrics.successful_operations = successful_ops;
    metrics.failed_operations = failed_ops;
    
    metrics.avg_latency_ns = total_latency / results_.size();
    metrics.min_latency_ns = latencies.front();
    metrics.max_latency_ns = latencies.back();
    metrics.p50_latency_ns = latencies[latencies.size() / 2];
    metrics.p95_latency_ns = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    metrics.p99_latency_ns = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    
    // Calculate throughput
    double total_time_seconds = total_latency / 1e9;
    metrics.operations_per_second = successful_ops / total_time_seconds;
    metrics.mb_per_second = (total_bytes / (1024.0 * 1024.0)) / total_time_seconds;
    
    // Calculate accuracy metrics
    metrics.prediction_accuracy = learned_index_ops > 0 ? 
        static_cast<double>(learned_index_ops) / results_.size() : 0.0;
    metrics.cache_hit_rate = cache_hits > 0 ? 
        static_cast<double>(cache_hits) / results_.size() : 0.0;
    metrics.fallback_rate = 1.0 - metrics.prediction_accuracy;
    
    // Memory usage
    metrics.memory_usage_bytes = memory_tracker_.GetPeakUsage();
    
    return metrics;
}

void BenchmarkRunner::SaveResults(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    file << "operation_id,latency_ns,success,used_learned_index,cache_hit,predicted_block,actual_block,bytes_read\n";
    
    for (size_t i = 0; i < results_.size(); ++i) {
        const auto& result = results_[i];
        file << i << ","
             << result.GetLatencyNs() << ","
             << (result.success ? 1 : 0) << ","
             << (result.used_learned_index ? 1 : 0) << ","
             << (result.cache_hit ? 1 : 0) << ","
             << result.predicted_block << ","
             << result.actual_block << ","
             << result.bytes_read << "\n";
    }
}

// BenchmarkComparison implementation
void BenchmarkComparison::AddResult(const std::string& name, const PerformanceMetrics& metrics) {
    results_[name] = metrics;
}

void BenchmarkComparison::RunComparison(const std::vector<BenchmarkConfig>& configs) {
    configs_ = configs;
    results_.clear();
    
    for (const auto& config : configs) {
        std::string config_name = (config.enable_learned_index ? "LearnedIndex_" : "Traditional_") +
                                 std::to_string(static_cast<int>(config.workload_type));
        
        BenchmarkRunner runner(config);
        if (runner.SetupBenchmark()) {
            runner.RunBenchmark();
            auto metrics = runner.AnalyzeResults();
            AddResult(config_name, metrics);
            runner.CleanupBenchmark();
        }
    }
}

std::map<std::string, double> BenchmarkComparison::CalculateImprovements() const {
    std::map<std::string, double> improvements;
    
    // Find traditional and learned index results for comparison
    for (const auto& learned_result : results_) {
        if (learned_result.first.find("LearnedIndex_") == 0) {
            std::string workload_type = learned_result.first.substr(13); // Remove "LearnedIndex_"
            std::string traditional_name = "Traditional_" + workload_type;
            
            auto traditional_it = results_.find(traditional_name);
            if (traditional_it != results_.end()) {
                const auto& learned = learned_result.second;
                const auto& traditional = traditional_it->second;
                
                // Calculate improvements (positive = better)
                if (traditional.avg_latency_ns > 0) {
                    double latency_improvement = 
                        (traditional.avg_latency_ns - learned.avg_latency_ns) / traditional.avg_latency_ns * 100.0;
                    improvements[workload_type + "_latency"] = latency_improvement;
                }
                
                if (traditional.operations_per_second > 0) {
                    double throughput_improvement = 
                        (learned.operations_per_second - traditional.operations_per_second) / traditional.operations_per_second * 100.0;
                    improvements[workload_type + "_throughput"] = throughput_improvement;
                }
                
                if (traditional.memory_usage_bytes > 0) {
                    double memory_overhead = 
                        (learned.memory_usage_bytes - traditional.memory_usage_bytes) / traditional.memory_usage_bytes * 100.0;
                    improvements[workload_type + "_memory_overhead"] = memory_overhead;
                }
            }
        }
    }
    
    return improvements;
}

void BenchmarkComparison::PrintSummary() const {
    std::cout << "\n=== Benchmark Results Summary ===\n";
    
    for (const auto& result : results_) {
        std::cout << "\n" << result.first << ":\n";
        const auto& metrics = result.second;
        
        std::cout << "  Avg Latency: " << std::fixed << std::setprecision(2) 
                  << metrics.avg_latency_ns / 1000.0 << " μs\n";
        std::cout << "  P95 Latency: " << metrics.p95_latency_ns / 1000.0 << " μs\n";
        std::cout << "  Throughput: " << std::setprecision(0) 
                  << metrics.operations_per_second << " ops/sec\n";
        std::cout << "  Memory Usage: " << metrics.memory_usage_bytes / 1024 << " KB\n";
        
        if (metrics.prediction_accuracy > 0) {
            std::cout << "  Prediction Accuracy: " << std::setprecision(1) 
                      << metrics.prediction_accuracy * 100.0 << "%\n";
            std::cout << "  Cache Hit Rate: " << metrics.cache_hit_rate * 100.0 << "%\n";
        }
    }
    
    // Show improvements
    auto improvements = CalculateImprovements();
    if (!improvements.empty()) {
        std::cout << "\n=== Improvements ===\n";
        for (const auto& improvement : improvements) {
            std::cout << improvement.first << ": " 
                      << std::showpos << std::setprecision(1) << improvement.second << "%\n";
        }
    }
}

void BenchmarkComparison::GenerateTextReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "Learned Index RocksDB - Performance Benchmark Report\n";
    file << "================================================\n\n";
    
    file << "Benchmark Results:\n";
    file << "------------------\n";
    
    for (const auto& result : results_) {
        file << "\n" << result.first << ":\n";
        const auto& metrics = result.second;
        
        file << std::fixed << std::setprecision(2);
        file << "  Total Operations: " << metrics.total_operations << "\n";
        file << "  Successful Operations: " << metrics.successful_operations << "\n";
        file << "  Average Latency: " << metrics.avg_latency_ns / 1000.0 << " μs\n";
        file << "  P50 Latency: " << metrics.p50_latency_ns / 1000.0 << " μs\n";
        file << "  P95 Latency: " << metrics.p95_latency_ns / 1000.0 << " μs\n";
        file << "  P99 Latency: " << metrics.p99_latency_ns / 1000.0 << " μs\n";
        file << "  Max Latency: " << metrics.max_latency_ns / 1000.0 << " μs\n";
        file << "  Throughput: " << std::setprecision(0) << metrics.operations_per_second << " ops/sec\n";
        file << "  Bandwidth: " << std::setprecision(2) << metrics.mb_per_second << " MB/sec\n";
        file << "  Memory Usage: " << metrics.memory_usage_bytes / 1024 << " KB\n";
        
        if (metrics.prediction_accuracy > 0) {
            file << "  Prediction Accuracy: " << std::setprecision(1) << metrics.prediction_accuracy * 100.0 << "%\n";
            file << "  Cache Hit Rate: " << metrics.cache_hit_rate * 100.0 << "%\n";
            file << "  Fallback Rate: " << metrics.fallback_rate * 100.0 << "%\n";
        }
    }
    
    // Performance improvements
    auto improvements = CalculateImprovements();
    if (!improvements.empty()) {
        file << "\nPerformance Improvements:\n";
        file << "------------------------\n";
        for (const auto& improvement : improvements) {
            file << improvement.first << ": " << std::showpos << std::setprecision(1) 
                 << improvement.second << "%\n";
        }
    }
}

void BenchmarkComparison::GenerateCSVReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    // Header
    file << "Configuration,Total_Ops,Success_Ops,Avg_Latency_us,P50_Latency_us,P95_Latency_us,"
         << "P99_Latency_us,Max_Latency_us,Throughput_ops_sec,Bandwidth_MB_sec,Memory_KB,"
         << "Prediction_Accuracy,Cache_Hit_Rate,Fallback_Rate\n";
    
    // Data rows
    for (const auto& result : results_) {
        const auto& metrics = result.second;
        file << result.first << ","
             << metrics.total_operations << ","
             << metrics.successful_operations << ","
             << std::fixed << std::setprecision(2) << metrics.avg_latency_ns / 1000.0 << ","
             << metrics.p50_latency_ns / 1000.0 << ","
             << metrics.p95_latency_ns / 1000.0 << ","
             << metrics.p99_latency_ns / 1000.0 << ","
             << metrics.max_latency_ns / 1000.0 << ","
             << std::setprecision(0) << metrics.operations_per_second << ","
             << std::setprecision(2) << metrics.mb_per_second << ","
             << metrics.memory_usage_bytes / 1024 << ","
             << std::setprecision(3) << metrics.prediction_accuracy << ","
             << metrics.cache_hit_rate << ","
             << metrics.fallback_rate << "\n";
    }
}

void BenchmarkComparison::GenerateHTMLReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "<!DOCTYPE html>\n<html>\n<head>\n";
    file << "<title>Learned Index RocksDB Benchmark Report</title>\n";
    file << "<style>\n";
    file << "body { font-family: Arial, sans-serif; margin: 40px; }\n";
    file << "h1, h2 { color: #333; }\n";
    file << "table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n";
    file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n";
    file << "th { background-color: #f2f2f2; }\n";
    file << ".improvement-positive { color: green; font-weight: bold; }\n";
    file << ".improvement-negative { color: red; font-weight: bold; }\n";
    file << "</style>\n</head>\n<body>\n";
    
    file << "<h1>Learned Index RocksDB - Performance Benchmark Report</h1>\n";
    file << "<h2>Benchmark Results</h2>\n";
    
    file << "<table>\n<tr>\n";
    file << "<th>Configuration</th><th>Avg Latency (μs)</th><th>P95 Latency (μs)</th>";
    file << "<th>Throughput (ops/sec)</th><th>Memory (KB)</th>";
    file << "<th>Prediction Accuracy</th><th>Cache Hit Rate</th></tr>\n";
    
    for (const auto& result : results_) {
        const auto& metrics = result.second;
        file << "<tr>\n<td>" << result.first << "</td>\n";
        file << "<td>" << std::fixed << std::setprecision(2) << metrics.avg_latency_ns / 1000.0 << "</td>\n";
        file << "<td>" << metrics.p95_latency_ns / 1000.0 << "</td>\n";
        file << "<td>" << std::setprecision(0) << metrics.operations_per_second << "</td>\n";
        file << "<td>" << metrics.memory_usage_bytes / 1024 << "</td>\n";
        file << "<td>" << std::setprecision(1) << metrics.prediction_accuracy * 100.0 << "%</td>\n";
        file << "<td>" << metrics.cache_hit_rate * 100.0 << "%</td>\n</tr>\n";
    }
    file << "</table>\n";
    
    // Performance improvements
    auto improvements = CalculateImprovements();
    if (!improvements.empty()) {
        file << "<h2>Performance Improvements</h2>\n<table>\n";
        file << "<tr><th>Metric</th><th>Improvement</th></tr>\n";
        
        for (const auto& improvement : improvements) {
            std::string css_class = improvement.second >= 0 ? "improvement-positive" : "improvement-negative";
            file << "<tr><td>" << improvement.first << "</td>";
            file << "<td class=\"" << css_class << "\">" << std::showpos 
                 << std::setprecision(1) << improvement.second << "%</td></tr>\n";
        }
        file << "</table>\n";
    }
    
    file << "</body>\n</html>\n";
}

void BenchmarkComparison::GenerateJSONReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"benchmark_results\": {\n";
    
    bool first = true;
    for (const auto& result : results_) {
        if (!first) file << ",\n";
        first = false;
        
        const auto& metrics = result.second;
        file << "    \"" << result.first << "\": {\n";
        file << "      \"total_operations\": " << metrics.total_operations << ",\n";
        file << "      \"successful_operations\": " << metrics.successful_operations << ",\n";
        file << "      \"avg_latency_ns\": " << metrics.avg_latency_ns << ",\n";
        file << "      \"p50_latency_ns\": " << metrics.p50_latency_ns << ",\n";
        file << "      \"p95_latency_ns\": " << metrics.p95_latency_ns << ",\n";
        file << "      \"p99_latency_ns\": " << metrics.p99_latency_ns << ",\n";
        file << "      \"max_latency_ns\": " << metrics.max_latency_ns << ",\n";
        file << "      \"operations_per_second\": " << metrics.operations_per_second << ",\n";
        file << "      \"mb_per_second\": " << metrics.mb_per_second << ",\n";
        file << "      \"memory_usage_bytes\": " << metrics.memory_usage_bytes << ",\n";
        file << "      \"prediction_accuracy\": " << metrics.prediction_accuracy << ",\n";
        file << "      \"cache_hit_rate\": " << metrics.cache_hit_rate << ",\n";
        file << "      \"fallback_rate\": " << metrics.fallback_rate << "\n";
        file << "    }";
    }
    
    file << "\n  },\n";
    
    // Performance improvements
    auto improvements = CalculateImprovements();
    file << "  \"improvements\": {\n";
    first = true;
    for (const auto& improvement : improvements) {
        if (!first) file << ",\n";
        first = false;
        file << "    \"" << improvement.first << "\": " << improvement.second;
    }
    file << "\n  }\n";
    
    file << "}\n";
}

} // namespace benchmark
} // namespace learned_index
} // namespace rocksdb