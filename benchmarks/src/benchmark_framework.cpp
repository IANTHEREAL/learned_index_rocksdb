#include "../include/benchmark_framework.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace benchmark {

// PerformanceTimer Implementation
void PerformanceTimer::StartMeasurement() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void PerformanceTimer::EndMeasurement() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
    measurements_.push_back(duration.count() / 1000.0); // Convert to microseconds
}

void PerformanceTimer::Reset() {
    measurements_.clear();
}

double PerformanceTimer::GetAverageLatencyUs() const {
    if (measurements_.empty()) return 0.0;
    
    double sum = 0.0;
    for (double measurement : measurements_) {
        sum += measurement;
    }
    return sum / measurements_.size();
}

double PerformanceTimer::GetPercentileLatencyUs(double percentile) const {
    if (measurements_.empty()) return 0.0;
    
    std::vector<double> sorted_measurements = measurements_;
    std::sort(sorted_measurements.begin(), sorted_measurements.end());
    
    size_t index = static_cast<size_t>((percentile / 100.0) * (sorted_measurements.size() - 1));
    index = std::min(index, sorted_measurements.size() - 1);
    
    return sorted_measurements[index];
}

std::vector<double> PerformanceTimer::GetAllMeasurementsUs() const {
    return measurements_;
}

// MemoryTracker Implementation
void MemoryTracker::AddAllocation(size_t bytes) {
    current_memory_ += bytes;
    peak_memory_ = std::max(peak_memory_, current_memory_);
}

void MemoryTracker::RemoveAllocation(size_t bytes) {
    current_memory_ = (current_memory_ >= bytes) ? current_memory_ - bytes : 0;
}

// WorkloadGenerator Implementation
WorkloadGenerator::WorkloadGenerator(const WorkloadConfig& config) : config_(config) {}

std::vector<std::pair<uint64_t, uint32_t>> WorkloadGenerator::GenerateTrainingData() {
    std::vector<uint64_t> keys;
    
    // Generate keys based on workload type
    switch (config_.type) {
        case WorkloadType::Sequential:
            keys = GenerateSequentialKeys(config_.dataset_size, config_.key_range_min, 
                                        (config_.key_range_max - config_.key_range_min) / config_.dataset_size);
            break;
        case WorkloadType::Random:
            keys = GenerateRandomKeys(config_.dataset_size, config_.key_range_min, config_.key_range_max);
            break;
        case WorkloadType::Mixed: {
            size_t sequential_count = static_cast<size_t>(config_.dataset_size * config_.sequential_ratio);
            size_t random_count = config_.dataset_size - sequential_count;
            
            auto seq_keys = GenerateSequentialKeys(sequential_count, config_.key_range_min, 10);
            auto rand_keys = GenerateRandomKeys(random_count, config_.key_range_min, config_.key_range_max);
            
            keys.insert(keys.end(), seq_keys.begin(), seq_keys.end());
            keys.insert(keys.end(), rand_keys.begin(), rand_keys.end());
            break;
        }
        case WorkloadType::Zipfian:
            keys = GenerateZipfianKeys(config_.dataset_size, config_.key_range_min, config_.key_range_max);
            break;
        case WorkloadType::Temporal:
            keys = GenerateTemporalKeys(config_.dataset_size);
            break;
    }
    
    // Sort keys to simulate SST file organization
    std::sort(keys.begin(), keys.end());
    
    // Map keys to blocks (simulate block boundaries)
    std::vector<std::pair<uint64_t, uint32_t>> training_data;
    size_t keys_per_block = std::max(1UL, keys.size() / 100); // Assume 100 blocks
    
    for (size_t i = 0; i < keys.size(); ++i) {
        uint32_t block_id = static_cast<uint32_t>(i / keys_per_block);
        training_data.emplace_back(keys[i], block_id);
    }
    
    return training_data;
}

std::vector<uint64_t> WorkloadGenerator::GenerateQueryKeys() {
    std::vector<uint64_t> query_keys;
    
    // Generate query keys based on workload type
    switch (config_.type) {
        case WorkloadType::Sequential:
            query_keys = GenerateSequentialKeys(config_.num_queries, config_.key_range_min, 
                                              (config_.key_range_max - config_.key_range_min) / config_.num_queries);
            break;
        case WorkloadType::Random:
            query_keys = GenerateRandomKeys(config_.num_queries, config_.key_range_min, config_.key_range_max);
            break;
        case WorkloadType::Mixed: {
            size_t sequential_count = static_cast<size_t>(config_.num_queries * config_.sequential_ratio);
            size_t random_count = config_.num_queries - sequential_count;
            
            auto seq_keys = GenerateSequentialKeys(sequential_count, config_.key_range_min, 5);
            auto rand_keys = GenerateRandomKeys(random_count, config_.key_range_min, config_.key_range_max);
            
            query_keys.insert(query_keys.end(), seq_keys.begin(), seq_keys.end());
            query_keys.insert(query_keys.end(), rand_keys.begin(), rand_keys.end());
            
            // Shuffle to mix sequential and random queries
            std::mt19937 rng(config_.seed);
            std::shuffle(query_keys.begin(), query_keys.end(), rng);
            break;
        }
        case WorkloadType::Zipfian:
            query_keys = GenerateZipfianKeys(config_.num_queries, config_.key_range_min, config_.key_range_max);
            break;
        case WorkloadType::Temporal:
            query_keys = GenerateTemporalKeys(config_.num_queries);
            break;
    }
    
    return query_keys;
}

std::string WorkloadGenerator::GetWorkloadDescription() const {
    std::string desc;
    switch (config_.type) {
        case WorkloadType::Sequential: desc = "Sequential"; break;
        case WorkloadType::Random: desc = "Random"; break;
        case WorkloadType::Mixed: desc = "Mixed"; break;
        case WorkloadType::Zipfian: desc = "Zipfian"; break;
        case WorkloadType::Temporal: desc = "Temporal"; break;
    }
    
    std::stringstream ss;
    ss << desc << " (dataset=" << config_.dataset_size 
       << ", queries=" << config_.num_queries << ")";
    return ss.str();
}

std::vector<uint64_t> WorkloadGenerator::GenerateSequentialKeys(size_t count, uint64_t start, uint64_t step) {
    std::vector<uint64_t> keys;
    keys.reserve(count);
    
    uint64_t current_key = start;
    for (size_t i = 0; i < count; ++i) {
        keys.push_back(current_key);
        current_key += step;
    }
    
    return keys;
}

std::vector<uint64_t> WorkloadGenerator::GenerateRandomKeys(size_t count, uint64_t min_key, uint64_t max_key) {
    std::vector<uint64_t> keys;
    keys.reserve(count);
    
    std::mt19937_64 rng(config_.seed);
    std::uniform_int_distribution<uint64_t> dist(min_key, max_key);
    
    for (size_t i = 0; i < count; ++i) {
        keys.push_back(dist(rng));
    }
    
    return keys;
}

std::vector<uint64_t> WorkloadGenerator::GenerateZipfianKeys(size_t count, uint64_t min_key, uint64_t max_key) {
    std::vector<uint64_t> keys;
    keys.reserve(count);
    
    std::mt19937 rng(config_.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    // Simple Zipfian approximation
    uint64_t range = max_key - min_key + 1;
    
    for (size_t i = 0; i < count; ++i) {
        double u = uniform(rng);
        // Zipfian approximation: P(k) ∝ k^(-θ)
        uint64_t k = static_cast<uint64_t>(std::pow(u, -1.0 / config_.zipfian_theta) * range);
        k = std::min(k, range - 1);
        keys.push_back(min_key + k);
    }
    
    return keys;
}

std::vector<uint64_t> WorkloadGenerator::GenerateTemporalKeys(size_t count) {
    std::vector<uint64_t> keys;
    keys.reserve(count);
    
    // Generate time-based keys (simulating timestamp-like data)
    uint64_t base_timestamp = 1600000000; // Sept 2020 timestamp
    uint64_t time_increment = 86400;      // 1 day in seconds
    
    std::mt19937_64 rng(config_.seed);
    std::normal_distribution<double> jitter(0.0, time_increment * 0.1); // 10% jitter
    
    for (size_t i = 0; i < count; ++i) {
        uint64_t timestamp = base_timestamp + i * time_increment + static_cast<uint64_t>(jitter(rng));
        keys.push_back(timestamp);
    }
    
    return keys;
}

uint32_t WorkloadGenerator::MapKeyToBlock(uint64_t key, size_t num_blocks) const {
    uint64_t range = config_.key_range_max - config_.key_range_min + 1;
    uint64_t normalized_key = key - config_.key_range_min;
    return static_cast<uint32_t>((normalized_key * num_blocks) / range);
}

// BenchmarkRunner Implementation
BenchmarkRunner::BenchmarkRunner() {}

void BenchmarkRunner::AddIndex(std::unique_ptr<IndexInterface> index) {
    indexes_.push_back(std::move(index));
}

void BenchmarkRunner::RunBenchmark(const WorkloadConfig& workload_config) {
    std::cout << "\n=== Running Benchmark: " << WorkloadGenerator(workload_config).GetWorkloadDescription() << " ===" << std::endl;
    
    for (auto& index : indexes_) {
        BenchmarkResult result = RunSingleBenchmark(index.get(), workload_config);
        results_.push_back(result);
        
        std::cout << "\nIndex: " << result.index_type << std::endl;
        std::cout << "  Avg Latency: " << std::fixed << std::setprecision(2) << result.avg_lookup_latency_us << " μs" << std::endl;
        std::cout << "  P95 Latency: " << result.p95_lookup_latency_us << " μs" << std::endl;
        std::cout << "  Throughput:  " << std::setprecision(0) << result.throughput_qps << " QPS" << std::endl;
        std::cout << "  Memory:      " << (result.index_memory_bytes / 1024.0) << " KB" << std::endl;
        if (result.prediction_accuracy > 0) {
            std::cout << "  Accuracy:    " << std::setprecision(1) << (result.prediction_accuracy * 100) << "%" << std::endl;
        }
    }
}

BenchmarkResult BenchmarkRunner::RunSingleBenchmark(IndexInterface* index, 
                                                   const WorkloadConfig& workload_config) {
    BenchmarkResult result;
    result.index_type = index->GetIndexType();
    result.workload_type = WorkloadGenerator(workload_config).GetWorkloadDescription();
    result.dataset_size = workload_config.dataset_size;
    result.num_queries = workload_config.num_queries;
    
    WorkloadGenerator generator(workload_config);
    
    // Generate training data and train the index
    memory_tracker_.Reset();
    auto training_data = generator.GenerateTrainingData();
    
    bool training_success = index->Train(training_data);
    
    if (!training_success) {
        std::cerr << "Training failed for index: " << index->GetIndexType() << std::endl;
        return result;
    }
    
    // Measure index memory usage
    result.index_memory_bytes = index->GetMemoryUsage();
    
    // Generate query keys
    auto query_keys = generator.GenerateQueryKeys();
    
    // Run queries and measure performance
    timer_.Reset();
    
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t key : query_keys) {
        timer_.StartMeasurement();
        uint32_t block = index->Lookup(key);
        timer_.EndMeasurement();
        
        // Prevent compiler optimization
        volatile uint32_t dummy = block;
        (void)dummy;
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    
    // Calculate results
    result.avg_lookup_latency_us = timer_.GetAverageLatencyUs();
    result.p50_lookup_latency_us = timer_.GetPercentileLatencyUs(50.0);
    result.p95_lookup_latency_us = timer_.GetPercentileLatencyUs(95.0);
    result.p99_lookup_latency_us = timer_.GetPercentileLatencyUs(99.0);
    
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start);
    result.total_time_us = total_duration.count();
    result.throughput_qps = (static_cast<double>(workload_config.num_queries) / result.total_time_us) * 1000000.0;
    
    result.peak_memory_bytes = memory_tracker_.GetPeakMemory();
    
    // Get index-specific statistics
    index->GetStats(result);
    
    return result;
}

void BenchmarkRunner::SaveResults(const std::string& output_file) const {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }
    
    // CSV header
    file << "test_name,index_type,workload_type,dataset_size,num_queries,"
         << "avg_latency_us,p50_latency_us,p95_latency_us,p99_latency_us,"
         << "throughput_qps,index_memory_bytes,peak_memory_bytes,"
         << "prediction_accuracy,fallback_rate\n";
    
    for (const auto& result : results_) {
        file << result.test_name << ","
             << result.index_type << ","
             << result.workload_type << ","
             << result.dataset_size << ","
             << result.num_queries << ","
             << result.avg_lookup_latency_us << ","
             << result.p50_lookup_latency_us << ","
             << result.p95_lookup_latency_us << ","
             << result.p99_lookup_latency_us << ","
             << result.throughput_qps << ","
             << result.index_memory_bytes << ","
             << result.peak_memory_bytes << ","
             << result.prediction_accuracy << ","
             << result.fallback_rate << "\n";
    }
    
    file.close();
    std::cout << "Results saved to: " << output_file << std::endl;
}

void BenchmarkRunner::PrintResults() const {
    std::cout << "\n=== Benchmark Summary ===" << std::endl;
    
    for (const auto& result : results_) {
        std::cout << "\nTest: " << result.workload_type << " - " << result.index_type << std::endl;
        std::cout << "  Dataset Size: " << result.dataset_size << std::endl;
        std::cout << "  Queries: " << result.num_queries << std::endl;
        std::cout << "  Avg Latency: " << std::fixed << std::setprecision(2) << result.avg_lookup_latency_us << " μs" << std::endl;
        std::cout << "  P95 Latency: " << result.p95_lookup_latency_us << " μs" << std::endl;
        std::cout << "  P99 Latency: " << result.p99_lookup_latency_us << " μs" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(0) << result.throughput_qps << " QPS" << std::endl;
        std::cout << "  Index Memory: " << (result.index_memory_bytes / 1024.0) << " KB" << std::endl;
        if (result.prediction_accuracy > 0) {
            std::cout << "  Accuracy: " << std::setprecision(1) << (result.prediction_accuracy * 100) << "%" << std::endl;
            std::cout << "  Fallback Rate: " << (result.fallback_rate * 100) << "%" << std::endl;
        }
    }
}

void BenchmarkRunner::GenerateCharts(const std::string& output_dir) const {
    GenerateLatencyChart(output_dir + "/latency_comparison.py");
    GenerateMemoryChart(output_dir + "/memory_comparison.py");
    GenerateThroughputChart(output_dir + "/throughput_comparison.py");
    GenerateAccuracyChart(output_dir + "/accuracy_comparison.py");
}

void BenchmarkRunner::GenerateLatencyChart(const std::string& output_file) const {
    std::ofstream file(output_file);
    if (!file.is_open()) return;
    
    file << "#!/usr/bin/env python3\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n\n";
    
    file << "# Latency Comparison Chart\n";
    file << "workloads = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << "'" << results_[i].workload_type << "'";
    }
    file << "]\n\n";
    
    file << "learned_index_latency = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << results_[i].avg_lookup_latency_us;
    }
    file << "]\n\n";
    
    file << "btree_latency = [";
    for (size_t i = 1; i < results_.size(); i += 2) {
        if (i > 1) file << ", ";
        file << results_[i].avg_lookup_latency_us;
    }
    file << "]\n\n";
    
    file << R"(
x = np.arange(len(workloads))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, learned_index_latency, width, label='Learned Index', color='skyblue')
bars2 = ax.bar(x + width/2, btree_latency, width, label='B+ Tree', color='lightcoral')

ax.set_xlabel('Workload Type')
ax.set_ylabel('Average Lookup Latency (μs)')
ax.set_title('Lookup Latency Comparison: Learned Index vs B+ Tree')
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
)";
    
    file.close();
}

void BenchmarkRunner::GenerateMemoryChart(const std::string& output_file) const {
    std::ofstream file(output_file);
    if (!file.is_open()) return;
    
    file << "#!/usr/bin/env python3\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n\n";
    
    file << "# Memory Usage Comparison Chart\n";
    file << "workloads = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << "'" << results_[i].workload_type << "'";
    }
    file << "]\n\n";
    
    file << "learned_index_memory = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << (results_[i].index_memory_bytes / 1024.0); // Convert to KB
    }
    file << "]\n\n";
    
    file << "btree_memory = [";
    for (size_t i = 1; i < results_.size(); i += 2) {
        if (i > 1) file << ", ";
        file << (results_[i].index_memory_bytes / 1024.0); // Convert to KB
    }
    file << "]\n\n";
    
    file << R"(
x = np.arange(len(workloads))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, learned_index_memory, width, label='Learned Index', color='lightgreen')
bars2 = ax.bar(x + width/2, btree_memory, width, label='B+ Tree', color='orange')

ax.set_xlabel('Workload Type')
ax.set_ylabel('Index Memory Usage (KB)')
ax.set_title('Memory Usage Comparison: Learned Index vs B+ Tree')
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
)";
    
    file.close();
}

void BenchmarkRunner::GenerateThroughputChart(const std::string& output_file) const {
    std::ofstream file(output_file);
    if (!file.is_open()) return;
    
    file << "#!/usr/bin/env python3\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n\n";
    
    file << "# Throughput Comparison Chart\n";
    file << "workloads = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << "'" << results_[i].workload_type << "'";
    }
    file << "]\n\n";
    
    file << "learned_index_throughput = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << results_[i].throughput_qps;
    }
    file << "]\n\n";
    
    file << "btree_throughput = [";
    for (size_t i = 1; i < results_.size(); i += 2) {
        if (i > 1) file << ", ";
        file << results_[i].throughput_qps;
    }
    file << "]\n\n";
    
    file << R"(
x = np.arange(len(workloads))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, learned_index_throughput, width, label='Learned Index', color='gold')
bars2 = ax.bar(x + width/2, btree_throughput, width, label='B+ Tree', color='purple')

ax.set_xlabel('Workload Type')
ax.set_ylabel('Throughput (Queries Per Second)')
ax.set_title('Throughput Comparison: Learned Index vs B+ Tree')
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
)";
    
    file.close();
}

void BenchmarkRunner::GenerateAccuracyChart(const std::string& output_file) const {
    std::ofstream file(output_file);
    if (!file.is_open()) return;
    
    file << "#!/usr/bin/env python3\n";
    file << "import matplotlib.pyplot as plt\n";
    file << "import numpy as np\n\n";
    
    file << "# Accuracy Analysis Chart (Learned Index Only)\n";
    file << "workloads = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << "'" << results_[i].workload_type << "'";
    }
    file << "]\n\n";
    
    file << "accuracy = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << (results_[i].prediction_accuracy * 100);
    }
    file << "]\n\n";
    
    file << "fallback_rate = [";
    for (size_t i = 0; i < results_.size(); i += 2) {
        if (i > 0) file << ", ";
        file << (results_[i].fallback_rate * 100);
    }
    file << "]\n\n";
    
    file << R"(
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
bars1 = ax1.bar(workloads, accuracy, color='mediumseagreen')
ax1.set_xlabel('Workload Type')
ax1.set_ylabel('Prediction Accuracy (%)')
ax1.set_title('Learned Index Prediction Accuracy')
ax1.set_ylim(0, 100)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Fallback rate plot
bars2 = ax2.bar(workloads, fallback_rate, color='salmon')
ax2.set_xlabel('Workload Type')
ax2.set_ylabel('Fallback Rate (%)')
ax2.set_title('Learned Index Fallback Rate')
ax2.set_ylim(0, 100)

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
)";
    
    file.close();
}

} // namespace benchmark