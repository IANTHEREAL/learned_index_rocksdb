#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include "sst_learned_index_manager.h"

namespace rocksdb {
namespace learned_index {
namespace benchmark {

// Performance metrics for benchmarking
struct PerformanceMetrics {
    // Latency measurements (nanoseconds)
    double avg_latency_ns;
    double p50_latency_ns;
    double p95_latency_ns;
    double p99_latency_ns;
    double max_latency_ns;
    double min_latency_ns;
    
    // Throughput measurements
    double operations_per_second;
    double mb_per_second;
    
    // Accuracy measurements
    double prediction_accuracy;
    double cache_hit_rate;
    double fallback_rate;
    
    // Resource usage
    uint64_t memory_usage_bytes;
    uint64_t cpu_cycles;
    
    // Error counts
    uint64_t total_operations;
    uint64_t successful_operations;
    uint64_t failed_operations;
    
    PerformanceMetrics() 
        : avg_latency_ns(0), p50_latency_ns(0), p95_latency_ns(0), p99_latency_ns(0)
        , max_latency_ns(0), min_latency_ns(0), operations_per_second(0), mb_per_second(0)
        , prediction_accuracy(0), cache_hit_rate(0), fallback_rate(0)
        , memory_usage_bytes(0), cpu_cycles(0), total_operations(0)
        , successful_operations(0), failed_operations(0) {}
};

// Workload types for benchmarking
enum class WorkloadType {
    SEQUENTIAL_READ,
    RANDOM_READ,
    RANGE_QUERY,
    MIXED_WORKLOAD,
    WRITE_HEAVY,
    READ_HEAVY,
    COMPACTION_HEAVY
};

// Benchmark configuration
struct BenchmarkConfig {
    WorkloadType workload_type;
    size_t num_operations;
    size_t num_keys;
    size_t key_size;
    size_t value_size;
    double read_ratio;
    double write_ratio;
    size_t range_size;
    size_t num_threads;
    bool enable_learned_index;
    SSTLearnedIndexOptions learned_index_options;
    std::string output_file;
    
    BenchmarkConfig()
        : workload_type(WorkloadType::RANDOM_READ)
        , num_operations(100000)
        , num_keys(1000000)
        , key_size(16)
        , value_size(100)
        , read_ratio(0.8)
        , write_ratio(0.2)
        , range_size(100)
        , num_threads(1)
        , enable_learned_index(false) {}
};

// Individual operation result
struct OperationResult {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool success;
    bool used_learned_index;
    bool cache_hit;
    uint32_t predicted_block;
    uint32_t actual_block;
    size_t bytes_read;
    
    double GetLatencyNs() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time).count();
    }
};

// Timer class for high-precision timing
class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
    
public:
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void Stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }
    
    double GetElapsedNs() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_ - start_).count();
    }
    
    double GetElapsedMs() const {
        return GetElapsedNs() / 1000000.0;
    }
    
    double GetElapsedSeconds() const {
        return GetElapsedNs() / 1000000000.0;
    }
};

// Memory usage tracker
class MemoryTracker {
private:
    size_t baseline_memory_;
    size_t peak_memory_;
    
public:
    MemoryTracker();
    void RecordBaseline();
    void UpdatePeakUsage();
    size_t GetCurrentUsage() const;
    size_t GetPeakUsage() const;
    size_t GetAdditionalUsage() const;
};

// Workload generator interface
class WorkloadGenerator {
public:
    virtual ~WorkloadGenerator() = default;
    virtual std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) = 0;
    virtual std::string GetName() const = 0;
    virtual std::string GetDescription() const = 0;
};

// Sequential workload generator
class SequentialWorkloadGenerator : public WorkloadGenerator {
public:
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "Sequential"; }
    std::string GetDescription() const override { 
        return "Sequential key access pattern"; 
    }
};

// Random workload generator
class RandomWorkloadGenerator : public WorkloadGenerator {
private:
    uint64_t seed_;
    
public:
    explicit RandomWorkloadGenerator(uint64_t seed = 42) : seed_(seed) {}
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "Random"; }
    std::string GetDescription() const override { 
        return "Uniformly random key access pattern"; 
    }
};

// Zipfian workload generator
class ZipfianWorkloadGenerator : public WorkloadGenerator {
private:
    double alpha_;
    uint64_t seed_;
    
public:
    explicit ZipfianWorkloadGenerator(double alpha = 1.0, uint64_t seed = 42) 
        : alpha_(alpha), seed_(seed) {}
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "Zipfian"; }
    std::string GetDescription() const override { 
        return "Zipfian distributed key access pattern (alpha=" + 
               std::to_string(alpha_) + ")"; 
    }
};

// Mock SST interface for benchmarking
class MockSST {
private:
    std::map<uint64_t, std::pair<uint32_t, std::vector<uint8_t>>> data_;
    std::vector<std::vector<uint64_t>> blocks_;
    size_t block_size_;
    std::unique_ptr<SSTLearnedIndexManager> learned_index_manager_;
    bool learned_index_enabled_;
    
public:
    explicit MockSST(size_t block_size = 4096);
    
    // Data management
    void AddKey(uint64_t key, const std::vector<uint8_t>& value);
    void AddKeys(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>>& keys);
    void Finalize(); // Organize data into blocks and train learned index
    
    // Query operations
    OperationResult Get(uint64_t key);
    std::vector<OperationResult> RangeQuery(uint64_t start_key, uint64_t end_key);
    
    // Configuration
    void EnableLearnedIndex(const SSTLearnedIndexOptions& options);
    void DisableLearnedIndex();
    
    // Statistics
    size_t GetNumBlocks() const { return blocks_.size(); }
    size_t GetNumKeys() const { return data_.size(); }
    size_t GetDataSize() const;
    const SSTLearnedIndexManager* GetLearnedIndexManager() const { 
        return learned_index_manager_.get(); 
    }

private:
    uint32_t FindBlockTraditional(uint64_t key) const;
    void OrganizeIntoBlocks();
    void TrainLearnedIndex();
};

// Main benchmark runner
class BenchmarkRunner {
private:
    BenchmarkConfig config_;
    std::unique_ptr<WorkloadGenerator> workload_generator_;
    std::unique_ptr<MockSST> sst_;
    std::vector<OperationResult> results_;
    MemoryTracker memory_tracker_;
    
public:
    explicit BenchmarkRunner(const BenchmarkConfig& config);
    
    // Setup and teardown
    bool SetupBenchmark();
    void CleanupBenchmark();
    
    // Benchmark execution
    void RunBenchmark();
    void RunSequentialReads();
    void RunRandomReads();
    void RunRangeQueries();
    void RunMixedWorkload();
    
    // Results analysis
    PerformanceMetrics AnalyzeResults() const;
    void SaveResults(const std::string& filename) const;
    
    // Getters
    const std::vector<OperationResult>& GetResults() const { return results_; }
    const BenchmarkConfig& GetConfig() const { return config_; }
};

// Comparison framework
class BenchmarkComparison {
private:
    std::map<std::string, PerformanceMetrics> results_;
    std::vector<BenchmarkConfig> configs_;
    
public:
    void AddResult(const std::string& name, const PerformanceMetrics& metrics);
    void RunComparison(const std::vector<BenchmarkConfig>& configs);
    
    // Report generation
    void GenerateTextReport(const std::string& filename) const;
    void GenerateCSVReport(const std::string& filename) const;
    void GenerateHTMLReport(const std::string& filename) const;
    void GenerateJSONReport(const std::string& filename) const;
    
    // Analysis
    std::map<std::string, double> CalculateImprovements() const;
    void PrintSummary() const;
};

} // namespace benchmark
} // namespace learned_index
} // namespace rocksdb