#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <functional>

namespace benchmark {

struct BenchmarkResult {
    std::string test_name;
    std::string index_type;
    std::string workload_type;
    size_t dataset_size;
    size_t num_queries;
    
    // Timing results (microseconds)
    double avg_lookup_latency_us;
    double p50_lookup_latency_us;
    double p95_lookup_latency_us;
    double p99_lookup_latency_us;
    double total_time_us;
    
    // Memory results (bytes)
    size_t index_memory_bytes;
    size_t peak_memory_bytes;
    
    // Throughput (queries per second)
    double throughput_qps;
    
    // Accuracy (for learned index)
    double prediction_accuracy;
    double fallback_rate;
    size_t successful_predictions;
    size_t total_predictions;
    
    BenchmarkResult() : dataset_size(0), num_queries(0), avg_lookup_latency_us(0),
                       p50_lookup_latency_us(0), p95_lookup_latency_us(0), 
                       p99_lookup_latency_us(0), total_time_us(0),
                       index_memory_bytes(0), peak_memory_bytes(0), throughput_qps(0),
                       prediction_accuracy(0), fallback_rate(0), 
                       successful_predictions(0), total_predictions(0) {}
};

class PerformanceTimer {
public:
    PerformanceTimer() : measurements_() {}
    
    void StartMeasurement();
    void EndMeasurement();
    void Reset();
    
    double GetAverageLatencyUs() const;
    double GetPercentileLatencyUs(double percentile) const;
    std::vector<double> GetAllMeasurementsUs() const;
    size_t GetMeasurementCount() const { return measurements_.size(); }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<double> measurements_; // in microseconds
};

class MemoryTracker {
public:
    MemoryTracker() : peak_memory_(0), current_memory_(0) {}
    
    void AddAllocation(size_t bytes);
    void RemoveAllocation(size_t bytes);
    size_t GetPeakMemory() const { return peak_memory_; }
    size_t GetCurrentMemory() const { return current_memory_; }
    void Reset() { peak_memory_ = 0; current_memory_ = 0; }

private:
    size_t peak_memory_;
    size_t current_memory_;
};

enum class WorkloadType {
    Sequential,
    Random,
    Mixed,
    Zipfian,
    Temporal
};

struct WorkloadConfig {
    WorkloadType type;
    size_t dataset_size;
    size_t num_queries;
    uint64_t key_range_min;
    uint64_t key_range_max;
    double sequential_ratio;  // For mixed workloads
    double zipfian_theta;     // For Zipfian distribution
    uint32_t seed;
    
    WorkloadConfig() : type(WorkloadType::Sequential), dataset_size(10000), 
                      num_queries(1000), key_range_min(1000), key_range_max(100000),
                      sequential_ratio(0.8), zipfian_theta(0.99), seed(42) {}
};

class WorkloadGenerator {
public:
    explicit WorkloadGenerator(const WorkloadConfig& config);
    
    std::vector<std::pair<uint64_t, uint32_t>> GenerateTrainingData();
    std::vector<uint64_t> GenerateQueryKeys();
    
    std::string GetWorkloadDescription() const;

private:
    WorkloadConfig config_;
    
    std::vector<uint64_t> GenerateSequentialKeys(size_t count, uint64_t start, uint64_t step);
    std::vector<uint64_t> GenerateRandomKeys(size_t count, uint64_t min_key, uint64_t max_key);
    std::vector<uint64_t> GenerateZipfianKeys(size_t count, uint64_t min_key, uint64_t max_key);
    std::vector<uint64_t> GenerateTemporalKeys(size_t count);
    
    uint32_t MapKeyToBlock(uint64_t key, size_t num_blocks) const;
};

// Abstract base class for index implementations
class IndexInterface {
public:
    virtual ~IndexInterface() = default;
    
    virtual bool Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) = 0;
    virtual uint32_t Lookup(uint64_t key) = 0;
    virtual size_t GetMemoryUsage() const = 0;
    virtual std::string GetIndexType() const = 0;
    virtual void GetStats(BenchmarkResult& result) const { (void)result; }
};

class BenchmarkRunner {
public:
    BenchmarkRunner();
    ~BenchmarkRunner() = default;
    
    void AddIndex(std::unique_ptr<IndexInterface> index);
    void RunBenchmark(const WorkloadConfig& workload_config);
    void SaveResults(const std::string& output_file) const;
    void PrintResults() const;
    void GenerateCharts(const std::string& output_dir) const;

private:
    std::vector<std::unique_ptr<IndexInterface>> indexes_;
    std::vector<BenchmarkResult> results_;
    PerformanceTimer timer_;
    MemoryTracker memory_tracker_;
    
    BenchmarkResult RunSingleBenchmark(IndexInterface* index, 
                                     const WorkloadConfig& workload_config);
    void GenerateLatencyChart(const std::string& output_file) const;
    void GenerateMemoryChart(const std::string& output_file) const;
    void GenerateThroughputChart(const std::string& output_file) const;
    void GenerateAccuracyChart(const std::string& output_file) const;
};

} // namespace benchmark