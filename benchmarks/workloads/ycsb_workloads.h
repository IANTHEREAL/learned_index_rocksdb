#pragma once

#include "learned_index/benchmark_framework.h"

namespace rocksdb {
namespace learned_index {
namespace benchmark {
namespace ycsb {

// YCSB (Yahoo! Cloud Serving Benchmark) workload configurations
// These replicate the standard YCSB workloads for database benchmarking

class YCSBWorkloadA : public WorkloadGenerator {
public:
    // Workload A: Update heavy workload
    // 50% reads, 50% updates
    // Request distribution: Zipfian
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "YCSB-A"; }
    std::string GetDescription() const override { 
        return "Update heavy workload (50% reads, 50% updates) with Zipfian distribution"; 
    }
};

class YCSBWorkloadB : public WorkloadGenerator {
public:
    // Workload B: Read mostly workload
    // 95% reads, 5% updates
    // Request distribution: Zipfian
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "YCSB-B"; }
    std::string GetDescription() const override { 
        return "Read mostly workload (95% reads, 5% updates) with Zipfian distribution"; 
    }
};

class YCSBWorkloadC : public WorkloadGenerator {
public:
    // Workload C: Read only
    // 100% reads
    // Request distribution: Zipfian
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "YCSB-C"; }
    std::string GetDescription() const override { 
        return "Read only workload (100% reads) with Zipfian distribution"; 
    }
};

class YCSBWorkloadD : public WorkloadGenerator {
public:
    // Workload D: Read latest workload
    // 95% reads, 5% inserts
    // Read distribution: Latest distribution (recently inserted records)
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "YCSB-D"; }
    std::string GetDescription() const override { 
        return "Read latest workload (95% reads, 5% inserts) with latest distribution"; 
    }
};

class YCSBWorkloadE : public WorkloadGenerator {
public:
    // Workload E: Short ranges
    // 95% scans, 5% inserts
    // Scan length distribution: Uniform 1-100
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "YCSB-E"; }
    std::string GetDescription() const override { 
        return "Short ranges workload (95% scans, 5% inserts)"; 
    }
};

class YCSBWorkloadF : public WorkloadGenerator {
public:
    // Workload F: Read-modify-write
    // 50% reads, 50% read-modify-write
    // Request distribution: Zipfian
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "YCSB-F"; }
    std::string GetDescription() const override { 
        return "Read-modify-write workload (50% reads, 50% RMW) with Zipfian distribution"; 
    }
};

// Time-series specific workloads
class TimeSeriesWorkload : public WorkloadGenerator {
public:
    // Time-series workload with temporal locality
    // Keys represent timestamps with high temporal locality
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "TimeSeries"; }
    std::string GetDescription() const override { 
        return "Time-series workload with temporal locality patterns"; 
    }
};

// Log-structured workloads
class LogStructuredWorkload : public WorkloadGenerator {
public:
    // Append-heavy workload simulating log-structured data
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "LogStructured"; }
    std::string GetDescription() const override { 
        return "Log-structured workload with append-heavy patterns"; 
    }
};

// Analytics workloads
class AnalyticsWorkload : public WorkloadGenerator {
public:
    // Large range scans typical of analytical queries
    std::vector<uint64_t> GenerateKeys(const BenchmarkConfig& config) override;
    std::string GetName() const override { return "Analytics"; }
    std::string GetDescription() const override { 
        return "Analytics workload with large range scans"; 
    }
};

// Utility functions for YCSB workloads
class YCSBUtils {
public:
    // Generate Zipfian distributed values
    static std::vector<uint64_t> GenerateZipfian(size_t num_values, size_t max_key, 
                                                 double alpha = 0.99, uint64_t seed = 42);
    
    // Generate Latest distribution (exponentially decaying probability)
    static std::vector<uint64_t> GenerateLatest(size_t num_values, size_t max_key, 
                                                uint64_t seed = 42);
    
    // Generate Uniform distribution
    static std::vector<uint64_t> GenerateUniform(size_t num_values, size_t max_key, 
                                                 uint64_t seed = 42);
    
    // Generate Hotspot distribution (90% of accesses to 10% of data)
    static std::vector<uint64_t> GenerateHotspot(size_t num_values, size_t max_key, 
                                                 double hot_fraction = 0.1, 
                                                 double hot_probability = 0.9,
                                                 uint64_t seed = 42);
    
    // Generate temporal patterns for time-series workloads
    static std::vector<uint64_t> GenerateTemporal(size_t num_values, size_t max_key,
                                                  double locality_strength = 0.8,
                                                  uint64_t seed = 42);
};

// Benchmark configuration factory for YCSB workloads
class YCSBConfigFactory {
public:
    static BenchmarkConfig CreateWorkloadA(size_t num_keys = 1000000, 
                                          size_t num_operations = 100000);
    
    static BenchmarkConfig CreateWorkloadB(size_t num_keys = 1000000, 
                                          size_t num_operations = 100000);
    
    static BenchmarkConfig CreateWorkloadC(size_t num_keys = 1000000, 
                                          size_t num_operations = 100000);
    
    static BenchmarkConfig CreateWorkloadD(size_t num_keys = 1000000, 
                                          size_t num_operations = 100000);
    
    static BenchmarkConfig CreateWorkloadE(size_t num_keys = 1000000, 
                                          size_t num_operations = 100000);
    
    static BenchmarkConfig CreateWorkloadF(size_t num_keys = 1000000, 
                                          size_t num_operations = 100000);
    
    static BenchmarkConfig CreateTimeSeriesWorkload(size_t num_keys = 1000000, 
                                                    size_t num_operations = 100000);
    
    static BenchmarkConfig CreateLogStructuredWorkload(size_t num_keys = 1000000, 
                                                       size_t num_operations = 100000);
    
    static BenchmarkConfig CreateAnalyticsWorkload(size_t num_keys = 1000000, 
                                                   size_t num_operations = 100000);
};

} // namespace ycsb
} // namespace benchmark
} // namespace learned_index
} // namespace rocksdb