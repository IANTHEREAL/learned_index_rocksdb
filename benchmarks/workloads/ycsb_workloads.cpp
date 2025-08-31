#include "ycsb_workloads.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace rocksdb {
namespace learned_index {
namespace benchmark {
namespace ycsb {

// YCSBWorkloadA implementation
std::vector<uint64_t> YCSBWorkloadA::GenerateKeys(const BenchmarkConfig& config) {
    return YCSBUtils::GenerateZipfian(config.num_operations, config.num_keys, 0.99);
}

// YCSBWorkloadB implementation
std::vector<uint64_t> YCSBWorkloadB::GenerateKeys(const BenchmarkConfig& config) {
    return YCSBUtils::GenerateZipfian(config.num_operations, config.num_keys, 0.99);
}

// YCSBWorkloadC implementation
std::vector<uint64_t> YCSBWorkloadC::GenerateKeys(const BenchmarkConfig& config) {
    return YCSBUtils::GenerateZipfian(config.num_operations, config.num_keys, 0.99);
}

// YCSBWorkloadD implementation
std::vector<uint64_t> YCSBWorkloadD::GenerateKeys(const BenchmarkConfig& config) {
    return YCSBUtils::GenerateLatest(config.num_operations, config.num_keys);
}

// YCSBWorkloadE implementation
std::vector<uint64_t> YCSBWorkloadE::GenerateKeys(const BenchmarkConfig& config) {
    // Generate starting points for range scans
    return YCSBUtils::GenerateZipfian(config.num_operations, 
                                     config.num_keys - config.range_size, 0.99);
}

// YCSBWorkloadF implementation
std::vector<uint64_t> YCSBWorkloadF::GenerateKeys(const BenchmarkConfig& config) {
    return YCSBUtils::GenerateZipfian(config.num_operations, config.num_keys, 0.99);
}

// TimeSeriesWorkload implementation
std::vector<uint64_t> TimeSeriesWorkload::GenerateKeys(const BenchmarkConfig& config) {
    return YCSBUtils::GenerateTemporal(config.num_operations, config.num_keys, 0.8);
}

// LogStructuredWorkload implementation
std::vector<uint64_t> LogStructuredWorkload::GenerateKeys(const BenchmarkConfig& config) {
    std::vector<uint64_t> keys;
    keys.reserve(config.num_operations);
    
    std::mt19937_64 gen(42);
    
    // 80% of accesses are to the most recent 20% of data (append-heavy)
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    size_t recent_threshold = static_cast<size_t>(config.num_keys * 0.8);
    std::uniform_int_distribution<uint64_t> recent_dist(recent_threshold, config.num_keys - 1);
    std::uniform_int_distribution<uint64_t> old_dist(0, recent_threshold - 1);
    
    for (size_t i = 0; i < config.num_operations; ++i) {
        if (prob_dist(gen) < 0.8) {
            // Access recent data
            keys.push_back(recent_dist(gen));
        } else {
            // Access older data
            keys.push_back(old_dist(gen));
        }
    }
    
    return keys;
}

// AnalyticsWorkload implementation
std::vector<uint64_t> AnalyticsWorkload::GenerateKeys(const BenchmarkConfig& config) {
    std::vector<uint64_t> keys;
    keys.reserve(config.num_operations * config.range_size);
    
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<uint64_t> start_dist(0, 
        config.num_keys - config.range_size);
    
    // Generate range scan starting points
    for (size_t i = 0; i < config.num_operations; ++i) {
        uint64_t start_key = start_dist(gen);
        
        // Add all keys in the range
        for (size_t j = 0; j < config.range_size; ++j) {
            keys.push_back(start_key + j);
        }
    }
    
    return keys;
}

// YCSBUtils implementation
std::vector<uint64_t> YCSBUtils::GenerateZipfian(size_t num_values, size_t max_key, 
                                                 double alpha, uint64_t seed) {
    std::vector<uint64_t> keys;
    keys.reserve(num_values);
    
    // Pre-compute Zipfian probabilities
    std::vector<double> probabilities(max_key);
    double sum = 0.0;
    
    for (size_t i = 0; i < max_key; ++i) {
        probabilities[i] = 1.0 / std::pow(i + 1, alpha);
        sum += probabilities[i];
    }
    
    // Normalize probabilities
    for (double& prob : probabilities) {
        prob /= sum;
    }
    
    // Create cumulative distribution
    std::vector<double> cumulative(max_key);
    std::partial_sum(probabilities.begin(), probabilities.end(), cumulative.begin());
    
    // Generate keys
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < num_values; ++i) {
        double rand_val = dist(gen);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rand_val);
        size_t key_index = std::distance(cumulative.begin(), it);
        keys.push_back(key_index);
    }
    
    return keys;
}

std::vector<uint64_t> YCSBUtils::GenerateLatest(size_t num_values, size_t max_key, 
                                               uint64_t seed) {
    std::vector<uint64_t> keys;
    keys.reserve(num_values);
    
    std::mt19937_64 gen(seed);
    
    // Latest distribution: exponentially decaying probability from most recent
    // P(key) = exp(-lambda * (max_key - key))
    double lambda = 0.01; // Decay rate
    
    std::vector<double> probabilities(max_key);
    double sum = 0.0;
    
    for (size_t i = 0; i < max_key; ++i) {
        probabilities[i] = std::exp(-lambda * (max_key - 1 - i));
        sum += probabilities[i];
    }
    
    // Normalize and create cumulative distribution
    std::vector<double> cumulative(max_key);
    double running_sum = 0.0;
    for (size_t i = 0; i < max_key; ++i) {
        running_sum += probabilities[i] / sum;
        cumulative[i] = running_sum;
    }
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < num_values; ++i) {
        double rand_val = dist(gen);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), rand_val);
        size_t key_index = std::distance(cumulative.begin(), it);
        keys.push_back(key_index);
    }
    
    return keys;
}

std::vector<uint64_t> YCSBUtils::GenerateUniform(size_t num_values, size_t max_key, 
                                                 uint64_t seed) {
    std::vector<uint64_t> keys;
    keys.reserve(num_values);
    
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint64_t> dist(0, max_key - 1);
    
    for (size_t i = 0; i < num_values; ++i) {
        keys.push_back(dist(gen));
    }
    
    return keys;
}

std::vector<uint64_t> YCSBUtils::GenerateHotspot(size_t num_values, size_t max_key, 
                                                 double hot_fraction, 
                                                 double hot_probability,
                                                 uint64_t seed) {
    std::vector<uint64_t> keys;
    keys.reserve(num_values);
    
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    // Hot data region
    size_t hot_size = static_cast<size_t>(max_key * hot_fraction);
    size_t hot_start = max_key - hot_size; // Hot data at the end
    
    std::uniform_int_distribution<uint64_t> hot_dist(hot_start, max_key - 1);
    std::uniform_int_distribution<uint64_t> cold_dist(0, hot_start - 1);
    
    for (size_t i = 0; i < num_values; ++i) {
        if (prob_dist(gen) < hot_probability) {
            keys.push_back(hot_dist(gen));
        } else {
            keys.push_back(cold_dist(gen));
        }
    }
    
    return keys;
}

std::vector<uint64_t> YCSBUtils::GenerateTemporal(size_t num_values, size_t max_key,
                                                  double locality_strength,
                                                  uint64_t seed) {
    std::vector<uint64_t> keys;
    keys.reserve(num_values);
    
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> temporal_dist(0.0, 1.0);
    std::uniform_int_distribution<uint64_t> uniform_dist(0, max_key - 1);
    
    // Simulate temporal locality with moving window
    double current_time = 0.0;
    double window_size = max_key * 0.1; // 10% of keyspace
    
    for (size_t i = 0; i < num_values; ++i) {
        if (i % 1000 == 0) {
            // Advance time window periodically
            current_time += window_size * 0.1;
        }
        
        if (temporal_dist(gen) < locality_strength) {
            // Generate key within temporal window
            uint64_t center = static_cast<uint64_t>(current_time) % max_key;
            uint64_t offset = static_cast<uint64_t>(
                std::abs(temporal_dist(gen) * window_size / 4.0));
            uint64_t key = (center + offset) % max_key;
            keys.push_back(key);
        } else {
            // Random access
            keys.push_back(uniform_dist(gen));
        }
    }
    
    return keys;
}

// YCSBConfigFactory implementation
BenchmarkConfig YCSBConfigFactory::CreateWorkloadA(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::MIXED_WORKLOAD;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.5;
    config.write_ratio = 0.5;
    config.key_size = 10;
    config.value_size = 100;
    config.learned_index_options.confidence_threshold = 0.8;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateWorkloadB(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::READ_HEAVY;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.95;
    config.write_ratio = 0.05;
    config.key_size = 10;
    config.value_size = 100;
    config.learned_index_options.confidence_threshold = 0.85;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateWorkloadC(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::RANDOM_READ;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 1.0;
    config.write_ratio = 0.0;
    config.key_size = 10;
    config.value_size = 100;
    config.learned_index_options.confidence_threshold = 0.9;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateWorkloadD(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::READ_HEAVY;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.95;
    config.write_ratio = 0.05;
    config.key_size = 10;
    config.value_size = 100;
    config.learned_index_options.confidence_threshold = 0.8;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateWorkloadE(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::RANGE_QUERY;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.95;
    config.write_ratio = 0.05;
    config.range_size = 100; // Short ranges
    config.key_size = 10;
    config.value_size = 100;
    config.learned_index_options.confidence_threshold = 0.8;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateWorkloadF(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::MIXED_WORKLOAD;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.5;
    config.write_ratio = 0.5; // Read-modify-write
    config.key_size = 10;
    config.value_size = 100;
    config.learned_index_options.confidence_threshold = 0.8;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateTimeSeriesWorkload(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::SEQUENTIAL_READ;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.8;
    config.write_ratio = 0.2;
    config.key_size = 16; // Timestamp-like keys
    config.value_size = 200; // Larger time-series values
    config.learned_index_options.confidence_threshold = 0.9; // High confidence for temporal patterns
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateLogStructuredWorkload(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::MIXED_WORKLOAD;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 0.7;
    config.write_ratio = 0.3; // Append-heavy
    config.key_size = 12;
    config.value_size = 150;
    config.learned_index_options.confidence_threshold = 0.85;
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

BenchmarkConfig YCSBConfigFactory::CreateAnalyticsWorkload(size_t num_keys, size_t num_operations) {
    BenchmarkConfig config;
    config.workload_type = WorkloadType::RANGE_QUERY;
    config.num_keys = num_keys;
    config.num_operations = num_operations;
    config.read_ratio = 1.0;
    config.write_ratio = 0.0;
    config.range_size = 10000; // Large range scans
    config.key_size = 8;
    config.value_size = 500; // Larger analytical data
    config.learned_index_options.confidence_threshold = 0.95; // Very high confidence for analytics
    config.learned_index_options.model_type = ModelType::LINEAR;
    return config;
}

} // namespace ycsb
} // namespace benchmark
} // namespace learned_index
} // namespace rocksdb