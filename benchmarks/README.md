# Learned Index Benchmark Suite

A comprehensive benchmark suite comparing learned index performance against traditional indexing methods (B+ Tree, Sorted Array, Hash Table) for RocksDB SST files.

## Overview

This benchmark suite evaluates:
- **Lookup latency** across different workload patterns
- **Memory usage** for various index implementations  
- **Throughput** performance comparison
- **Prediction accuracy** for learned indexes
- **Scalability** with different dataset sizes

## Benchmark Results Summary

### Key Findings
- **Learned Index**: 0.132μs avg latency, 5.82 MQPS, 93.6% accuracy
- **B+ Tree**: 0.055μs avg latency, 10.87 MQPS, 100% accuracy (fastest)
- **Hash Table**: 0.037μs avg latency, 13.31 MQPS, 100% accuracy (best throughput)
- **Sorted Array**: 0.074μs avg latency, 8.78 MQPS, 100% accuracy

### Workload Performance
- **Sequential**: Learned index achieves 88.9% accuracy
- **Random**: More challenging, 94.5% accuracy with 5.5% fallback
- **Mixed**: Excellent 96.3% accuracy, 3.7% fallback rate
- **Zipfian**: Best performance, 97.2% accuracy (skewed data)
- **Temporal**: Optimal for time-series, 97.8% accuracy

## Building and Running

### Prerequisites
- C++17 compatible compiler
- Python 3 with matplotlib (for chart generation)

### Quick Start
```bash
# Build benchmark
make benchmark

# Run comprehensive benchmark
make run-comprehensive

# Run scalability tests
make run-scalability

# Generate visualization charts
make charts
```

### Manual Execution
```bash
# Build
make benchmark

# Run specific benchmark types
./benchmark --comprehensive
./benchmark --scalability
./benchmark --help

# Generate charts (requires Python)
cd results
python3 latency_comparison.py
python3 throughput_comparison.py
python3 memory_comparison.py
python3 accuracy_comparison.py
```

## Workload Types

### Sequential
- Keys generated in ascending order
- Simulates sequential SST file scans
- **Best for**: Time-series data, log files

### Random
- Uniformly random key distribution
- Most challenging for learned indexes
- **Best for**: Testing worst-case scenarios

### Mixed
- Configurable ratio of sequential vs random (default 80/20)
- Realistic production workload simulation
- **Best for**: General-purpose applications

### Zipfian
- Skewed access pattern following Zipfian distribution
- Hot keys accessed more frequently
- **Best for**: Social media, web applications

### Temporal
- Time-based key generation with controlled jitter
- Simulates timestamp-based data
- **Best for**: Monitoring data, event logs

## Index Implementations

### LearnedIndex
- Linear regression model with block-level predictions
- Confidence-based fallback to traditional methods
- ~157KB memory usage per SST file
- 88-98% prediction accuracy depending on workload

### B+Tree
- Traditional tree-based index
- Optimal for range queries and sorted access
- Minimal memory usage (~1KB)
- Always 100% accurate

### SortedArray
- Binary search on sorted key-value pairs
- Simple but effective for read-only scenarios
- Moderate memory usage (~156KB)
- Always 100% accurate

### HashTable
- Hash-based O(1) lookup
- Best raw throughput performance
- Higher memory usage (~292KB average)
- Always 100% accurate for existing keys

## Output Files

### results/benchmark_results.csv
Complete benchmark data in CSV format with columns:
- `index_type`: Type of index tested
- `workload_type`: Workload pattern used
- `avg_latency_us`: Average lookup latency in microseconds
- `throughput_qps`: Queries per second
- `index_memory_bytes`: Memory usage in bytes
- `prediction_accuracy`: Prediction accuracy (learned index only)

### Visualization Charts
- `latency_comparison.py`: Latency comparison across workloads
- `throughput_comparison.py`: Throughput performance comparison  
- `memory_comparison.py`: Memory usage analysis
- `accuracy_comparison.py`: Learned index prediction accuracy

### Analysis Report
- `benchmark_analysis.md`: Comprehensive performance analysis and recommendations

## Configuration

### Workload Parameters
Modify `main_benchmark.cpp` to adjust:
- Dataset sizes (default: 10,000 records)
- Query counts (default: 5,000 queries)
- Key ranges and distributions
- Sequential/random ratios for mixed workloads

### Index Options
Modify learned index parameters in `learned_index_adapter.cpp`:
- Model types (Linear, Polynomial, Neural Network)
- Confidence thresholds
- Cache sizes
- Fallback behavior

## Performance Insights

### When to Use Learned Index
✅ **Recommended for:**
- Time-series and temporal data (97.8% accuracy)
- Zipfian distributions (97.2% accuracy)
- Large datasets where memory overhead is acceptable
- Read-heavy workloads with predictable patterns

❌ **Not recommended for:**
- Pure random access patterns
- Memory-constrained environments  
- Latency-critical applications requiring sub-50μs response
- Write-heavy workloads

### Production Considerations
1. **Hybrid Approach**: Use learned index with B+ tree fallback
2. **Adaptive Selection**: Choose index type based on detected patterns
3. **Memory Budget**: Learned index uses ~157x more memory than B+ trees
4. **Training Overhead**: Consider model training time during SST creation

## Extending the Benchmark

### Adding New Index Types
1. Implement `IndexInterface` in new header/source files
2. Add to `main_benchmark.cpp` in the `AddIndex` calls
3. Rebuild and run benchmarks

### Adding New Workload Types
1. Add new `WorkloadType` enum value
2. Implement generation logic in `WorkloadGenerator`
3. Add configuration in `main_benchmark.cpp`

### Custom Metrics
1. Extend `BenchmarkResult` structure
2. Update CSV output format in `SaveResults`
3. Modify chart generation scripts

## Dependencies

### Runtime
- C++17 standard library
- Learned index implementation (included)

### Chart Generation
- Python 3
- matplotlib: `pip install matplotlib`
- numpy: `pip install numpy`

## Performance Baseline

Benchmark environment: Linux x86_64, g++ -O2
Results will vary based on:
- CPU architecture and speed
- Memory hierarchy (L1/L2/L3 cache sizes)
- Compiler optimizations
- System load and other processes

For consistent results, run benchmarks on dedicated systems with minimal background processes.