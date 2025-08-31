CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -I../include -I../include/mock -Iinclude

# Source files
BENCHMARK_SOURCES = src/benchmark_framework.cpp \
                   src/btree_index.cpp \
                   src/learned_index_adapter.cpp \
                   src/main_benchmark.cpp

LEARNED_INDEX_SOURCES = ../src/learned_index/learned_index_block.cpp \
                       ../src/learned_index/sst_learned_index_manager.cpp

# Object files
BENCHMARK_OBJECTS = $(BENCHMARK_SOURCES:.cpp=.o)
LEARNED_INDEX_OBJECTS = $(LEARNED_INDEX_SOURCES:.cpp=.o)

# Main target
benchmark: $(BENCHMARK_OBJECTS) $(LEARNED_INDEX_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create results directory
results:
	mkdir -p results data

# Run comprehensive benchmark
run-comprehensive: benchmark results
	./benchmark --comprehensive

# Run scalability benchmark  
run-scalability: benchmark results
	./benchmark --scalability

# Run both benchmarks
run-all: run-comprehensive run-scalability

# Generate charts (requires Python and matplotlib)
charts: results/benchmark_results.csv
	cd results && python3 latency_comparison.py
	cd results && python3 memory_comparison.py
	cd results && python3 throughput_comparison.py
	cd results && python3 accuracy_comparison.py

# Install Python dependencies for charting
install-deps:
	pip3 install matplotlib numpy

# Clean
clean:
	rm -f $(BENCHMARK_OBJECTS) $(LEARNED_INDEX_OBJECTS) benchmark
	rm -rf results data

# Help target
help:
	@echo "Benchmark Suite Makefile"
	@echo "========================"
	@echo ""
	@echo "Targets:"
	@echo "  benchmark           Build the benchmark executable"
	@echo "  run-comprehensive   Run comprehensive benchmark with all workload types"
	@echo "  run-scalability     Run scalability benchmark with different dataset sizes"
	@echo "  run-all             Run both comprehensive and scalability benchmarks"
	@echo "  charts              Generate performance charts (requires Python/matplotlib)"
	@echo "  install-deps        Install Python dependencies for charting"
	@echo "  clean              Clean build artifacts and results"
	@echo "  help               Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make benchmark      # Build"
	@echo "  make run-all        # Run all benchmarks"
	@echo "  make charts         # Generate visualization charts"

.PHONY: benchmark run-comprehensive run-scalability run-all charts install-deps clean help results