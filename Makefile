CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g -pthread
INCLUDES = -Iinclude -Iinclude/mock

# Source files
LIB_SOURCES = src/learned_index/learned_index_block.cpp \
              src/learned_index/sst_learned_index_manager.cpp

ADAPTIVE_SOURCES = src/learned_index/adaptive/model_performance_tracker.cpp \
                  src/learned_index/adaptive/adaptive_retraining_manager.cpp \
                  src/learned_index/adaptive_sst_manager.cpp

TEST_SOURCES = test/learned_index/learned_index_block_test.cpp \
               test/learned_index/sst_learned_index_manager_test.cpp \
               test/learned_index/test_main.cpp

# Object files
LIB_OBJECTS = $(LIB_SOURCES:.cpp=.o)
ADAPTIVE_OBJECTS = $(ADAPTIVE_SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# Main targets
all: liblearned_index.a learned_index_test

# Core library
liblearned_index.a: $(LIB_OBJECTS)
	ar rcs $@ $^

# Adaptive functionality
adaptive: liblearned_index_adaptive.a

liblearned_index_adaptive.a: $(LIB_OBJECTS) $(ADAPTIVE_OBJECTS)
	ar rcs $@ $^

# Tests
learned_index_test: $(LIB_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Examples
examples: basic_usage adaptive_demo

basic_usage: $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o examples/basic_usage examples/basic_usage.cpp $^

adaptive_demo: liblearned_index_adaptive.a
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o examples/adaptive_demo examples/adaptive_retraining_demo.cpp liblearned_index_adaptive.a

# Dashboard setup
dashboard-setup:
	cd dashboard && pip install -r requirements.txt

dashboard-start: dashboard-setup
	cd dashboard && python3 dashboard_server.py

# Complete adaptive build
all-adaptive: adaptive examples dashboard-setup

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: learned_index_test
	./learned_index_test

# Clean targets
clean:
	rm -f $(LIB_OBJECTS) $(ADAPTIVE_OBJECTS) $(TEST_OBJECTS) 
	rm -f liblearned_index.a liblearned_index_adaptive.a learned_index_test
	rm -f examples/basic_usage examples/adaptive_demo

clean-dashboard:
	cd dashboard && rm -f *.db *.log

help:
	@echo "Learned Index Makefile"
	@echo "======================"
	@echo ""
	@echo "Core targets:"
	@echo "  all                 Build core library and tests"
	@echo "  liblearned_index.a  Build core learned index library"
	@echo "  test               Run unit tests"
	@echo ""
	@echo "Adaptive functionality:"
	@echo "  adaptive           Build adaptive retraining library"
	@echo "  examples           Build example applications"
	@echo "  adaptive_demo      Build and run adaptive retraining demo"
	@echo ""
	@echo "Dashboard:"
	@echo "  dashboard-setup    Install Python dashboard dependencies"
	@echo "  dashboard-start    Start performance dashboard server"
	@echo ""
	@echo "Convenience targets:"
	@echo "  all-adaptive       Build everything including adaptive features"
	@echo "  clean              Clean all build artifacts"
	@echo "  clean-dashboard    Clean dashboard database files"
	@echo "  help               Show this help message"

.PHONY: all test clean adaptive examples dashboard-setup dashboard-start all-adaptive clean-dashboard help