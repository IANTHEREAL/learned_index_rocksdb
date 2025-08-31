CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -Iinclude -Iinclude/mock

# Source files
LIB_SOURCES = src/learned_index/learned_index_block.cpp \
              src/learned_index/sst_learned_index_manager.cpp

TEST_SOURCES = test/learned_index/learned_index_block_test.cpp \
               test/learned_index/sst_learned_index_manager_test.cpp \
               test/learned_index/test_main.cpp

# Object files
LIB_OBJECTS = $(LIB_SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# Targets
all: liblearned_index.a learned_index_test

liblearned_index.a: $(LIB_OBJECTS)
	ar rcs $@ $^

learned_index_test: $(LIB_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: learned_index_test
	./learned_index_test

# Clean target
clean:
	rm -f $(LIB_OBJECTS) $(TEST_OBJECTS) liblearned_index.a learned_index_test

.PHONY: all test clean