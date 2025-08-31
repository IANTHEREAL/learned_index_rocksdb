#include <iostream>
#include <vector>
#include <functional>

struct TestCase {
    std::string name;
    std::function<bool()> test_func;
};

extern std::vector<TestCase> GetLearnedIndexBlockTests();
extern std::vector<TestCase> GetSSTLearnedIndexManagerTests();

int main() {
    std::vector<TestCase> all_tests;
    
    // Collect all test cases
    auto block_tests = GetLearnedIndexBlockTests();
    auto manager_tests = GetSSTLearnedIndexManagerTests();
    
    all_tests.insert(all_tests.end(), block_tests.begin(), block_tests.end());
    all_tests.insert(all_tests.end(), manager_tests.begin(), manager_tests.end());
    
    int passed = 0;
    int failed = 0;
    
    std::cout << "Running " << all_tests.size() << " tests...\n" << std::endl;
    
    for (const auto& test : all_tests) {
        std::cout << "Running test: " << test.name << " ... ";
        try {
            if (test.test_func()) {
                std::cout << "PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "FAILED" << std::endl;
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (exception: " << e.what() << ")" << std::endl;
            failed++;
        }
    }
    
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "  Passed: " << passed << std::endl;
    std::cout << "  Failed: " << failed << std::endl;
    std::cout << "  Total:  " << all_tests.size() << std::endl;
    
    return failed > 0 ? 1 : 0;
}