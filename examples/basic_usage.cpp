#include "learned_index/sst_learned_index_manager.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace rocksdb::learned_index;

void RunBasicExample() {
    std::cout << "=== Learned Index Basic Usage Example ===" << std::endl;
    
    // Configure the learned index manager
    SSTLearnedIndexOptions options;
    options.enable_learned_index = true;
    options.default_model_type = ModelType::kLinear;
    options.confidence_threshold = 0.8;
    options.cache_models = true;
    options.max_cache_size = 100;
    
    SSTLearnedIndexManager manager(options);
    
    // Simulate training data from an SST file
    // This represents key-to-block mappings extracted from an SST file
    std::vector<std::pair<uint64_t, uint32_t>> training_data;
    
    // Block 0: keys 1000-1999
    for (uint64_t key = 1000; key < 2000; key += 100) {
        training_data.emplace_back(key, 0);
    }
    
    // Block 1: keys 2000-2999  
    for (uint64_t key = 2000; key < 3000; key += 100) {
        training_data.emplace_back(key, 1);
    }
    
    // Block 2: keys 3000-3999
    for (uint64_t key = 3000; key < 4000; key += 100) {
        training_data.emplace_back(key, 2);
    }
    
    std::string sst_file_path = "example.sst";
    
    std::cout << "Training learned index with " << training_data.size() << " samples..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool training_success = manager.TrainModel(sst_file_path, training_data);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (training_success) {
        std::cout << "Training completed in " << duration.count() << " microseconds" << std::endl;
    } else {
        std::cout << "Training failed!" << std::endl;
        return;
    }
    
    // Test predictions
    std::cout << "\nTesting predictions:" << std::endl;
    
    struct TestCase {
        uint64_t key;
        uint32_t expected_block;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {1500, 0, "Key 1500 (should be in block 0)"},
        {2500, 1, "Key 2500 (should be in block 1)"},
        {3500, 2, "Key 3500 (should be in block 2)"},
        {500,  0, "Key 500 (outside training range, should fallback)"},
        {5000, 2, "Key 5000 (outside training range, should fallback)"}
    };
    
    int correct_predictions = 0;
    
    for (const auto& test_case : test_cases) {
        uint32_t predicted_block = manager.PredictBlockIndex(sst_file_path, test_case.key);
        double confidence = manager.GetPredictionConfidence(sst_file_path, test_case.key);
        
        bool is_correct = (predicted_block == test_case.expected_block);
        if (is_correct) correct_predictions++;
        
        std::cout << "  " << test_case.description << std::endl;
        std::cout << "    Predicted: " << predicted_block 
                  << ", Expected: " << test_case.expected_block 
                  << ", Confidence: " << confidence
                  << " [" << (is_correct ? "CORRECT" : "INCORRECT") << "]" << std::endl;
    }
    
    // Display statistics
    const auto& stats = manager.GetStats(sst_file_path);
    std::cout << "\nLearned Index Statistics:" << std::endl;
    std::cout << "  Total queries: " << stats.total_queries << std::endl;
    std::cout << "  Successful predictions: " << stats.successful_predictions << std::endl;
    std::cout << "  Fallback queries: " << stats.fallback_queries << std::endl;
    std::cout << "  Success rate: " << (stats.GetSuccessRate() * 100) << "%" << std::endl;
    std::cout << "  Fallback rate: " << (stats.GetFallbackRate() * 100) << "%" << std::endl;
    
    // Test model persistence
    std::cout << "\nTesting model save/load:" << std::endl;
    std::string serialized_model;
    if (manager.SaveLearnedIndex(sst_file_path, &serialized_model)) {
        std::cout << "  Model saved (size: " << serialized_model.size() << " bytes)" << std::endl;
        
        // Create new manager and load the model
        SSTLearnedIndexManager new_manager(options);
        if (new_manager.LoadLearnedIndex("loaded_" + sst_file_path, serialized_model)) {
            std::cout << "  Model loaded successfully" << std::endl;
            
            // Test that loaded model works
            uint32_t original_prediction = manager.PredictBlockIndex(sst_file_path, 1500);
            uint32_t loaded_prediction = new_manager.PredictBlockIndex("loaded_" + sst_file_path, 1500);
            
            if (original_prediction == loaded_prediction) {
                std::cout << "  Loaded model produces same predictions ✓" << std::endl;
            } else {
                std::cout << "  Loaded model predictions differ ✗" << std::endl;
            }
        } else {
            std::cout << "  Model loading failed!" << std::endl;
        }
    } else {
        std::cout << "  Model saving failed!" << std::endl;
    }
    
    std::cout << "\n=== Example completed ===" << std::endl;
}

int main() {
    try {
        RunBasicExample();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}