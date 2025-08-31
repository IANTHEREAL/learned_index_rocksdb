#include "learned_index/sst_learned_index_manager.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace learned_index;

int main() {
    std::cout << "Learned Index RocksDB - Basic Usage Example\n";
    std::cout << "==========================================\n\n";
    
    // Configure learned index options
    SSTLearnedIndexOptions options;
    options.enabled = true;
    options.cache_models = true;
    options.max_cache_size = 100;
    options.confidence_threshold = 0.8;
    options.preferred_model_type = ModelType::LINEAR;
    
    std::cout << "Configuration:\n";
    std::cout << "  Cache enabled: " << (options.cache_models ? "Yes" : "No") << "\n";
    std::cout << "  Max cache size: " << options.max_cache_size << "\n";
    std::cout << "  Confidence threshold: " << options.confidence_threshold << "\n";
    std::cout << "  Model type: Linear\n\n";
    
    // Create manager
    SSTLearnedIndexManager manager(options);
    
    // Generate sample training data
    std::cout << "Generating training data...\n";
    std::vector<std::pair<uint64_t, uint32_t>> training_data;
    
    // Simulate key-block pairs from an SST file
    // Keys are sorted, blocks increase with keys
    for (uint64_t i = 0; i < 1000; ++i) {
        uint64_t key = i * 1000 + (i % 100);  // Keys: 0, 1001, 2002, etc.
        uint32_t block = static_cast<uint32_t>(i / 50);  // 50 keys per block
        training_data.emplace_back(key, block);
    }
    
    std::cout << "  Generated " << training_data.size() << " key-block pairs\n";
    std::cout << "  Key range: " << training_data.front().first 
              << " to " << training_data.back().first << "\n";
    std::cout << "  Block range: " << training_data.front().second 
              << " to " << training_data.back().second << "\n\n";
    
    // Create learned index for SST file
    std::string sst_file_path = "/tmp/example.sst";
    std::cout << "Creating learned index for SST file: " << sst_file_path << "\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = manager.CreateLearnedIndex(sst_file_path, training_data);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (success) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "  ✓ Learned index created successfully in " << duration.count() << " μs\n";
    } else {
        std::cout << "  ✗ Failed to create learned index\n";
        return 1;
    }
    
    std::cout << "  Cache size: " << manager.GetCacheSize() << "\n\n";
    
    // Test predictions
    std::cout << "Testing predictions...\n";
    std::vector<uint64_t> test_keys = {500, 50500, 100500, 250500, 500500, 750500, 999500};
    
    for (uint64_t key : test_keys) {
        int predicted_block = manager.PredictBlockIndex(sst_file_path, key);
        double confidence = manager.GetPredictionConfidence(sst_file_path, key);
        
        std::cout << "  Key " << key << ": ";
        if (predicted_block >= 0) {
            std::cout << "Block " << predicted_block 
                      << " (confidence: " << std::fixed << std::setprecision(2) << confidence << ")\n";
        } else {
            std::cout << "Prediction failed (low confidence)\n";
        }
    }
    
    // Test batch predictions
    std::cout << "\nTesting batch predictions...\n";
    std::vector<uint64_t> batch_keys;
    for (int i = 0; i < 10; ++i) {
        batch_keys.push_back(i * 100000);
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    auto batch_predictions = manager.BatchPredictBlockIndices(sst_file_path, batch_keys);
    end_time = std::chrono::high_resolution_clock::now();
    
    auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "  Batch prediction for " << batch_keys.size() 
              << " keys completed in " << batch_duration.count() << " μs\n";
    
    for (size_t i = 0; i < batch_keys.size(); ++i) {
        std::cout << "    Key " << batch_keys[i] << " -> Block " << batch_predictions[i] << "\n";
    }
    
    // Display statistics
    std::cout << "\nPerformance statistics:\n";
    auto stats = manager.GetStats(sst_file_path);
    std::cout << "  Total queries: " << stats.total_queries << "\n";
    std::cout << "  Successful predictions: " << stats.successful_predictions << "\n";
    std::cout << "  Fallback queries: " << stats.fallback_queries << "\n";
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(1) << stats.GetAccuracy() << "%\n";
    
    // Test model persistence
    std::cout << "\nTesting model persistence...\n";
    std::cout << "  Removing from cache...\n";
    manager.RemoveLearnedIndex(sst_file_path);
    std::cout << "  Cache size: " << manager.GetCacheSize() << "\n";
    
    std::cout << "  Reloading from disk...\n";
    if (manager.LoadLearnedIndex(sst_file_path)) {
        std::cout << "  ✓ Model successfully reloaded from disk\n";
        std::cout << "  Cache size: " << manager.GetCacheSize() << "\n";
        
        // Test prediction after reload
        int test_prediction = manager.PredictBlockIndex(sst_file_path, 250500);
        std::cout << "  Test prediction after reload: Block " << test_prediction << "\n";
    } else {
        std::cout << "  ✗ Failed to reload model from disk\n";
    }
    
    // Final statistics
    std::cout << "\nFinal aggregated statistics:\n";
    auto final_stats = manager.GetAggregatedStats();
    std::cout << "  Total queries: " << final_stats.total_queries << "\n";
    std::cout << "  Overall accuracy: " << std::fixed << std::setprecision(1) << final_stats.GetAccuracy() << "%\n";
    std::cout << "  Cache hit rate: " << std::fixed << std::setprecision(1) << final_stats.GetCacheHitRate() << "%\n";
    
    std::cout << "\nExample completed successfully!\n";
    return 0;
}