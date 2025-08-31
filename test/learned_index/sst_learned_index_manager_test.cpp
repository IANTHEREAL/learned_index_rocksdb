#include "learned_index/sst_learned_index_manager.h"
#include <vector>
#include <functional>
#include <string>

using namespace rocksdb::learned_index;

struct TestCase {
    std::string name;
    std::function<bool()> test_func;
};

bool TestSSTManagerConstruction() {
    SSTLearnedIndexOptions options;
    options.enable_learned_index = true;
    options.default_model_type = ModelType::kLinear;
    options.max_cache_size = 10;
    
    SSTLearnedIndexManager manager(options);
    
    // Test that options are set correctly
    const auto& stored_options = manager.GetOptions();
    if (!stored_options.enable_learned_index) return false;
    if (stored_options.default_model_type != ModelType::kLinear) return false;
    if (stored_options.max_cache_size != 10) return false;
    
    return true;
}

bool TestSSTManagerTraining() {
    SSTLearnedIndexOptions options;
    SSTLearnedIndexManager manager(options);
    
    // Create training data: keys mapped to block indices
    std::vector<std::pair<uint64_t, uint32_t>> training_data = {
        {100, 0}, {200, 0}, {300, 0},     // Block 0: keys 100-300
        {1100, 1}, {1200, 1}, {1300, 1}, // Block 1: keys 1100-1300
        {2100, 2}, {2200, 2}, {2300, 2}  // Block 2: keys 2100-2300
    };
    
    std::string sst_path = "test_file.sst";
    
    // Train model
    if (!manager.TrainModel(sst_path, training_data)) {
        return false;
    }
    
    // Test predictions
    uint32_t pred1 = manager.PredictBlockIndex(sst_path, 150);   // Should predict block 0
    uint32_t pred2 = manager.PredictBlockIndex(sst_path, 1150);  // Should predict block 1
    uint32_t pred3 = manager.PredictBlockIndex(sst_path, 2150);  // Should predict block 2
    
    // The predictions should be reasonable (within expected block ranges)
    if (pred1 > 2 || pred2 > 2 || pred3 > 2) {
        return false;
    }
    
    // Check statistics
    const auto& stats = manager.GetStats(sst_path);
    if (stats.total_queries < 3) return false;
    
    return true;
}

bool TestSSTManagerCaching() {
    SSTLearnedIndexOptions options;
    options.cache_models = true;
    options.max_cache_size = 2;
    
    SSTLearnedIndexManager manager(options);
    
    // Create minimal training data
    std::vector<std::pair<uint64_t, uint32_t>> training_data = {
        {100, 0}, {200, 1}
    };
    
    // Train models for multiple files
    std::string file1 = "file1.sst";
    std::string file2 = "file2.sst";
    std::string file3 = "file3.sst";
    
    manager.TrainModel(file1, training_data);
    manager.TrainModel(file2, training_data);
    
    // Both should be cached
    if (!manager.GetCachedModel(file1)) return false;
    if (!manager.GetCachedModel(file2)) return false;
    
    // Add third file - should trigger LRU eviction
    manager.TrainModel(file3, training_data);
    
    // file1 should be evicted (LRU), file2 and file3 should remain
    if (manager.GetCachedModel(file1)) return false; // Should be evicted
    if (!manager.GetCachedModel(file2)) return false;
    if (!manager.GetCachedModel(file3)) return false;
    
    return true;
}

bool TestSSTManagerStatistics() {
    SSTLearnedIndexOptions options;
    SSTLearnedIndexManager manager(options);
    
    std::string sst_path = "stats_test.sst";
    
    // Train a simple model
    std::vector<std::pair<uint64_t, uint32_t>> training_data = {
        {100, 0}, {200, 1}
    };
    
    manager.TrainModel(sst_path, training_data);
    
    // Make some predictions to generate statistics
    manager.PredictBlockIndex(sst_path, 150);
    manager.PredictBlockIndex(sst_path, 250);
    
    const auto& stats = manager.GetStats(sst_path);
    
    // Check basic statistics
    if (stats.total_queries == 0) return false;
    if (stats.GetSuccessRate() < 0.0 || stats.GetSuccessRate() > 1.0) return false;
    if (stats.GetFallbackRate() < 0.0 || stats.GetFallbackRate() > 1.0) return false;
    
    // Update statistics manually
    manager.UpdateStats(sst_path, true, 10.0);
    const auto& updated_stats = manager.GetStats(sst_path);
    
    if (updated_stats.total_queries <= stats.total_queries) return false;
    if (updated_stats.successful_predictions <= stats.successful_predictions) return false;
    
    return true;
}

bool TestSSTManagerSaveLoad() {
    SSTLearnedIndexOptions options;
    SSTLearnedIndexManager manager(options);
    
    std::string sst_path = "saveload_test.sst";
    
    // Train a model
    std::vector<std::pair<uint64_t, uint32_t>> training_data = {
        {100, 0}, {200, 1}, {300, 2}
    };
    
    if (!manager.TrainModel(sst_path, training_data)) {
        return false;
    }
    
    // Save the model
    std::string saved_data;
    if (!manager.SaveLearnedIndex(sst_path, &saved_data)) {
        return false;
    }
    
    if (saved_data.empty()) return false;
    
    // Create a new manager and load the model
    SSTLearnedIndexManager new_manager(options);
    if (!new_manager.LoadLearnedIndex(sst_path, saved_data)) {
        return false;
    }
    
    // Test that the loaded model works
    uint32_t prediction = new_manager.PredictBlockIndex(sst_path, 150);
    if (prediction > 2) return false; // Should be a reasonable prediction
    
    return true;
}

bool TestSSTManagerOptionsUpdate() {
    SSTLearnedIndexOptions initial_options;
    initial_options.cache_models = true;
    initial_options.max_cache_size = 5;
    
    SSTLearnedIndexManager manager(initial_options);
    
    // Train some models to populate cache
    std::vector<std::pair<uint64_t, uint32_t>> training_data = {{100, 0}};
    manager.TrainModel("file1.sst", training_data);
    manager.TrainModel("file2.sst", training_data);
    
    // Update options to disable caching
    SSTLearnedIndexOptions new_options = initial_options;
    new_options.cache_models = false;
    
    manager.UpdateOptions(new_options);
    
    // Cache should be cleared
    if (manager.GetCachedModel("file1.sst")) return false;
    if (manager.GetCachedModel("file2.sst")) return false;
    
    return true;
}

std::vector<TestCase> GetSSTLearnedIndexManagerTests() {
    return {
        {"SSTManager Construction", TestSSTManagerConstruction},
        {"SSTManager Training", TestSSTManagerTraining},
        {"SSTManager Caching", TestSSTManagerCaching},
        {"SSTManager Statistics", TestSSTManagerStatistics},
        {"SSTManager Save/Load", TestSSTManagerSaveLoad},
        {"SSTManager Options Update", TestSSTManagerOptionsUpdate}
    };
}