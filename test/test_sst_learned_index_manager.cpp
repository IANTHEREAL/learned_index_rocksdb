#include <gtest/gtest.h>
#include "learned_index/sst_learned_index_manager.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <random>

using namespace learned_index;

class SSTLearnedIndexManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for test files
        test_dir = std::filesystem::temp_directory_path() / "learned_index_test";
        std::filesystem::create_directories(test_dir);
        
        // Setup default options
        options.enabled = true;
        options.cache_models = true;
        options.max_cache_size = 10;
        options.confidence_threshold = 0.8;
        
        manager = std::make_unique<SSTLearnedIndexManager>(options);
    }

    void TearDown() override {
        manager.reset();
        
        // Clean up test directory
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }

    std::filesystem::path test_dir;
    SSTLearnedIndexOptions options;
    std::unique_ptr<SSTLearnedIndexManager> manager;
    
    // Helper function to generate test data
    std::vector<std::pair<uint64_t, uint32_t>> GenerateTrainingData(size_t count) {
        std::vector<std::pair<uint64_t, uint32_t>> data;
        data.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t key = i * 100;  // Keys: 0, 100, 200, 300, ...
            uint32_t block = static_cast<uint32_t>(i / 10);  // 10 keys per block
            data.emplace_back(key, block);
        }
        
        return data;
    }
    
    std::string GetTestFilePath(const std::string& filename) {
        return (test_dir / filename).string();
    }
};

TEST_F(SSTLearnedIndexManagerTest, ConstructorAndDefaults) {
    // Test default constructor
    SSTLearnedIndexManager default_manager;
    EXPECT_EQ(default_manager.GetCacheSize(), 0);
    
    // Test with custom options
    EXPECT_EQ(manager->GetCacheSize(), 0);
}

TEST_F(SSTLearnedIndexManagerTest, CreateAndLoadLearnedIndex) {
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    // Create learned index
    EXPECT_TRUE(manager->CreateLearnedIndex(sst_file, training_data));
    EXPECT_TRUE(manager->HasLearnedIndex(sst_file));
    EXPECT_EQ(manager->GetCacheSize(), 1);
    
    // Test loading existing index
    EXPECT_TRUE(manager->LoadLearnedIndex(sst_file));
    EXPECT_EQ(manager->GetCacheSize(), 1); // Should not increase cache size
}

TEST_F(SSTLearnedIndexManagerTest, BlockPrediction) {
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    EXPECT_TRUE(manager->CreateLearnedIndex(sst_file, training_data));
    
    // Test prediction for different keys
    for (uint64_t key = 0; key < 1000; key += 50) {
        int predicted_block = manager->PredictBlockIndex(sst_file, key);
        
        if (predicted_block >= 0) {
            // If prediction is valid, it should be reasonable
            EXPECT_GE(predicted_block, 0);
            EXPECT_LT(predicted_block, 10); // We have 10 blocks in training data
            
            // Check confidence
            double confidence = manager->GetPredictionConfidence(sst_file, key);
            EXPECT_GE(confidence, 0.0);
            EXPECT_LE(confidence, 1.0);
        }
    }
}

TEST_F(SSTLearnedIndexManagerTest, BatchPrediction) {
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    EXPECT_TRUE(manager->CreateLearnedIndex(sst_file, training_data));
    
    // Prepare batch of keys
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 20; ++i) {
        keys.push_back(i * 50);
    }
    
    // Test batch prediction
    auto predictions = manager->BatchPredictBlockIndices(sst_file, keys);
    EXPECT_EQ(predictions.size(), keys.size());
    
    // Verify individual predictions match batch predictions
    for (size_t i = 0; i < keys.size(); ++i) {
        int individual_prediction = manager->PredictBlockIndex(sst_file, keys[i]);
        if (individual_prediction >= 0) {
            EXPECT_EQ(predictions[i], individual_prediction);
        }
    }
}

TEST_F(SSTLearnedIndexManagerTest, CacheManagement) {
    // Set small cache size for testing
    options.max_cache_size = 3;
    manager = std::make_unique<SSTLearnedIndexManager>(options);
    
    auto training_data = GenerateTrainingData(50);
    
    // Create multiple learned indexes
    std::vector<std::string> files;
    for (int i = 0; i < 5; ++i) {
        std::string filename = GetTestFilePath("test" + std::to_string(i) + ".sst");
        files.push_back(filename);
        EXPECT_TRUE(manager->CreateLearnedIndex(filename, training_data));
    }
    
    // Cache should not exceed max size
    EXPECT_LE(manager->GetCacheSize(), options.max_cache_size);
    EXPECT_GT(manager->GetCacheSize(), 0);
}

TEST_F(SSTLearnedIndexManagerTest, StatisticsTracking) {
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    EXPECT_TRUE(manager->CreateLearnedIndex(sst_file, training_data));
    
    // Perform some predictions
    for (int i = 0; i < 10; ++i) {
        manager->PredictBlockIndex(sst_file, i * 100);
    }
    
    // Check statistics
    auto stats = manager->GetStats(sst_file);
    EXPECT_GT(stats.total_queries, 0);
    
    // Check aggregated statistics
    auto aggregated_stats = manager->GetAggregatedStats();
    EXPECT_GT(aggregated_stats.total_queries, 0);
    EXPECT_GT(aggregated_stats.last_update_timestamp, 0);
}

TEST_F(SSTLearnedIndexManagerTest, ConfigurationUpdate) {
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    EXPECT_TRUE(manager->CreateLearnedIndex(sst_file, training_data));
    
    // Update options
    SSTLearnedIndexOptions new_options = options;
    new_options.confidence_threshold = 0.95;
    new_options.max_cache_size = 1;
    
    manager->UpdateOptions(new_options);
    
    // Cache should be reduced to new size
    EXPECT_LE(manager->GetCacheSize(), new_options.max_cache_size);
}

TEST_F(SSTLearnedIndexManagerTest, RemoveLearnedIndex) {
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    EXPECT_TRUE(manager->CreateLearnedIndex(sst_file, training_data));
    EXPECT_TRUE(manager->HasLearnedIndex(sst_file));
    EXPECT_EQ(manager->GetCacheSize(), 1);
    
    // Remove the learned index
    manager->RemoveLearnedIndex(sst_file);
    EXPECT_FALSE(manager->HasLearnedIndex(sst_file));
    EXPECT_EQ(manager->GetCacheSize(), 0);
    
    // Statistics should also be cleared
    auto stats = manager->GetStats(sst_file);
    EXPECT_EQ(stats.total_queries, 0);
}

TEST_F(SSTLearnedIndexManagerTest, ClearCache) {
    auto training_data = GenerateTrainingData(50);
    
    // Create multiple indexes
    for (int i = 0; i < 3; ++i) {
        std::string filename = GetTestFilePath("test" + std::to_string(i) + ".sst");
        EXPECT_TRUE(manager->CreateLearnedIndex(filename, training_data));
    }
    
    EXPECT_GT(manager->GetCacheSize(), 0);
    
    // Clear cache
    manager->ClearCache();
    EXPECT_EQ(manager->GetCacheSize(), 0);
    
    // All statistics should be cleared
    auto aggregated_stats = manager->GetAggregatedStats();
    EXPECT_EQ(aggregated_stats.total_queries, 0);
}

TEST_F(SSTLearnedIndexManagerTest, DisabledManager) {
    // Test with disabled manager
    options.enabled = false;
    auto disabled_manager = std::make_unique<SSTLearnedIndexManager>(options);
    
    std::string sst_file = GetTestFilePath("test.sst");
    auto training_data = GenerateTrainingData(100);
    
    // Should not create learned index when disabled
    EXPECT_FALSE(disabled_manager->CreateLearnedIndex(sst_file, training_data));
    EXPECT_FALSE(disabled_manager->HasLearnedIndex(sst_file));
    
    // Predictions should return invalid results
    int prediction = disabled_manager->PredictBlockIndex(sst_file, 100);
    EXPECT_EQ(prediction, -1);
}

TEST_F(SSTLearnedIndexManagerTest, InvalidTrainingData) {
    std::string sst_file = GetTestFilePath("test.sst");
    
    // Test with empty training data
    std::vector<std::pair<uint64_t, uint32_t>> empty_data;
    EXPECT_FALSE(manager->CreateLearnedIndex(sst_file, empty_data));
    
    // Test with insufficient training data
    std::vector<std::pair<uint64_t, uint32_t>> insufficient_data = {{0, 0}};
    EXPECT_FALSE(manager->CreateLearnedIndex(sst_file, insufficient_data));
}

TEST_F(SSTLearnedIndexManagerTest, NonExistentFile) {
    std::string non_existent_file = GetTestFilePath("non_existent.sst");
    
    // Should not be able to load non-existent learned index
    EXPECT_FALSE(manager->LoadLearnedIndex(non_existent_file));
    EXPECT_FALSE(manager->HasLearnedIndex(non_existent_file));
    
    // Prediction should return invalid result
    int prediction = manager->PredictBlockIndex(non_existent_file, 100);
    EXPECT_EQ(prediction, -1);
}