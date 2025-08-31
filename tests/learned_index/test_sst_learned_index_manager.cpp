#include <gtest/gtest.h>
#include "learned_index/sst_learned_index_manager.h"
#include <vector>

using namespace rocksdb::learned_index;

class SSTLearnedIndexManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        options_.model_type = ModelType::LINEAR;
        options_.confidence_threshold = 0.8;
        options_.max_prediction_error_bytes = 1024;
        options_.min_training_samples = 10;
        
        manager_ = std::make_unique<SSTLearnedIndexManager>(options_);
    }
    
    void CreateTrainingData(std::vector<KeyRange>& key_ranges, size_t num_ranges = 10) {
        key_ranges.clear();
        for (size_t i = 0; i < num_ranges; ++i) {
            uint64_t start = i * 1000;
            uint64_t end = (i + 1) * 1000;
            key_ranges.emplace_back(start, end, static_cast<uint32_t>(i), 100);
        }
    }
    
    SSTLearnedIndexOptions options_;
    std::unique_ptr<SSTLearnedIndexManager> manager_;
};

TEST_F(SSTLearnedIndexManagerTest, Constructor) {
    EXPECT_EQ(manager_->GetModelType(), ModelType::LINEAR);
    EXPECT_FALSE(manager_->IsModelReady());
    EXPECT_EQ(manager_->GetModelSize(), 0U);
    EXPECT_DOUBLE_EQ(manager_->GetTrainingAccuracy(), 0.0);
}

TEST_F(SSTLearnedIndexManagerTest, Initialization) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    // Test invalid initialization
    EXPECT_FALSE(manager_->Initialize("", 10000)); // Empty filename
    EXPECT_FALSE(manager_->Initialize("test.sst", 0)); // Zero file size
}

TEST_F(SSTLearnedIndexManagerTest, TrainingSuccess) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::vector<KeyRange> key_ranges;
    CreateTrainingData(key_ranges);
    
    EXPECT_TRUE(manager_->TrainModel(key_ranges));
    EXPECT_TRUE(manager_->IsModelReady());
    EXPECT_GT(manager_->GetModelSize(), 0U);
    EXPECT_GE(manager_->GetTrainingAccuracy(), 0.0);
}

TEST_F(SSTLearnedIndexManagerTest, TrainingFailure) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    // Empty training data
    std::vector<KeyRange> empty_ranges;
    EXPECT_FALSE(manager_->TrainModel(empty_ranges));
    EXPECT_FALSE(manager_->IsModelReady());
    
    // Insufficient training samples
    std::vector<KeyRange> insufficient_ranges;
    CreateTrainingData(insufficient_ranges, 2); // Only 2 ranges, need at least 10 samples
    EXPECT_FALSE(manager_->TrainModel(insufficient_ranges));
}

TEST_F(SSTLearnedIndexManagerTest, PredictionAfterTraining) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::vector<KeyRange> key_ranges;
    CreateTrainingData(key_ranges);
    EXPECT_TRUE(manager_->TrainModel(key_ranges));
    
    // Test predictions
    double confidence;
    uint32_t predicted_block = manager_->PredictBlock(500, &confidence);
    EXPECT_GE(confidence, 0.0);
    EXPECT_LE(confidence, 1.0);
    
    // Test block prediction with confidence check
    uint32_t block_index;
    bool prediction_success = manager_->GetBlockPrediction(1500, &block_index, &confidence);
    if (prediction_success) {
        EXPECT_GE(confidence, options_.confidence_threshold);
        EXPECT_GT(block_index, 0U); // Should predict a reasonable block
    }
}

TEST_F(SSTLearnedIndexManagerTest, PredictionWithoutTraining) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    // Should return 0 for untrained model
    double confidence;
    uint32_t predicted_block = manager_->PredictBlock(500, &confidence);
    EXPECT_EQ(predicted_block, 0U);
    EXPECT_DOUBLE_EQ(confidence, 0.0);
    
    // Block prediction should fail
    uint32_t block_index;
    EXPECT_FALSE(manager_->GetBlockPrediction(500, &block_index, &confidence));
}

TEST_F(SSTLearnedIndexManagerTest, StatisticsUpdating) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::vector<KeyRange> key_ranges;
    CreateTrainingData(key_ranges);
    EXPECT_TRUE(manager_->TrainModel(key_ranges));
    
    const auto& stats = manager_->GetStats();
    EXPECT_EQ(stats.total_queries, 0U);
    EXPECT_EQ(stats.successful_predictions, 0U);
    EXPECT_EQ(stats.fallback_queries, 0U);
    
    // Update statistics
    manager_->UpdateStats(500, 0, 0, false); // Successful prediction
    manager_->UpdateStats(1500, 1, 2, false); // Failed prediction
    manager_->UpdateStats(2500, 2, 0, true); // Fallback used
    
    const auto& updated_stats = manager_->GetStats();
    EXPECT_EQ(updated_stats.total_queries, 3U);
    EXPECT_EQ(updated_stats.successful_predictions, 1U);
    EXPECT_EQ(updated_stats.fallback_queries, 1U);
    EXPECT_DOUBLE_EQ(updated_stats.GetSuccessRate(), 1.0/3.0);
    EXPECT_DOUBLE_EQ(updated_stats.GetFallbackRate(), 1.0/3.0);
}

TEST_F(SSTLearnedIndexManagerTest, ModelSaveAndLoad) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::vector<KeyRange> key_ranges;
    CreateTrainingData(key_ranges);
    EXPECT_TRUE(manager_->TrainModel(key_ranges));
    
    // Save model
    LearnedIndexBlock block;
    EXPECT_TRUE(manager_->SaveModel(&block));
    EXPECT_TRUE(block.IsValid());
    
    // Create new manager and load model
    SSTLearnedIndexManager new_manager(options_);
    EXPECT_TRUE(new_manager.Initialize("test2.sst", 10000));
    EXPECT_TRUE(new_manager.LoadModel(block));
    EXPECT_TRUE(new_manager.IsModelReady());
    
    // Predictions should be similar (though not identical due to different file sizes)
    double orig_confidence, new_confidence;
    uint32_t orig_prediction = manager_->PredictBlock(500, &orig_confidence);
    uint32_t new_prediction = new_manager.PredictBlock(500, &new_confidence);
    
    // At least confidence should be similar
    EXPECT_NEAR(orig_confidence, new_confidence, 0.1);
}

TEST_F(SSTLearnedIndexManagerTest, CacheManagement) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::vector<KeyRange> key_ranges;
    CreateTrainingData(key_ranges);
    EXPECT_TRUE(manager_->TrainModel(key_ranges));
    
    EXPECT_EQ(manager_->GetCacheSize(), 0U);
    EXPECT_DOUBLE_EQ(manager_->GetCacheHitRate(), 0.0);
    
    // Make some predictions to populate cache
    manager_->PredictBlock(500);
    manager_->PredictBlock(1500);
    EXPECT_GT(manager_->GetCacheSize(), 0U);
    
    // Make same prediction again - should hit cache
    manager_->PredictBlock(500);
    EXPECT_GT(manager_->GetCacheHitRate(), 0.0);
    
    // Clear cache
    manager_->ClearPredictionCache();
    EXPECT_EQ(manager_->GetCacheSize(), 0U);
}

TEST_F(SSTLearnedIndexManagerTest, Validation) {
    // Initially invalid (not trained)
    EXPECT_FALSE(manager_->ValidateModel());
    
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::vector<KeyRange> key_ranges;
    CreateTrainingData(key_ranges);
    EXPECT_TRUE(manager_->TrainModel(key_ranges));
    
    // Should be valid after training
    EXPECT_TRUE(manager_->ValidateModel());
}

TEST_F(SSTLearnedIndexManagerTest, DiagnosticsInfo) {
    EXPECT_TRUE(manager_->Initialize("test.sst", 10000));
    
    std::string diagnostics = manager_->GetDiagnosticsInfo();
    EXPECT_FALSE(diagnostics.empty());
    EXPECT_NE(diagnostics.find("test.sst"), std::string::npos);
    EXPECT_NE(diagnostics.find("10000"), std::string::npos);
}

TEST(SSTLearnedIndexStatsTest, SuccessAndFallbackRates) {
    SSTLearnedIndexStats stats;
    
    // Initially zero rates
    EXPECT_DOUBLE_EQ(stats.GetSuccessRate(), 0.0);
    EXPECT_DOUBLE_EQ(stats.GetFallbackRate(), 0.0);
    
    // Add some statistics
    stats.total_queries = 100;
    stats.successful_predictions = 80;
    stats.fallback_queries = 15;
    
    EXPECT_DOUBLE_EQ(stats.GetSuccessRate(), 0.8);
    EXPECT_DOUBLE_EQ(stats.GetFallbackRate(), 0.15);
}

TEST(KeyRangeTest, Constructor) {
    KeyRange range(1000, 2000, 5, 100);
    
    EXPECT_EQ(range.start_key, 1000U);
    EXPECT_EQ(range.end_key, 2000U);
    EXPECT_EQ(range.block_index, 5U);
    EXPECT_EQ(range.key_count, 100U);
}