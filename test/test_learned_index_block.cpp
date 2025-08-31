#include <gtest/gtest.h>
#include "learned_index/learned_index_block.h"
#include <vector>
#include <cstdint>
#include <thread>
#include <chrono>

using namespace learned_index;

class LearnedIndexBlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        block = std::make_unique<LearnedIndexBlock>();
    }

    void TearDown() override {
        block.reset();
    }

    std::unique_ptr<LearnedIndexBlock> block;
};

TEST_F(LearnedIndexBlockTest, DefaultConstructor) {
    EXPECT_EQ(block->magic_number, LEARNED_INDEX_MAGIC_NUMBER);
    EXPECT_EQ(block->version, LEARNED_INDEX_VERSION);
    EXPECT_EQ(block->model_type, ModelType::LINEAR);
    EXPECT_EQ(block->feature_dimensions, 1);
    EXPECT_EQ(block->parameter_count, 0);
    EXPECT_TRUE(block->parameters.empty());
    EXPECT_TRUE(block->block_predictions.empty());
}

TEST_F(LearnedIndexBlockTest, SerializationDeserialization) {
    // Setup test data
    block->model_type = ModelType::LINEAR;
    block->feature_dimensions = 1;
    block->parameters = {2.5, 10.0}; // slope and intercept
    block->parameter_count = 2;
    
    // Add block predictions
    BlockPrediction pred1(0, 100, 200, 0.95);
    BlockPrediction pred2(1, 300, 400, 0.90);
    block->AddBlockPrediction(pred1);
    block->AddBlockPrediction(pred2);
    
    // Update metadata
    block->metadata.training_samples = 1000;
    block->metadata.training_accuracy = 0.95;
    block->metadata.validation_accuracy = 0.92;
    
    block->UpdateChecksum();
    
    // Serialize
    std::string serialized = block->Serialize();
    EXPECT_FALSE(serialized.empty());
    
    // Deserialize into new block
    LearnedIndexBlock new_block;
    EXPECT_TRUE(new_block.Deserialize(serialized));
    
    // Verify deserialized data
    EXPECT_EQ(new_block.magic_number, LEARNED_INDEX_MAGIC_NUMBER);
    EXPECT_EQ(new_block.version, LEARNED_INDEX_VERSION);
    EXPECT_EQ(new_block.model_type, ModelType::LINEAR);
    EXPECT_EQ(new_block.feature_dimensions, 1);
    EXPECT_EQ(new_block.parameter_count, 2);
    EXPECT_EQ(new_block.parameters.size(), 2);
    EXPECT_DOUBLE_EQ(new_block.parameters[0], 2.5);
    EXPECT_DOUBLE_EQ(new_block.parameters[1], 10.0);
    
    EXPECT_EQ(new_block.metadata.training_samples, 1000);
    EXPECT_DOUBLE_EQ(new_block.metadata.training_accuracy, 0.95);
    EXPECT_DOUBLE_EQ(new_block.metadata.validation_accuracy, 0.92);
    
    EXPECT_EQ(new_block.block_predictions.size(), 2);
    
    // Verify checksum
    EXPECT_TRUE(new_block.VerifyChecksum());
}

TEST_F(LearnedIndexBlockTest, InvalidDeserialization) {
    // Test with invalid data
    std::string invalid_data = "invalid_data";
    LearnedIndexBlock invalid_block;
    EXPECT_FALSE(invalid_block.Deserialize(invalid_data));
    
    // Test with empty data
    std::string empty_data;
    LearnedIndexBlock empty_block;
    EXPECT_FALSE(empty_block.Deserialize(empty_data));
}

TEST_F(LearnedIndexBlockTest, BlockPrediction) {
    // Setup linear model: y = 0.5x + 10
    block->model_type = ModelType::LINEAR;
    block->parameters = {0.5, 10.0}; // slope = 0.5, intercept = 10.0
    block->parameter_count = 2;
    
    // Add some block predictions
    for (int i = 0; i < 5; ++i) {
        BlockPrediction pred;
        pred.block_index = i;
        pred.predicted_start_key = i * 100;
        pred.predicted_end_key = (i + 1) * 100 - 1;
        pred.confidence = 0.9;
        block->AddBlockPrediction(pred);
    }
    
    block->UpdateChecksum();
    
    // Test predictions
    EXPECT_TRUE(block->IsValid());
    
    // Test key 0: should predict block 5 (0.5 * 0 + 10 = 10, but clamped to valid range)
    int predicted_block = block->PredictBlockIndex(0);
    EXPECT_GE(predicted_block, 0);
    EXPECT_LT(predicted_block, 5);
    
    // Test confidence
    double confidence = block->GetPredictionConfidence(0);
    EXPECT_GE(confidence, 0.0);
    EXPECT_LE(confidence, 1.0);
}

TEST_F(LearnedIndexBlockTest, UpdateModelParameters) {
    // Initial parameters
    std::vector<double> initial_params = {1.0, 5.0};
    block->UpdateModelParameters(initial_params);
    
    EXPECT_EQ(block->parameters.size(), 2);
    EXPECT_EQ(block->parameter_count, 2);
    EXPECT_DOUBLE_EQ(block->parameters[0], 1.0);
    EXPECT_DOUBLE_EQ(block->parameters[1], 5.0);
    EXPECT_GT(block->metadata.last_update_timestamp, 0);
    
    // Update parameters
    std::vector<double> updated_params = {2.0, 8.0, 1.5};
    uint64_t old_timestamp = block->metadata.last_update_timestamp;
    
    // Add small delay to ensure timestamp changes
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    
    block->UpdateModelParameters(updated_params);
    
    EXPECT_EQ(block->parameters.size(), 3);
    EXPECT_EQ(block->parameter_count, 3);
    EXPECT_DOUBLE_EQ(block->parameters[0], 2.0);
    EXPECT_DOUBLE_EQ(block->parameters[1], 8.0);
    EXPECT_DOUBLE_EQ(block->parameters[2], 1.5);
    EXPECT_GT(block->metadata.last_update_timestamp, old_timestamp);
}

TEST_F(LearnedIndexBlockTest, BlockPredictionManagement) {
    // Test adding block predictions
    BlockPrediction pred1(0, 100, 199, 0.95);
    BlockPrediction pred2(1, 200, 299, 0.90);
    BlockPrediction pred3(2, 50, 99, 0.85);  // Will be inserted in the middle due to sorting
    
    block->AddBlockPrediction(pred1);
    block->AddBlockPrediction(pred2);
    block->AddBlockPrediction(pred3);
    
    EXPECT_EQ(block->block_predictions.size(), 3);
    
    // Verify sorting (should be sorted by predicted_start_key)
    EXPECT_LE(block->block_predictions[0].predicted_start_key,
              block->block_predictions[1].predicted_start_key);
    EXPECT_LE(block->block_predictions[1].predicted_start_key,
              block->block_predictions[2].predicted_start_key);
    
    // Verify the order is: pred3 (50), pred1 (100), pred2 (200)
    EXPECT_EQ(block->block_predictions[0].predicted_start_key, 50);
    EXPECT_EQ(block->block_predictions[1].predicted_start_key, 100);
    EXPECT_EQ(block->block_predictions[2].predicted_start_key, 200);
}

TEST_F(LearnedIndexBlockTest, ChecksumValidation) {
    // Setup some data
    block->parameters = {1.5, 2.5};
    block->parameter_count = 2;
    
    BlockPrediction pred(0, 100, 200, 0.9);
    block->AddBlockPrediction(pred);
    
    // Calculate checksum
    block->UpdateChecksum();
    uint32_t original_checksum = block->checksum;
    
    // Verify checksum is valid
    EXPECT_TRUE(block->VerifyChecksum());
    EXPECT_TRUE(block->IsValid());
    
    // Corrupt the checksum
    block->checksum = 0xDEADBEEF;
    EXPECT_FALSE(block->VerifyChecksum());
    EXPECT_FALSE(block->IsValid());
    
    // Restore checksum
    block->checksum = original_checksum;
    EXPECT_TRUE(block->VerifyChecksum());
    EXPECT_TRUE(block->IsValid());
}