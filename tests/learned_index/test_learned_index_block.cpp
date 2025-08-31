#include <gtest/gtest.h>
#include "learned_index/learned_index_block.h"

using namespace rocksdb::learned_index;

class LearnedIndexBlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a valid block for testing
        block_.model_type = ModelType::LINEAR;
        block_.feature_dimensions = 1;
        block_.parameters = {1.5, 0.8};
        block_.parameter_count = 2;
        
        block_.metadata.training_samples = 1000;
        block_.metadata.training_accuracy = 0.85;
        block_.metadata.validation_accuracy = 0.83;
        
        block_.block_predictions.push_back(BlockPrediction(0, 1000, 2000, 0.95));
        block_.block_predictions.push_back(BlockPrediction(1, 2000, 3000, 0.88));
        
        block_.UpdateChecksum();
    }
    
    LearnedIndexBlock block_;
};

TEST_F(LearnedIndexBlockTest, DefaultConstructor) {
    LearnedIndexBlock default_block;
    
    EXPECT_EQ(default_block.magic_number, LEARNED_INDEX_MAGIC);
    EXPECT_EQ(default_block.version, LEARNED_INDEX_VERSION);
    EXPECT_EQ(default_block.model_type, ModelType::LINEAR);
    EXPECT_EQ(default_block.feature_dimensions, 1U);
    EXPECT_EQ(default_block.parameter_count, 0U);
    EXPECT_TRUE(default_block.parameters.empty());
    EXPECT_TRUE(default_block.block_predictions.empty());
}

TEST_F(LearnedIndexBlockTest, IsValid) {
    EXPECT_TRUE(block_.IsValid());
    
    // Test invalid magic number
    LearnedIndexBlock invalid_block = block_;
    invalid_block.magic_number = 0x12345678;
    EXPECT_FALSE(invalid_block.IsValid());
    
    // Test invalid version
    invalid_block = block_;
    invalid_block.version = 999;
    EXPECT_FALSE(invalid_block.IsValid());
    
    // Test parameter count mismatch
    invalid_block = block_;
    invalid_block.parameter_count = 10;
    EXPECT_FALSE(invalid_block.IsValid());
    
    // Test zero feature dimensions
    invalid_block = block_;
    invalid_block.feature_dimensions = 0;
    EXPECT_FALSE(invalid_block.IsValid());
}

TEST_F(LearnedIndexBlockTest, Serialization) {
    std::string serialized = block_.Serialize();
    EXPECT_FALSE(serialized.empty());
    EXPECT_GE(serialized.size(), block_.GetSerializedSize());
}

TEST_F(LearnedIndexBlockTest, Deserialization) {
    std::string serialized = block_.Serialize();
    
    LearnedIndexBlock deserialized;
    EXPECT_TRUE(deserialized.Deserialize(serialized));
    EXPECT_TRUE(deserialized.IsValid());
    
    // Check that all fields are preserved
    EXPECT_EQ(deserialized.magic_number, block_.magic_number);
    EXPECT_EQ(deserialized.version, block_.version);
    EXPECT_EQ(deserialized.model_type, block_.model_type);
    EXPECT_EQ(deserialized.feature_dimensions, block_.feature_dimensions);
    EXPECT_EQ(deserialized.parameter_count, block_.parameter_count);
    EXPECT_EQ(deserialized.parameters, block_.parameters);
    EXPECT_EQ(deserialized.checksum, block_.checksum);
    
    // Check metadata
    EXPECT_EQ(deserialized.metadata.training_samples, block_.metadata.training_samples);
    EXPECT_DOUBLE_EQ(deserialized.metadata.training_accuracy, block_.metadata.training_accuracy);
    
    // Check block predictions
    EXPECT_EQ(deserialized.block_predictions.size(), block_.block_predictions.size());
    for (size_t i = 0; i < block_.block_predictions.size(); ++i) {
        EXPECT_EQ(deserialized.block_predictions[i].block_index, 
                  block_.block_predictions[i].block_index);
        EXPECT_EQ(deserialized.block_predictions[i].predicted_start_key, 
                  block_.block_predictions[i].predicted_start_key);
        EXPECT_EQ(deserialized.block_predictions[i].predicted_end_key, 
                  block_.block_predictions[i].predicted_end_key);
        EXPECT_DOUBLE_EQ(deserialized.block_predictions[i].confidence, 
                        block_.block_predictions[i].confidence);
    }
}

TEST_F(LearnedIndexBlockTest, DeserializationInvalidData) {
    // Test with empty data
    LearnedIndexBlock block;
    EXPECT_FALSE(block.Deserialize(""));
    
    // Test with too small data
    std::string small_data(10, '\0');
    EXPECT_FALSE(block.Deserialize(small_data));
    
    // Test with corrupted magic number
    std::string serialized = block_.Serialize();
    serialized[0] = 0xFF; // Corrupt first byte
    EXPECT_FALSE(block.Deserialize(serialized));
}

TEST_F(LearnedIndexBlockTest, ChecksumValidation) {
    std::string serialized = block_.Serialize();
    
    // Corrupt the data (but not checksum)
    serialized[20] = ~serialized[20];
    
    LearnedIndexBlock corrupted_block;
    EXPECT_FALSE(corrupted_block.Deserialize(serialized));
}

TEST_F(LearnedIndexBlockTest, GetSerializedSize) {
    size_t expected_size = 
        sizeof(uint32_t) * 4 +  // magic, version, model_type, feature_dims, param_count
        block_.parameters.size() * sizeof(double) +  // parameters
        sizeof(ModelMetadata) +  // metadata
        sizeof(uint32_t) +  // prediction count
        block_.block_predictions.size() * sizeof(BlockPrediction) +  // predictions
        sizeof(uint32_t);  // checksum
    
    EXPECT_EQ(block_.GetSerializedSize(), expected_size);
}

TEST(BlockPredictionTest, Constructor) {
    BlockPrediction prediction;
    EXPECT_EQ(prediction.block_index, 0U);
    EXPECT_EQ(prediction.predicted_start_key, 0U);
    EXPECT_EQ(prediction.predicted_end_key, 0U);
    EXPECT_DOUBLE_EQ(prediction.confidence, 0.0);
    
    BlockPrediction prediction2(5, 1000, 2000, 0.95);
    EXPECT_EQ(prediction2.block_index, 5U);
    EXPECT_EQ(prediction2.predicted_start_key, 1000U);
    EXPECT_EQ(prediction2.predicted_end_key, 2000U);
    EXPECT_DOUBLE_EQ(prediction2.confidence, 0.95);
}

TEST(ModelMetadataTest, DefaultConstructor) {
    ModelMetadata metadata;
    EXPECT_EQ(metadata.training_samples, 0U);
    EXPECT_DOUBLE_EQ(metadata.training_accuracy, 0.0);
    EXPECT_DOUBLE_EQ(metadata.validation_accuracy, 0.0);
    EXPECT_EQ(metadata.training_timestamp, 0U);
    EXPECT_EQ(metadata.update_at, 0U);
}