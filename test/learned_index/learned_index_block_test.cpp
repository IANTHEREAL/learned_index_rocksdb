#include "learned_index/learned_index_block.h"
#include <vector>
#include <functional>
#include <cassert>
#include <iostream>

using namespace rocksdb::learned_index;

struct TestCase {
    std::string name;
    std::function<bool()> test_func;
};

bool TestLearnedIndexBlockConstruction() {
    LearnedIndexBlock block;
    
    // Test default construction
    if (block.GetModelType() != ModelType::kLinear) return false;
    if (!block.GetParameters().empty()) return false;
    if (!block.GetBlockPredictions().empty()) return false;
    
    return true;
}

bool TestLearnedIndexBlockSerialization() {
    LearnedIndexBlock original;
    
    // Set up test data
    original.SetModelType(ModelType::kLinear);
    original.SetParameters({1.5, 2.3});
    
    ModelMetadata metadata;
    metadata.training_samples = 100;
    metadata.training_accuracy = 0.95;
    metadata.validation_accuracy = 0.92;
    metadata.training_timestamp = 1234567890;
    metadata.update_at = 1234567890;
    original.SetMetadata(metadata);
    
    original.AddBlockPrediction(BlockPrediction(0, 1000, 2000, 0.9));
    original.AddBlockPrediction(BlockPrediction(1, 2001, 3000, 0.8));
    
    // Serialize
    std::string serialized;
    original.EncodeTo(&serialized);
    
    if (serialized.empty()) return false;
    
    // Deserialize
    LearnedIndexBlock deserialized;
    if (!deserialized.DecodeFrom(serialized.data(), serialized.size())) {
        return false;
    }
    
    // Verify deserialization
    if (deserialized.GetModelType() != ModelType::kLinear) return false;
    
    const auto& params = deserialized.GetParameters();
    if (params.size() != 2 || params[0] != 1.5 || params[1] != 2.3) return false;
    
    const auto& meta = deserialized.GetMetadata();
    if (meta.training_samples != 100) return false;
    if (meta.training_accuracy != 0.95) return false;
    
    const auto& predictions = deserialized.GetBlockPredictions();
    if (predictions.size() != 2) return false;
    if (predictions[0].block_index != 0 || predictions[0].confidence != 0.9) return false;
    
    return true;
}

bool TestLearnedIndexBlockPrediction() {
    LearnedIndexBlock block;
    
    // Set up linear model: y = 0 + 0.001 * x
    // This should map key ranges to block indices
    block.SetModelType(ModelType::kLinear);
    block.SetParameters({0.0, 0.001});
    
    // Add block predictions for validation
    block.AddBlockPrediction(BlockPrediction(0, 0, 999, 0.9));
    block.AddBlockPrediction(BlockPrediction(1, 1000, 1999, 0.85));
    block.AddBlockPrediction(BlockPrediction(2, 2000, 2999, 0.8));
    
    // Test predictions
    uint32_t pred1 = block.PredictBlockIndex(500);   // Should be in block 0
    uint32_t pred2 = block.PredictBlockIndex(1500);  // Should be in block 1
    uint32_t pred3 = block.PredictBlockIndex(2500);  // Should be in block 2
    
    if (pred1 != 0 || pred2 != 1 || pred3 != 2) {
        std::cout << "Prediction failed: " << pred1 << ", " << pred2 << ", " << pred3 << std::endl;
        return false;
    }
    
    // Test confidence
    double conf1 = block.GetPredictionConfidence(500);
    double conf2 = block.GetPredictionConfidence(5000); // Outside known range
    
    if (conf1 < 0.8 || conf2 > 0.5) {
        std::cout << "Confidence failed: " << conf1 << ", " << conf2 << std::endl;
        return false;
    }
    
    return true;
}

bool TestLearnedIndexBlockValidation() {
    LearnedIndexBlock block;
    
    // Test invalid magic number scenario through serialization corruption
    std::string serialized;
    block.EncodeTo(&serialized);
    
    // Corrupt the magic number
    if (serialized.size() >= 4) {
        serialized[0] = 0xFF;
        serialized[1] = 0xFF;
        serialized[2] = 0xFF;
        serialized[3] = 0xFF;
    }
    
    LearnedIndexBlock corrupted;
    if (corrupted.DecodeFrom(serialized.data(), serialized.size())) {
        return false; // Should fail due to invalid magic number
    }
    
    return true;
}

bool TestBlockPredictionSorting() {
    LearnedIndexBlock block;
    
    // Add predictions out of order
    block.AddBlockPrediction(BlockPrediction(2, 2000, 3000, 0.8));
    block.AddBlockPrediction(BlockPrediction(0, 0, 1000, 0.9));
    block.AddBlockPrediction(BlockPrediction(1, 1000, 2000, 0.85));
    
    const auto& predictions = block.GetBlockPredictions();
    
    // Should be sorted by predicted_start_key
    if (predictions.size() != 3) return false;
    if (predictions[0].predicted_start_key != 0) return false;
    if (predictions[1].predicted_start_key != 1000) return false;
    if (predictions[2].predicted_start_key != 2000) return false;
    
    return true;
}

std::vector<TestCase> GetLearnedIndexBlockTests() {
    return {
        {"LearnedIndexBlock Construction", TestLearnedIndexBlockConstruction},
        {"LearnedIndexBlock Serialization", TestLearnedIndexBlockSerialization},
        {"LearnedIndexBlock Prediction", TestLearnedIndexBlockPrediction},
        {"LearnedIndexBlock Validation", TestLearnedIndexBlockValidation},
        {"BlockPrediction Sorting", TestBlockPredictionSorting}
    };
}