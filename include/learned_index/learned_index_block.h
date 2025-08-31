#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace rocksdb {
namespace learned_index {

// Magic number for learned index blocks: "LIDX"
constexpr uint32_t LEARNED_INDEX_MAGIC = 0x4C494458;

// Version of the learned index format
constexpr uint32_t LEARNED_INDEX_VERSION = 1;

// Supported model types
enum class ModelType : uint32_t {
    LINEAR = 1,
    NEURAL_NET = 2,
    POLYNOMIAL = 3
};

// Model metadata structure
struct ModelMetadata {
    uint64_t training_samples;
    double training_accuracy;
    double validation_accuracy;
    uint64_t training_timestamp;
    uint64_t update_at;

    ModelMetadata() 
        : training_samples(0)
        , training_accuracy(0.0)
        , validation_accuracy(0.0)
        , training_timestamp(0)
        , update_at(0) {}
};

// Block-level prediction structure
struct BlockPrediction {
    uint32_t block_index;
    uint64_t predicted_start_key;
    uint64_t predicted_end_key;
    double confidence;

    BlockPrediction()
        : block_index(0)
        , predicted_start_key(0)
        , predicted_end_key(0)
        , confidence(0.0) {}
        
    BlockPrediction(uint32_t idx, uint64_t start, uint64_t end, double conf)
        : block_index(idx)
        , predicted_start_key(start)
        , predicted_end_key(end)
        , confidence(conf) {}
};

// Main learned index block structure
struct LearnedIndexBlock {
    uint32_t magic_number;
    uint32_t version;
    ModelType model_type;
    uint32_t feature_dimensions;
    uint32_t parameter_count;
    
    // Model parameters (variable length)
    std::vector<double> parameters;
    
    // Model metadata
    ModelMetadata metadata;
    
    // Block-level predictions
    std::vector<BlockPrediction> block_predictions;
    
    uint32_t checksum;

    LearnedIndexBlock()
        : magic_number(LEARNED_INDEX_MAGIC)
        , version(LEARNED_INDEX_VERSION)
        , model_type(ModelType::LINEAR)
        , feature_dimensions(1)
        , parameter_count(0)
        , checksum(0) {}

    // Validate the block structure
    bool IsValid() const;
    
    // Calculate and update checksum
    void UpdateChecksum();
    
    // Serialize to binary format
    std::string Serialize() const;
    
    // Deserialize from binary format
    bool Deserialize(const std::string& data);
    
    // Get size in bytes when serialized
    size_t GetSerializedSize() const;
};

} // namespace learned_index
} // namespace rocksdb