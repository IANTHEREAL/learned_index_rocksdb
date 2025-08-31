#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace learned_index {

// Magic number for learned index blocks: "LIDX"
constexpr uint32_t LEARNED_INDEX_MAGIC_NUMBER = 0x4C494458;

// Current version of the learned index format
constexpr uint32_t LEARNED_INDEX_VERSION = 1;

// Supported model types
enum class ModelType : uint32_t {
  LINEAR = 1,
  NEURAL_NET = 2,
  POLYNOMIAL = 3,
  DECISION_TREE = 4
};

// Model metadata for tracking training and performance
struct ModelMetadata {
  uint64_t training_samples;      // Number of samples used for training
  double training_accuracy;       // Training accuracy (0.0 to 1.0)
  double validation_accuracy;     // Validation accuracy (0.0 to 1.0)
  uint64_t training_timestamp;    // When model was trained (milliseconds since epoch)
  uint64_t last_update_timestamp; // Last update timestamp
  
  ModelMetadata()
    : training_samples(0),
      training_accuracy(0.0),
      validation_accuracy(0.0),
      training_timestamp(0),
      last_update_timestamp(0) {}
};

// Block-level prediction for SST files
struct BlockPrediction {
  uint32_t block_index;           // Index of the data block
  uint64_t predicted_start_key;   // Predicted start key for this block
  uint64_t predicted_end_key;     // Predicted end key for this block
  double confidence;              // Confidence score (0.0 to 1.0)
  
  BlockPrediction()
    : block_index(0),
      predicted_start_key(0),
      predicted_end_key(0),
      confidence(0.0) {}
      
  BlockPrediction(uint32_t idx, uint64_t start_key, uint64_t end_key, double conf)
    : block_index(idx),
      predicted_start_key(start_key),
      predicted_end_key(end_key),
      confidence(conf) {}
};

// Main learned index block structure for SST files
struct LearnedIndexBlock {
  uint32_t magic_number;          // Magic number for validation
  uint32_t version;               // Format version
  ModelType model_type;           // Type of ML model used
  uint32_t feature_dimensions;    // Number of input features
  uint32_t parameter_count;       // Number of model parameters
  
  // Model parameters (variable length based on model type)
  std::vector<double> parameters;
  
  // Model training and performance metadata
  ModelMetadata metadata;
  
  // Block-level predictions for this SST file
  std::vector<BlockPrediction> block_predictions;
  
  uint32_t checksum;              // CRC32 checksum for integrity
  
  LearnedIndexBlock()
    : magic_number(LEARNED_INDEX_MAGIC_NUMBER),
      version(LEARNED_INDEX_VERSION),
      model_type(ModelType::LINEAR),
      feature_dimensions(1),
      parameter_count(0),
      checksum(0) {}
      
  // Serialize the learned index block to bytes
  std::string Serialize() const;
  
  // Deserialize from bytes
  bool Deserialize(const std::string& data);
  
  // Calculate and update checksum
  void UpdateChecksum();
  
  // Verify checksum integrity
  bool VerifyChecksum() const;
  
  // Get the predicted block index for a given key
  int PredictBlockIndex(uint64_t key) const;
  
  // Get prediction confidence for a given key
  double GetPredictionConfidence(uint64_t key) const;
  
  // Add a new block prediction
  void AddBlockPrediction(const BlockPrediction& prediction);
  
  // Update model parameters
  void UpdateModelParameters(const std::vector<double>& new_parameters);
  
  // Check if the learned index is valid and ready to use
  bool IsValid() const;
};

} // namespace learned_index