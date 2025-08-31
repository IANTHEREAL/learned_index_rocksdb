#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace rocksdb {
namespace learned_index {

constexpr uint32_t kLearnedIndexMagicNumber = 0x4C494458; // "LIDX"
constexpr uint32_t kLearnedIndexVersion = 1;

enum class ModelType : uint32_t {
  kLinear = 1,
  kNeuralNet = 2,
  kPolynomial = 3
};

struct ModelMetadata {
  uint64_t training_samples;
  double training_accuracy;
  double validation_accuracy;
  uint64_t training_timestamp;
  uint64_t update_at;
  
  ModelMetadata() : training_samples(0), training_accuracy(0.0), 
                   validation_accuracy(0.0), training_timestamp(0), update_at(0) {}
};

struct BlockPrediction {
  uint32_t block_index;
  uint64_t predicted_start_key;
  uint64_t predicted_end_key;
  double confidence;
  
  BlockPrediction() : block_index(0), predicted_start_key(0), 
                     predicted_end_key(0), confidence(0.0) {}
  
  BlockPrediction(uint32_t idx, uint64_t start, uint64_t end, double conf)
      : block_index(idx), predicted_start_key(start), 
        predicted_end_key(end), confidence(conf) {}
};

class LearnedIndexBlock {
public:
  LearnedIndexBlock();
  ~LearnedIndexBlock() = default;

  // Serialization
  void EncodeTo(std::string* dst) const;
  bool DecodeFrom(const char* data, size_t size);
  
  // Model management
  void SetModelType(ModelType type) { model_type_ = type; }
  ModelType GetModelType() const { return model_type_; }
  
  void SetParameters(const std::vector<double>& params) { parameters_ = params; }
  const std::vector<double>& GetParameters() const { return parameters_; }
  
  void SetMetadata(const ModelMetadata& metadata) { metadata_ = metadata; }
  const ModelMetadata& GetMetadata() const { return metadata_; }
  
  // Block predictions
  void AddBlockPrediction(const BlockPrediction& prediction);
  const std::vector<BlockPrediction>& GetBlockPredictions() const { return block_predictions_; }
  
  // Prediction methods
  uint32_t PredictBlockIndex(uint64_t key) const;
  double GetPredictionConfidence(uint64_t key) const;
  
  // Validation
  bool IsValid() const;
  uint32_t CalculateChecksum() const;

private:
  uint32_t magic_number_;
  uint32_t version_;
  ModelType model_type_;
  uint32_t feature_dimensions_;
  uint32_t parameter_count_;
  
  std::vector<double> parameters_;
  ModelMetadata metadata_;
  std::vector<BlockPrediction> block_predictions_;
  
  uint32_t checksum_;
  
  // Internal prediction helpers
  double EvaluateLinearModel(uint64_t key) const;
  double EvaluatePolynomialModel(uint64_t key) const;
  uint32_t FindBestBlockPrediction(uint64_t key) const;
};

} // namespace learned_index
} // namespace rocksdb