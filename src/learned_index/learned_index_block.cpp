#include "learned_index/learned_index_block.h"
#include <crc32c/crc32c.h>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <string>

namespace rocksdb {
namespace learned_index {

LearnedIndexBlock::LearnedIndexBlock() 
    : magic_number_(kLearnedIndexMagicNumber),
      version_(kLearnedIndexVersion),
      model_type_(ModelType::kLinear),
      feature_dimensions_(1),
      parameter_count_(0),
      checksum_(0) {
}

void LearnedIndexBlock::EncodeTo(std::string* dst) const {
  // Header
  dst->append(reinterpret_cast<const char*>(&magic_number_), sizeof(magic_number_));
  dst->append(reinterpret_cast<const char*>(&version_), sizeof(version_));
  dst->append(reinterpret_cast<const char*>(&model_type_), sizeof(model_type_));
  dst->append(reinterpret_cast<const char*>(&feature_dimensions_), sizeof(feature_dimensions_));
  
  // Parameters
  uint32_t param_count = static_cast<uint32_t>(parameters_.size());
  dst->append(reinterpret_cast<const char*>(&param_count), sizeof(param_count));
  
  for (const auto& param : parameters_) {
    dst->append(reinterpret_cast<const char*>(&param), sizeof(param));
  }
  
  // Metadata
  dst->append(reinterpret_cast<const char*>(&metadata_), sizeof(metadata_));
  
  // Block predictions
  uint32_t prediction_count = static_cast<uint32_t>(block_predictions_.size());
  dst->append(reinterpret_cast<const char*>(&prediction_count), sizeof(prediction_count));
  
  for (const auto& prediction : block_predictions_) {
    dst->append(reinterpret_cast<const char*>(&prediction), sizeof(prediction));
  }
  
  // Calculate and append checksum
  uint32_t checksum = crc32c::Crc32c(dst->data(), dst->size());
  dst->append(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
}

bool LearnedIndexBlock::DecodeFrom(const char* data, size_t size) {
  if (size < sizeof(magic_number_) + sizeof(version_) + sizeof(model_type_) + 
            sizeof(feature_dimensions_) + sizeof(uint32_t) + sizeof(checksum_)) {
    return false;
  }
  
  const char* ptr = data;
  
  // Header
  magic_number_ = *reinterpret_cast<const uint32_t*>(ptr);
  ptr += sizeof(magic_number_);
  
  if (magic_number_ != kLearnedIndexMagicNumber) {
    return false;
  }
  
  version_ = *reinterpret_cast<const uint32_t*>(ptr);
  ptr += sizeof(version_);
  
  model_type_ = *reinterpret_cast<const ModelType*>(ptr);
  ptr += sizeof(model_type_);
  
  feature_dimensions_ = *reinterpret_cast<const uint32_t*>(ptr);
  ptr += sizeof(feature_dimensions_);
  
  // Parameters
  parameter_count_ = *reinterpret_cast<const uint32_t*>(ptr);
  ptr += sizeof(parameter_count_);
  
  parameters_.clear();
  parameters_.reserve(parameter_count_);
  
  for (uint32_t i = 0; i < parameter_count_; ++i) {
    if (ptr + sizeof(double) > data + size) return false;
    parameters_.push_back(*reinterpret_cast<const double*>(ptr));
    ptr += sizeof(double);
  }
  
  // Metadata
  if (ptr + sizeof(metadata_) > data + size) return false;
  metadata_ = *reinterpret_cast<const ModelMetadata*>(ptr);
  ptr += sizeof(metadata_);
  
  // Block predictions
  if (ptr + sizeof(uint32_t) > data + size) return false;
  uint32_t prediction_count = *reinterpret_cast<const uint32_t*>(ptr);
  ptr += sizeof(uint32_t);
  
  block_predictions_.clear();
  block_predictions_.reserve(prediction_count);
  
  for (uint32_t i = 0; i < prediction_count; ++i) {
    if (ptr + sizeof(BlockPrediction) > data + size) return false;
    block_predictions_.push_back(*reinterpret_cast<const BlockPrediction*>(ptr));
    ptr += sizeof(BlockPrediction);
  }
  
  // Checksum validation
  if (ptr + sizeof(checksum_) > data + size) return false;
  checksum_ = *reinterpret_cast<const uint32_t*>(ptr);
  
  uint32_t calculated_checksum = crc32c::Crc32c(data, size - sizeof(checksum_));
  return calculated_checksum == checksum_;
}

void LearnedIndexBlock::AddBlockPrediction(const BlockPrediction& prediction) {
  block_predictions_.push_back(prediction);
  
  // Keep predictions sorted by predicted_start_key for binary search
  std::sort(block_predictions_.begin(), block_predictions_.end(),
            [](const BlockPrediction& a, const BlockPrediction& b) {
              return a.predicted_start_key < b.predicted_start_key;
            });
}

uint32_t LearnedIndexBlock::PredictBlockIndex(uint64_t key) const {
  if (parameters_.empty()) {
    return FindBestBlockPrediction(key);
  }
  
  double prediction = 0.0;
  
  switch (model_type_) {
    case ModelType::kLinear:
      prediction = EvaluateLinearModel(key);
      break;
    case ModelType::kPolynomial:
      prediction = EvaluatePolynomialModel(key);
      break;
    case ModelType::kNeuralNet:
      // Neural network evaluation would be implemented here
      // For now, fall back to block predictions
      return FindBestBlockPrediction(key);
    default:
      return FindBestBlockPrediction(key);
  }
  
  // Convert prediction to block index
  uint32_t predicted_block = static_cast<uint32_t>(std::max(0.0, prediction));
  
  // Validate against block predictions if available
  if (!block_predictions_.empty()) {
    uint32_t fallback_block = FindBestBlockPrediction(key);
    
    // Use fallback if model prediction seems unreasonable
    if (predicted_block >= block_predictions_.size()) {
      predicted_block = fallback_block;
    }
  }
  
  return predicted_block;
}

double LearnedIndexBlock::GetPredictionConfidence(uint64_t key) const {
  // Find the closest block prediction to estimate confidence
  if (block_predictions_.empty()) {
    return 0.5; // Default moderate confidence
  }
  
  auto it = std::lower_bound(block_predictions_.begin(), block_predictions_.end(), key,
                            [](const BlockPrediction& pred, uint64_t k) {
                              return pred.predicted_end_key < k;
                            });
  
  if (it != block_predictions_.end() && 
      key >= it->predicted_start_key && key <= it->predicted_end_key) {
    return it->confidence;
  }
  
  // Interpolate confidence between neighboring predictions
  if (it != block_predictions_.begin() && it != block_predictions_.end()) {
    auto prev_it = it - 1;
    double dist_to_prev = static_cast<double>(key - prev_it->predicted_end_key);
    double dist_to_next = static_cast<double>(it->predicted_start_key - key);
    double total_dist = dist_to_prev + dist_to_next;
    
    if (total_dist > 0) {
      return (prev_it->confidence * dist_to_next + it->confidence * dist_to_prev) / total_dist;
    }
  }
  
  return 0.3; // Low confidence for keys outside known ranges
}

bool LearnedIndexBlock::IsValid() const {
  if (magic_number_ != kLearnedIndexMagicNumber) return false;
  if (version_ != kLearnedIndexVersion) return false;
  if (parameters_.size() != parameter_count_) return false;
  
  // Validate block predictions are sorted
  for (size_t i = 1; i < block_predictions_.size(); ++i) {
    if (block_predictions_[i-1].predicted_start_key > block_predictions_[i].predicted_start_key) {
      return false;
    }
  }
  
  return true;
}

uint32_t LearnedIndexBlock::CalculateChecksum() const {
  std::string encoded;
  EncodeTo(&encoded);
  return crc32c::Crc32c(encoded.data(), encoded.size() - sizeof(checksum_));
}

double LearnedIndexBlock::EvaluateLinearModel(uint64_t key) const {
  if (parameters_.size() < 2) return 0.0;
  
  double normalized_key = static_cast<double>(key);
  return parameters_[0] + parameters_[1] * normalized_key;
}

double LearnedIndexBlock::EvaluatePolynomialModel(uint64_t key) const {
  if (parameters_.empty()) return 0.0;
  
  double normalized_key = static_cast<double>(key);
  double result = parameters_[0];
  double key_power = normalized_key;
  
  for (size_t i = 1; i < parameters_.size(); ++i) {
    result += parameters_[i] * key_power;
    key_power *= normalized_key;
  }
  
  return result;
}

uint32_t LearnedIndexBlock::FindBestBlockPrediction(uint64_t key) const {
  if (block_predictions_.empty()) return 0;
  
  auto it = std::lower_bound(block_predictions_.begin(), block_predictions_.end(), key,
                            [](const BlockPrediction& pred, uint64_t k) {
                              return pred.predicted_end_key < k;
                            });
  
  if (it != block_predictions_.end() && 
      key >= it->predicted_start_key && key <= it->predicted_end_key) {
    return it->block_index;
  }
  
  // If no exact match, return the closest prediction
  if (it == block_predictions_.end()) {
    return block_predictions_.back().block_index;
  }
  
  if (it == block_predictions_.begin()) {
    return it->block_index;
  }
  
  // Choose between previous and current based on distance
  auto prev_it = it - 1;
  uint64_t dist_to_prev = key - prev_it->predicted_end_key;
  uint64_t dist_to_current = it->predicted_start_key - key;
  
  return (dist_to_prev <= dist_to_current) ? prev_it->block_index : it->block_index;
}

} // namespace learned_index
} // namespace rocksdb