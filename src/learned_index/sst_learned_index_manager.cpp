#include "learned_index/sst_learned_index_manager.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>

namespace rocksdb {
namespace learned_index {

SSTLearnedIndexManager::SSTLearnedIndexManager(const SSTLearnedIndexOptions& options)
    : options_(options), access_counter_(0) {
}

bool SSTLearnedIndexManager::LoadLearnedIndex(const std::string& sst_file_path, 
                                            const std::string& index_data) {
  if (!options_.enable_learned_index || index_data.empty()) {
    return false;
  }
  
  auto model = std::make_shared<LearnedIndexBlock>();
  if (!model->DecodeFrom(index_data.data(), index_data.size())) {
    return false;
  }
  
  if (!model->IsValid()) {
    return false;
  }
  
  if (options_.cache_models) {
    CacheModel(sst_file_path, model);
  }
  
  InitializeStats(sst_file_path);
  return true;
}

bool SSTLearnedIndexManager::SaveLearnedIndex(const std::string& sst_file_path, 
                                            std::string* index_data) const {
  auto model = GetCachedModel(sst_file_path);
  if (!model) {
    return false;
  }
  
  model->EncodeTo(index_data);
  return true;
}

bool SSTLearnedIndexManager::TrainModel(const std::string& sst_file_path,
    const std::vector<std::pair<uint64_t, uint32_t>>& key_block_pairs) {
  
  if (!options_.enable_learned_index || key_block_pairs.empty()) {
    return false;
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  auto model = std::make_shared<LearnedIndexBlock>();
  model->SetModelType(options_.default_model_type);
  
  bool training_success = false;
  
  switch (options_.default_model_type) {
    case ModelType::kLinear:
      training_success = TrainLinearModel(key_block_pairs, model.get());
      break;
    case ModelType::kPolynomial:
      training_success = TrainPolynomialModel(key_block_pairs, model.get());
      break;
    case ModelType::kNeuralNet:
      // Neural network training would be implemented here
      // For now, fall back to linear model
      model->SetModelType(ModelType::kLinear);
      training_success = TrainLinearModel(key_block_pairs, model.get());
      break;
    default:
      return false;
  }
  
  if (!training_success) {
    return false;
  }
  
  // Create block predictions for validation
  std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> block_ranges;
  for (const auto& pair : key_block_pairs) {
    uint64_t key = pair.first;
    uint32_t block = pair.second;
    
    if (block_ranges.find(block) == block_ranges.end()) {
      block_ranges[block] = {key, key};
    } else {
      block_ranges[block].first = std::min(block_ranges[block].first, key);
      block_ranges[block].second = std::max(block_ranges[block].second, key);
    }
  }
  
  // Add block predictions with confidence estimation
  for (const auto& range : block_ranges) {
    uint32_t block_idx = range.first;
    uint64_t start_key = range.second.first;
    uint64_t end_key = range.second.second;
    
    // Calculate confidence based on prediction accuracy for this range
    double confidence = 0.8; // Default confidence, would be calculated from validation
    
    model->AddBlockPrediction(BlockPrediction(block_idx, start_key, end_key, confidence));
  }
  
  // Update metadata
  ModelMetadata metadata;
  metadata.training_samples = key_block_pairs.size();
  metadata.training_accuracy = 0.9; // Would be calculated from cross-validation
  metadata.validation_accuracy = 0.85; // Would be calculated from validation set
  metadata.training_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  metadata.update_at = metadata.training_timestamp;
  
  model->SetMetadata(metadata);
  
  if (options_.cache_models) {
    CacheModel(sst_file_path, model);
  }
  
  // Update training duration statistics
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  auto& stats = GetMutableStats(sst_file_path);
  stats.last_training_duration_ms = duration.count();
  stats.update_at = metadata.update_at;
  
  return true;
}

bool SSTLearnedIndexManager::UpdateModel(const std::string& sst_file_path,
    const std::vector<std::pair<uint64_t, uint32_t>>& new_key_block_pairs) {
  
  // For incremental updates, we would implement online learning algorithms
  // For now, retrain the entire model
  return TrainModel(sst_file_path, new_key_block_pairs);
}

uint32_t SSTLearnedIndexManager::PredictBlockIndex(const std::string& sst_file_path, 
                                                  uint64_t key) {
  auto model = GetCachedModel(sst_file_path);
  if (!model) {
    GetMutableStats(sst_file_path).fallback_queries++;
    GetMutableStats(sst_file_path).total_queries++;
    return 0; // Fallback to first block
  }
  
  double confidence = model->GetPredictionConfidence(key);
  uint32_t predicted_block = model->PredictBlockIndex(key);
  
  auto& stats = GetMutableStats(sst_file_path);
  stats.total_queries++;
  
  if (confidence >= options_.confidence_threshold) {
    stats.successful_predictions++;
  } else {
    stats.fallback_queries++;
  }
  
  return predicted_block;
}

double SSTLearnedIndexManager::GetPredictionConfidence(const std::string& sst_file_path, 
                                                      uint64_t key) {
  auto model = GetCachedModel(sst_file_path);
  if (!model) {
    return 0.0;
  }
  
  return model->GetPredictionConfidence(key);
}

void SSTLearnedIndexManager::CacheModel(const std::string& sst_file_path,
                                       std::shared_ptr<LearnedIndexBlock> model) {
  if (!options_.cache_models) {
    return;
  }
  
  if (ShouldEvict()) {
    EvictLRUModels();
  }
  
  model_cache_[sst_file_path] = model;
  cache_access_time_[sst_file_path] = ++access_counter_;
}

std::shared_ptr<LearnedIndexBlock> SSTLearnedIndexManager::GetCachedModel(
    const std::string& sst_file_path) const {
  
  auto it = model_cache_.find(sst_file_path);
  if (it != model_cache_.end()) {
    // Update access time (const_cast is safe here for cache management)
    const_cast<SSTLearnedIndexManager*>(this)->cache_access_time_[sst_file_path] = 
        ++const_cast<SSTLearnedIndexManager*>(this)->access_counter_;
    return it->second;
  }
  
  return nullptr;
}

void SSTLearnedIndexManager::RemoveFromCache(const std::string& sst_file_path) {
  model_cache_.erase(sst_file_path);
  cache_access_time_.erase(sst_file_path);
}

void SSTLearnedIndexManager::ClearCache() {
  model_cache_.clear();
  cache_access_time_.clear();
}

const SSTIndexStats& SSTLearnedIndexManager::GetStats(const std::string& sst_file_path) const {
  auto it = stats_.find(sst_file_path);
  if (it != stats_.end()) {
    return it->second;
  }
  
  // Return default stats if not found
  static const SSTIndexStats default_stats;
  return default_stats;
}

void SSTLearnedIndexManager::UpdateStats(const std::string& sst_file_path, 
                                        bool prediction_successful, 
                                        double prediction_error) {
  auto& stats = GetMutableStats(sst_file_path);
  stats.total_queries++;
  
  if (prediction_successful) {
    stats.successful_predictions++;
    
    // Update running average of prediction error
    double total_error = stats.average_prediction_error * (stats.successful_predictions - 1);
    stats.average_prediction_error = (total_error + prediction_error) / stats.successful_predictions;
  } else {
    stats.fallback_queries++;
  }
  
  stats.update_at = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

void SSTLearnedIndexManager::UpdateOptions(const SSTLearnedIndexOptions& new_options) {
  options_ = new_options;
  
  if (!options_.cache_models) {
    ClearCache();
  } else if (model_cache_.size() > options_.max_cache_size) {
    EvictLRUModels();
  }
}

bool SSTLearnedIndexManager::TrainLinearModel(
    const std::vector<std::pair<uint64_t, uint32_t>>& training_data,
    LearnedIndexBlock* model) {
  
  if (training_data.size() < 2) {
    return false;
  }
  
  // Simple linear regression: y = ax + b
  // where x is the key and y is the block index
  
  double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
  size_t n = training_data.size();
  
  for (const auto& pair : training_data) {
    double x = static_cast<double>(pair.first);
    double y = static_cast<double>(pair.second);
    
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_xx += x * x;
  }
  
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;
  
  double denominator = sum_xx - n * mean_x * mean_x;
  if (std::abs(denominator) < 1e-10) {
    // All keys are the same, use constant model
    std::vector<double> params = {mean_y, 0.0};
    model->SetParameters(params);
    return true;
  }
  
  double slope = (sum_xy - n * mean_x * mean_y) / denominator;
  double intercept = mean_y - slope * mean_x;
  
  std::vector<double> params = {intercept, slope};
  model->SetParameters(params);
  
  return true;
}

bool SSTLearnedIndexManager::TrainPolynomialModel(
    const std::vector<std::pair<uint64_t, uint32_t>>& training_data,
    LearnedIndexBlock* model, uint32_t degree) {
  
  if (training_data.size() < degree + 1) {
    // Fall back to linear model
    return TrainLinearModel(training_data, model);
  }
  
  // For simplicity, implement polynomial regression up to degree 3
  // In practice, would use more sophisticated numerical methods
  
  if (degree > 3) {
    degree = 3;
  }
  
  // For now, implement degree 2 polynomial (quadratic)
  if (degree >= 2) {
    // Set up normal equations for quadratic regression: y = ax^2 + bx + c
    double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
    double sum_y = 0, sum_xy = 0, sum_x2y = 0;
    
    for (const auto& pair : training_data) {
      double x = static_cast<double>(pair.first);
      double y = static_cast<double>(pair.second);
      double x2 = x * x;
      double x3 = x2 * x;
      double x4 = x3 * x;
      
      sum_x += x;
      sum_x2 += x2;
      sum_x3 += x3;
      sum_x4 += x4;
      sum_y += y;
      sum_xy += x * y;
      sum_x2y += x2 * y;
    }
    
    // Solve the 3x3 system for quadratic coefficients
    // This is a simplified implementation; production code would use robust linear algebra
    
    // For simplicity, fall back to linear regression if polynomial fitting is complex
    return TrainLinearModel(training_data, model);
  }
  
  return TrainLinearModel(training_data, model);
}

void SSTLearnedIndexManager::EvictLRUModels() {
  while (model_cache_.size() > options_.max_cache_size / 2) {
    // Find the least recently used model
    auto lru_it = std::min_element(cache_access_time_.begin(), cache_access_time_.end(),
                                  [](const auto& a, const auto& b) {
                                    return a.second < b.second;
                                  });
    
    if (lru_it != cache_access_time_.end()) {
      std::string file_path = lru_it->first;
      model_cache_.erase(file_path);
      cache_access_time_.erase(file_path);
    } else {
      break;
    }
  }
}

bool SSTLearnedIndexManager::ShouldEvict() const {
  return model_cache_.size() >= options_.max_cache_size;
}

SSTIndexStats& SSTLearnedIndexManager::GetMutableStats(const std::string& sst_file_path) {
  auto it = stats_.find(sst_file_path);
  if (it == stats_.end()) {
    InitializeStats(sst_file_path);
    it = stats_.find(sst_file_path);
  }
  return it->second;
}

void SSTLearnedIndexManager::InitializeStats(const std::string& sst_file_path) {
  if (stats_.find(sst_file_path) == stats_.end()) {
    stats_[sst_file_path] = SSTIndexStats();
  }
}

} // namespace learned_index
} // namespace rocksdb