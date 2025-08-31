#include "learned_index/sst_learned_index_manager.h"
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace learned_index {

SSTLearnedIndexManager::SSTLearnedIndexManager(const SSTLearnedIndexOptions& options)
  : options_(options) {
  model_cache_.reserve(options_.max_cache_size);
}

SSTLearnedIndexManager::~SSTLearnedIndexManager() {
  ClearCache();
}

bool SSTLearnedIndexManager::LoadLearnedIndex(const std::string& sst_file_path) {
  if (!options_.enabled) {
    return false;
  }
  
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  // Check if already cached
  auto it = model_cache_.find(sst_file_path);
  if (it != model_cache_.end()) {
    // Update access time
    it->second->last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    it->second->access_count++;
    return true;
  }
  
  // Load from file
  auto model = LoadModelFromFile(sst_file_path);
  if (!model || !model->IsValid()) {
    return false;
  }
  
  // Add to cache
  if (model_cache_.size() >= options_.max_cache_size) {
    EvictLeastRecentlyUsed();
  }
  
  auto cached_model = std::make_unique<CachedModel>(std::move(model), sst_file_path);
  cached_model->last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  cached_model->access_count = 1;
  
  model_cache_[sst_file_path] = std::move(cached_model);
  access_order_.push_back(sst_file_path);
  
  return true;
}

bool SSTLearnedIndexManager::CreateLearnedIndex(
    const std::string& sst_file_path,
    const std::vector<std::pair<uint64_t, uint32_t>>& key_block_pairs) {
  
  if (!options_.enabled || key_block_pairs.empty()) {
    return false;
  }
  
  // Train model based on preferred type
  std::unique_ptr<LearnedIndexBlock> model;
  switch (options_.preferred_model_type) {
    case ModelType::LINEAR:
      model = TrainLinearModel(key_block_pairs);
      break;
    default:
      // For now, fallback to linear model
      model = TrainLinearModel(key_block_pairs);
      break;
  }
  
  if (!model || !model->IsValid()) {
    return false;
  }
  
  // Save model to file
  if (!SaveModelToFile(sst_file_path, *model)) {
    return false;
  }
  
  // Add to cache
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (model_cache_.size() >= options_.max_cache_size) {
    EvictLeastRecentlyUsed();
  }
  
  auto cached_model = std::make_unique<CachedModel>(std::move(model), sst_file_path);
  cached_model->last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  cached_model->access_count = 1;
  
  model_cache_[sst_file_path] = std::move(cached_model);
  access_order_.push_back(sst_file_path);
  
  return true;
}

int SSTLearnedIndexManager::PredictBlockIndex(const std::string& sst_file_path, uint64_t key) {
  if (!options_.enabled) {
    return -1;
  }
  
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  auto it = model_cache_.find(sst_file_path);
  if (it == model_cache_.end()) {
    // Try to load the model
    lock.~lock_guard();
    if (!LoadLearnedIndex(sst_file_path)) {
      UpdateStats(sst_file_path, false, 0.0);
      return -1;
    }
    std::lock_guard<std::mutex> new_lock(cache_mutex_);
    it = model_cache_.find(sst_file_path);
    if (it == model_cache_.end()) {
      UpdateStats(sst_file_path, false, 0.0);
      return -1;
    }
  }
  
  // Update access statistics
  it->second->last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  it->second->access_count++;
  
  // Get prediction
  int predicted_block = it->second->model->PredictBlockIndex(key);
  double confidence = it->second->model->GetPredictionConfidence(key);
  
  // Check confidence threshold
  if (confidence < options_.confidence_threshold) {
    UpdateStats(sst_file_path, false, 0.0);
    return -1; // Use fallback
  }
  
  UpdateStats(sst_file_path, true, 0.0); // Error calculation would need actual block info
  return predicted_block;
}

double SSTLearnedIndexManager::GetPredictionConfidence(const std::string& sst_file_path, uint64_t key) {
  if (!options_.enabled) {
    return 0.0;
  }
  
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  auto it = model_cache_.find(sst_file_path);
  if (it == model_cache_.end()) {
    return 0.0;
  }
  
  return it->second->model->GetPredictionConfidence(key);
}

std::vector<int> SSTLearnedIndexManager::BatchPredictBlockIndices(
    const std::string& sst_file_path,
    const std::vector<uint64_t>& keys) {
  
  std::vector<int> results;
  results.reserve(keys.size());
  
  if (!options_.enabled || !options_.enable_batch_predictions) {
    // Fallback to individual predictions
    for (uint64_t key : keys) {
      results.push_back(PredictBlockIndex(sst_file_path, key));
    }
    return results;
  }
  
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  auto it = model_cache_.find(sst_file_path);
  if (it == model_cache_.end()) {
    // Return invalid predictions
    results.assign(keys.size(), -1);
    return results;
  }
  
  // Batch processing
  for (uint64_t key : keys) {
    int predicted_block = it->second->model->PredictBlockIndex(key);
    double confidence = it->second->model->GetPredictionConfidence(key);
    
    if (confidence >= options_.confidence_threshold) {
      results.push_back(predicted_block);
      UpdateStats(sst_file_path, true, 0.0);
    } else {
      results.push_back(-1);
      UpdateStats(sst_file_path, false, 0.0);
    }
  }
  
  // Update access statistics
  it->second->last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  it->second->access_count += keys.size();
  
  return results;
}

bool SSTLearnedIndexManager::UpdateLearnedIndex(
    const std::string& sst_file_path,
    const std::vector<std::pair<uint64_t, uint32_t>>& additional_data) {
  
  // For now, retrain the entire model with additional data
  // In a production implementation, this would use incremental learning
  return CreateLearnedIndex(sst_file_path, additional_data);
}

void SSTLearnedIndexManager::RemoveLearnedIndex(const std::string& sst_file_path) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  model_cache_.erase(sst_file_path);
  
  // Remove from access order
  auto it = std::find(access_order_.begin(), access_order_.end(), sst_file_path);
  if (it != access_order_.end()) {
    access_order_.erase(it);
  }
  
  // Remove statistics
  std::lock_guard<std::mutex> stats_lock(stats_mutex_);
  file_stats_.erase(sst_file_path);
}

SSTLearnedIndexStats SSTLearnedIndexManager::GetStats(const std::string& sst_file_path) const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  
  auto it = file_stats_.find(sst_file_path);
  if (it != file_stats_.end()) {
    return it->second;
  }
  
  return SSTLearnedIndexStats{};
}

SSTLearnedIndexStats SSTLearnedIndexManager::GetAggregatedStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  
  SSTLearnedIndexStats aggregated;
  
  for (const auto& pair : file_stats_) {
    const auto& stats = pair.second;
    aggregated.total_queries += stats.total_queries;
    aggregated.successful_predictions += stats.successful_predictions;
    aggregated.fallback_queries += stats.fallback_queries;
    aggregated.cache_hits += stats.cache_hits;
    aggregated.cache_misses += stats.cache_misses;
  }
  
  // Calculate weighted average of prediction errors
  if (!file_stats_.empty()) {
    double total_error = 0.0;
    uint64_t total_weight = 0;
    
    for (const auto& pair : file_stats_) {
      const auto& stats = pair.second;
      if (stats.total_queries > 0) {
        total_error += stats.average_prediction_error * stats.total_queries;
        total_weight += stats.total_queries;
      }
    }
    
    if (total_weight > 0) {
      aggregated.average_prediction_error = total_error / total_weight;
    }
  }
  
  aggregated.last_update_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  
  return aggregated;
}

void SSTLearnedIndexManager::ClearCache() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);
  std::lock_guard<std::mutex> stats_lock(stats_mutex_);
  
  model_cache_.clear();
  access_order_.clear();
  file_stats_.clear();
}

void SSTLearnedIndexManager::UpdateOptions(const SSTLearnedIndexOptions& new_options) {
  options_ = new_options;
  
  // If cache size was reduced, evict excess entries
  std::lock_guard<std::mutex> lock(cache_mutex_);
  while (model_cache_.size() > options_.max_cache_size) {
    EvictLeastRecentlyUsed();
  }
}

bool SSTLearnedIndexManager::HasLearnedIndex(const std::string& sst_file_path) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return model_cache_.find(sst_file_path) != model_cache_.end();
}

size_t SSTLearnedIndexManager::GetCacheSize() const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return model_cache_.size();
}

// Private helper methods

std::unique_ptr<LearnedIndexBlock> SSTLearnedIndexManager::LoadModelFromFile(const std::string& sst_file_path) {
  std::string model_file_path = GetModelFilePath(sst_file_path);
  
  std::ifstream file(model_file_path, std::ios::binary);
  if (!file.is_open()) {
    return nullptr;
  }
  
  std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();
  
  auto model = std::make_unique<LearnedIndexBlock>();
  if (!model->Deserialize(data)) {
    return nullptr;
  }
  
  return model;
}

bool SSTLearnedIndexManager::SaveModelToFile(const std::string& sst_file_path, const LearnedIndexBlock& model) {
  std::string model_file_path = GetModelFilePath(sst_file_path);
  
  std::ofstream file(model_file_path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  
  std::string serialized_data = model.Serialize();
  file.write(serialized_data.data(), serialized_data.size());
  file.close();
  
  return !file.fail();
}

void SSTLearnedIndexManager::EvictLeastRecentlyUsed() {
  if (access_order_.empty()) {
    return;
  }
  
  // Find least recently used entry
  std::string lru_file_path = access_order_.front();
  uint64_t oldest_time = model_cache_[lru_file_path]->last_access_time;
  
  for (const std::string& file_path : access_order_) {
    auto it = model_cache_.find(file_path);
    if (it != model_cache_.end() && it->second->last_access_time < oldest_time) {
      oldest_time = it->second->last_access_time;
      lru_file_path = file_path;
    }
  }
  
  // Remove LRU entry
  model_cache_.erase(lru_file_path);
  auto it = std::find(access_order_.begin(), access_order_.end(), lru_file_path);
  if (it != access_order_.end()) {
    access_order_.erase(it);
  }
}

void SSTLearnedIndexManager::UpdateStats(const std::string& sst_file_path, bool prediction_successful, double error) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  
  auto& stats = file_stats_[sst_file_path];
  stats.total_queries++;
  
  if (prediction_successful) {
    stats.successful_predictions++;
    
    // Update rolling average of prediction error
    if (stats.total_queries == 1) {
      stats.average_prediction_error = error;
    } else {
      stats.average_prediction_error = 
        (stats.average_prediction_error * (stats.total_queries - 1) + error) / stats.total_queries;
    }
  } else {
    stats.fallback_queries++;
  }
  
  stats.last_update_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string SSTLearnedIndexManager::GetModelFilePath(const std::string& sst_file_path) const {
  return sst_file_path + ".lidx"; // Learned Index extension
}

std::unique_ptr<LearnedIndexBlock> SSTLearnedIndexManager::TrainLinearModel(
    const std::vector<std::pair<uint64_t, uint32_t>>& training_data) {
  
  if (training_data.size() < 2) {
    return nullptr;
  }
  
  auto model = std::make_unique<LearnedIndexBlock>();
  model->model_type = ModelType::LINEAR;
  model->feature_dimensions = 1;
  
  // Linear regression: y = ax + b where y is block index, x is key
  double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
  size_t n = training_data.size();
  
  for (const auto& pair : training_data) {
    double x = static_cast<double>(pair.first);  // key
    double y = static_cast<double>(pair.second); // block index
    
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_x2 += x * x;
  }
  
  // Calculate slope and intercept
  double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
  double intercept = (sum_y - slope * sum_x) / n;
  
  model->parameters = {slope, intercept};
  model->parameter_count = 2;
  
  // Calculate training accuracy
  double total_error = 0.0;
  for (const auto& pair : training_data) {
    double predicted = slope * static_cast<double>(pair.first) + intercept;
    double actual = static_cast<double>(pair.second);
    total_error += std::abs(predicted - actual);
    
    // Add block prediction
    BlockPrediction block_pred;
    block_pred.block_index = pair.second;
    block_pred.predicted_start_key = pair.first;
    block_pred.predicted_end_key = pair.first; // Will be updated with actual ranges
    block_pred.confidence = std::max(0.0, 1.0 - std::abs(predicted - actual) / 10.0); // Simple confidence
    
    model->AddBlockPrediction(block_pred);
  }
  
  model->metadata.training_samples = n;
  model->metadata.training_accuracy = std::max(0.0, 1.0 - (total_error / n) / 10.0);
  model->metadata.validation_accuracy = model->metadata.training_accuracy; // For simplicity
  model->metadata.training_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  model->metadata.last_update_timestamp = model->metadata.training_timestamp;
  
  model->UpdateChecksum();
  
  return model;
}

bool SSTLearnedIndexManager::ValidatePrediction(int predicted_block, uint32_t actual_block) const {
  if (predicted_block < 0) {
    return false;
  }
  
  uint64_t error = std::abs(static_cast<int64_t>(predicted_block) - static_cast<int64_t>(actual_block));
  return error <= options_.max_prediction_error_blocks;
}

} // namespace learned_index