#pragma once

#include "learned_index/learned_index_block.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace rocksdb {
namespace learned_index {

struct SSTLearnedIndexOptions {
  bool enable_learned_index = true;
  ModelType default_model_type = ModelType::kLinear;
  double confidence_threshold = 0.8;
  uint64_t max_prediction_error_bytes = 4096;
  bool cache_models = true;
  size_t max_cache_size = 1000;
  
  SSTLearnedIndexOptions() = default;
};

struct SSTIndexStats {
  uint64_t total_queries = 0;
  uint64_t successful_predictions = 0;
  uint64_t fallback_queries = 0;
  double average_prediction_error = 0.0;
  uint64_t last_training_duration_ms = 0;
  uint64_t update_at = 0;
  
  double GetSuccessRate() const {
    return total_queries > 0 ? static_cast<double>(successful_predictions) / total_queries : 0.0;
  }
  
  double GetFallbackRate() const {
    return total_queries > 0 ? static_cast<double>(fallback_queries) / total_queries : 0.0;
  }
};

class SSTLearnedIndexManager {
public:
  explicit SSTLearnedIndexManager(const SSTLearnedIndexOptions& options);
  ~SSTLearnedIndexManager() = default;

  // Model management
  bool LoadLearnedIndex(const std::string& sst_file_path, const std::string& index_data);
  bool SaveLearnedIndex(const std::string& sst_file_path, std::string* index_data) const;
  
  // Training
  bool TrainModel(const std::string& sst_file_path, 
                 const std::vector<std::pair<uint64_t, uint32_t>>& key_block_pairs);
  bool UpdateModel(const std::string& sst_file_path,
                  const std::vector<std::pair<uint64_t, uint32_t>>& new_key_block_pairs);
  
  // Prediction
  uint32_t PredictBlockIndex(const std::string& sst_file_path, uint64_t key);
  double GetPredictionConfidence(const std::string& sst_file_path, uint64_t key);
  
  // Cache management
  void CacheModel(const std::string& sst_file_path, 
                 std::shared_ptr<LearnedIndexBlock> model);
  std::shared_ptr<LearnedIndexBlock> GetCachedModel(const std::string& sst_file_path) const;
  void RemoveFromCache(const std::string& sst_file_path);
  void ClearCache();
  
  // Statistics
  const SSTIndexStats& GetStats(const std::string& sst_file_path) const;
  void UpdateStats(const std::string& sst_file_path, bool prediction_successful, 
                  double prediction_error);
  
  // Configuration
  void UpdateOptions(const SSTLearnedIndexOptions& new_options);
  const SSTLearnedIndexOptions& GetOptions() const { return options_; }

private:
  SSTLearnedIndexOptions options_;
  
  // Model cache: file_path -> learned index block
  mutable std::unordered_map<std::string, std::shared_ptr<LearnedIndexBlock>> model_cache_;
  
  // Statistics per SST file
  mutable std::unordered_map<std::string, SSTIndexStats> stats_;
  
  // LRU cache management
  mutable std::unordered_map<std::string, uint64_t> cache_access_time_;
  mutable uint64_t access_counter_;
  
  // Training helpers
  bool TrainLinearModel(const std::vector<std::pair<uint64_t, uint32_t>>& training_data,
                       LearnedIndexBlock* model);
  bool TrainPolynomialModel(const std::vector<std::pair<uint64_t, uint32_t>>& training_data,
                           LearnedIndexBlock* model, uint32_t degree = 3);
  
  // Cache management helpers
  void EvictLRUModels();
  bool ShouldEvict() const;
  
  // Statistics helpers
  SSTIndexStats& GetMutableStats(const std::string& sst_file_path);
  void InitializeStats(const std::string& sst_file_path);
};

} // namespace learned_index
} // namespace rocksdb