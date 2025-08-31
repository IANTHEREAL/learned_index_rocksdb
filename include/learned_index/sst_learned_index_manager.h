#pragma once

#include "learned_index/learned_index_block.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

namespace learned_index {

// Statistics for SST file learned index performance
struct SSTLearnedIndexStats {
  uint64_t total_queries = 0;           // Total number of queries processed
  uint64_t successful_predictions = 0;  // Number of successful predictions
  uint64_t fallback_queries = 0;        // Number of queries that fell back to traditional index
  double average_prediction_error = 0.0; // Average prediction error in blocks
  uint64_t cache_hits = 0;              // Number of cache hits
  uint64_t cache_misses = 0;            // Number of cache misses
  uint64_t last_update_timestamp = 0;   // Last statistics update timestamp
  
  // Calculate prediction accuracy percentage
  double GetAccuracy() const {
    return total_queries > 0 ? (static_cast<double>(successful_predictions) / total_queries) * 100.0 : 0.0;
  }
  
  // Calculate cache hit rate percentage
  double GetCacheHitRate() const {
    uint64_t total_cache_requests = cache_hits + cache_misses;
    return total_cache_requests > 0 ? (static_cast<double>(cache_hits) / total_cache_requests) * 100.0 : 0.0;
  }
};

// Configuration options for SST learned index manager
struct SSTLearnedIndexOptions {
  bool enabled = true;                          // Enable learned indexes for SST files
  bool cache_models = true;                     // Cache loaded models in memory
  size_t max_cache_size = 1000;                // Maximum number of models to cache
  double confidence_threshold = 0.8;            // Minimum confidence for using predictions
  uint64_t max_prediction_error_blocks = 2;     // Maximum acceptable prediction error in blocks
  bool enable_fallback = true;                  // Enable fallback to traditional block index
  ModelType preferred_model_type = ModelType::LINEAR; // Preferred model type for new SST files
  
  // Performance tuning options
  bool enable_batch_predictions = true;         // Enable batch prediction optimization
  size_t max_batch_size = 100;                 // Maximum batch size for predictions
  uint64_t stats_update_interval_ms = 10000;   // Statistics update interval in milliseconds
};

// Cache entry for learned index models
struct CachedModel {
  std::unique_ptr<LearnedIndexBlock> model;
  uint64_t last_access_time;
  uint64_t access_count;
  std::string file_path;
  
  CachedModel(std::unique_ptr<LearnedIndexBlock> m, const std::string& path)
    : model(std::move(m)), last_access_time(0), access_count(0), file_path(path) {}
};

// Main SST file learned index manager
class SSTLearnedIndexManager {
public:
  explicit SSTLearnedIndexManager(const SSTLearnedIndexOptions& options = SSTLearnedIndexOptions{});
  ~SSTLearnedIndexManager();
  
  // Load learned index from SST file
  bool LoadLearnedIndex(const std::string& sst_file_path);
  
  // Create and train a new learned index for SST file
  bool CreateLearnedIndex(const std::string& sst_file_path, 
                         const std::vector<std::pair<uint64_t, uint32_t>>& key_block_pairs);
  
  // Predict block index for a given key in a specific SST file
  int PredictBlockIndex(const std::string& sst_file_path, uint64_t key);
  
  // Get prediction confidence for a given key
  double GetPredictionConfidence(const std::string& sst_file_path, uint64_t key);
  
  // Batch prediction for multiple keys
  std::vector<int> BatchPredictBlockIndices(const std::string& sst_file_path, 
                                           const std::vector<uint64_t>& keys);
  
  // Update learned index with new training data
  bool UpdateLearnedIndex(const std::string& sst_file_path,
                         const std::vector<std::pair<uint64_t, uint32_t>>& additional_data);
  
  // Remove learned index from cache and storage
  void RemoveLearnedIndex(const std::string& sst_file_path);
  
  // Get statistics for a specific SST file
  SSTLearnedIndexStats GetStats(const std::string& sst_file_path) const;
  
  // Get aggregated statistics across all SST files
  SSTLearnedIndexStats GetAggregatedStats() const;
  
  // Clear cache and reset all statistics
  void ClearCache();
  
  // Update configuration options
  void UpdateOptions(const SSTLearnedIndexOptions& new_options);
  
  // Check if learned index is available for SST file
  bool HasLearnedIndex(const std::string& sst_file_path) const;
  
  // Get current cache size
  size_t GetCacheSize() const;
  
private:
  // Internal helper methods
  std::unique_ptr<LearnedIndexBlock> LoadModelFromFile(const std::string& sst_file_path);
  bool SaveModelToFile(const std::string& sst_file_path, const LearnedIndexBlock& model);
  void EvictLeastRecentlyUsed();
  void UpdateStats(const std::string& sst_file_path, bool prediction_successful, double error);
  std::string GetModelFilePath(const std::string& sst_file_path) const;
  
  // Train a linear model from key-block pairs
  std::unique_ptr<LearnedIndexBlock> TrainLinearModel(
    const std::vector<std::pair<uint64_t, uint32_t>>& training_data);
  
  // Validate prediction accuracy
  bool ValidatePrediction(int predicted_block, uint32_t actual_block) const;
  
  // Thread-safe access to cache and statistics
  mutable std::mutex cache_mutex_;
  mutable std::mutex stats_mutex_;
  
  // Configuration
  SSTLearnedIndexOptions options_;
  
  // Model cache: file_path -> CachedModel
  std::unordered_map<std::string, std::unique_ptr<CachedModel>> model_cache_;
  
  // Statistics per SST file: file_path -> SSTLearnedIndexStats
  mutable std::unordered_map<std::string, SSTLearnedIndexStats> file_stats_;
  
  // Access order for LRU eviction
  std::vector<std::string> access_order_;
};

} // namespace learned_index