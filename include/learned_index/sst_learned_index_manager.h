#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "learned_index_block.h"
#include "ml_model.h"

namespace rocksdb {
namespace learned_index {

// Configuration options for SST file learned indexes
struct SSTLearnedIndexOptions {
    bool enabled = true;
    ModelType model_type = ModelType::LINEAR;
    double confidence_threshold = 0.8;
    uint64_t max_prediction_error_bytes = 4096;
    size_t min_training_samples = 100;
    bool enable_block_predictions = true;
    size_t max_cache_size = 1000;
    
    SSTLearnedIndexOptions() = default;
};

// Statistics for SST file learned index
struct SSTLearnedIndexStats {
    uint64_t total_queries = 0;
    uint64_t successful_predictions = 0;
    uint64_t fallback_queries = 0;
    double average_prediction_error = 0.0;
    uint64_t last_training_duration_ms = 0;
    uint64_t model_size_bytes = 0;
    uint64_t update_at = 0;
    
    double GetSuccessRate() const {
        return total_queries > 0 ? 
               static_cast<double>(successful_predictions) / total_queries : 0.0;
    }
    
    double GetFallbackRate() const {
        return total_queries > 0 ? 
               static_cast<double>(fallback_queries) / total_queries : 0.0;
    }
};

// Key range information for training
struct KeyRange {
    uint64_t start_key;
    uint64_t end_key;
    uint32_t block_index;
    size_t key_count;
    
    KeyRange(uint64_t start, uint64_t end, uint32_t idx, size_t count)
        : start_key(start), end_key(end), block_index(idx), key_count(count) {}
};

// Manages learned indexes for individual SST files
class SSTLearnedIndexManager {
private:
    std::unique_ptr<MLModel> model_;
    SSTLearnedIndexOptions options_;
    SSTLearnedIndexStats stats_;
    std::vector<BlockPrediction> block_predictions_;
    std::string sst_file_name_;
    uint64_t file_size_;
    bool is_trained_;
    
    // Cache for recent predictions
    mutable std::unordered_map<uint64_t, uint32_t> prediction_cache_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;

public:
    explicit SSTLearnedIndexManager(const SSTLearnedIndexOptions& options);
    
    ~SSTLearnedIndexManager() = default;
    
    // Initialize with SST file information
    bool Initialize(const std::string& file_name, uint64_t file_size);
    
    // Train the model using key ranges from SST file
    bool TrainModel(const std::vector<KeyRange>& key_ranges);
    
    // Load pre-trained model from LearnedIndexBlock
    bool LoadModel(const LearnedIndexBlock& block);
    
    // Save current model to LearnedIndexBlock
    bool SaveModel(LearnedIndexBlock* block) const;
    
    // Predict which block contains the given key
    uint32_t PredictBlock(uint64_t key, double* confidence = nullptr) const;
    
    // Get block prediction with confidence
    bool GetBlockPrediction(uint64_t key, uint32_t* block_index, 
                           double* confidence) const;
    
    // Update statistics after a query
    void UpdateStats(uint64_t key, uint32_t actual_block, 
                    uint32_t predicted_block, bool fallback_used);
    
    // Get current statistics
    const SSTLearnedIndexStats& GetStats() const { return stats_; }
    
    // Check if the model is ready for predictions
    bool IsModelReady() const { return is_trained_ && model_ && model_->IsValid(); }
    
    // Get model information
    ModelType GetModelType() const;
    size_t GetModelSize() const;
    double GetTrainingAccuracy() const;
    
    // Cache management
    void ClearPredictionCache();
    size_t GetCacheSize() const { return prediction_cache_.size(); }
    double GetCacheHitRate() const;
    
    // Validation and diagnostics
    bool ValidateModel() const;
    std::string GetDiagnosticsInfo() const;

private:
    // Convert key to feature vector
    std::vector<double> KeyToFeatures(uint64_t key) const;
    
    // Generate block predictions during training
    void GenerateBlockPredictions();
    
    // Update prediction cache
    void UpdatePredictionCache(uint64_t key, uint32_t block_index) const;
    
    // Check cache for prediction
    bool GetFromCache(uint64_t key, uint32_t* block_index) const;
    
    // Calculate prediction error
    double CalculatePredictionError(uint32_t predicted, uint32_t actual) const;
};

} // namespace learned_index
} // namespace rocksdb