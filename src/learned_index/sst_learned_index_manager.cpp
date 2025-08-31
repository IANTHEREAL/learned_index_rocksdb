#include "learned_index/sst_learned_index_manager.h"
#include <algorithm>
#include <chrono>
#include <sstream>

namespace rocksdb {
namespace learned_index {

SSTLearnedIndexManager::SSTLearnedIndexManager(const SSTLearnedIndexOptions& options)
    : options_(options)
    , file_size_(0)
    , is_trained_(false)
    , cache_hits_(0)
    , cache_misses_(0) {
    
    // Initialize model based on configuration
    model_ = MLModelFactory::CreateModel(options_.model_type, 1);
    
    // Initialize statistics
    stats_ = SSTLearnedIndexStats{};
    stats_.update_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool SSTLearnedIndexManager::Initialize(const std::string& file_name, uint64_t file_size) {
    if (file_name.empty() || file_size == 0) {
        return false;
    }
    
    sst_file_name_ = file_name;
    file_size_ = file_size;
    
    return true;
}

bool SSTLearnedIndexManager::TrainModel(const std::vector<KeyRange>& key_ranges) {
    if (!model_ || key_ranges.empty()) {
        return false;
    }
    
    // Check minimum training samples requirement
    size_t total_keys = 0;
    for (const auto& range : key_ranges) {
        total_keys += range.key_count;
    }
    
    if (total_keys < options_.min_training_samples) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare training data
    std::vector<std::vector<double>> features;
    std::vector<uint64_t> targets;
    
    features.reserve(key_ranges.size());
    targets.reserve(key_ranges.size());
    
    for (const auto& range : key_ranges) {
        // Use the middle of the key range as representative key
        uint64_t representative_key = (range.start_key + range.end_key) / 2;
        
        // Convert key to features (normalized position)
        std::vector<double> feature_vec = KeyToFeatures(representative_key);
        features.push_back(feature_vec);
        
        // Target is the block index
        targets.push_back(static_cast<uint64_t>(range.block_index));
    }
    
    // Train the model
    bool success = model_->Train(features, targets);
    if (!success) {
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Update statistics
    stats_.last_training_duration_ms = duration.count();
    stats_.update_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Generate block predictions
    GenerateBlockPredictions();
    
    is_trained_ = true;
    return true;
}

bool SSTLearnedIndexManager::LoadModel(const LearnedIndexBlock& block) {
    if (!block.IsValid()) {
        return false;
    }
    
    // Create model from block
    model_ = MLModelFactory::LoadModel(block);
    if (!model_) {
        return false;
    }
    
    // Load block predictions
    block_predictions_ = block.block_predictions;
    
    // Update statistics from metadata
    stats_.model_size_bytes = block.GetSerializedSize();
    stats_.update_at = block.metadata.update_at;
    
    is_trained_ = true;
    return true;
}

bool SSTLearnedIndexManager::SaveModel(LearnedIndexBlock* block) const {
    if (!block || !model_ || !is_trained_) {
        return false;
    }
    
    // Set basic information
    block->model_type = model_->GetType();
    block->feature_dimensions = static_cast<uint32_t>(model_->GetFeatureDimensions());
    block->parameters = model_->GetParameters();
    block->parameter_count = static_cast<uint32_t>(block->parameters.size());
    
    // Set metadata
    block->metadata.training_accuracy = model_->GetTrainingAccuracy();
    block->metadata.update_at = stats_.update_at;
    
    // Copy block predictions
    block->block_predictions = block_predictions_;
    
    // Update checksum
    block->UpdateChecksum();
    
    return block->IsValid();
}

uint32_t SSTLearnedIndexManager::PredictBlock(uint64_t key, double* confidence) const {
    if (!IsModelReady()) {
        if (confidence) {
            *confidence = 0.0;
        }
        return 0;
    }
    
    // Check cache first
    uint32_t cached_result;
    if (GetFromCache(key, &cached_result)) {
        if (confidence) {
            *confidence = model_->GetConfidence(KeyToFeatures(key));
        }
        return cached_result;
    }
    
    // Make prediction
    std::vector<double> features = KeyToFeatures(key);
    uint64_t predicted_block = model_->Predict(features);
    
    // Get confidence
    double pred_confidence = model_->GetConfidence(features);
    if (confidence) {
        *confidence = pred_confidence;
    }
    
    uint32_t result = static_cast<uint32_t>(predicted_block);
    
    // Update cache
    UpdatePredictionCache(key, result);
    
    return result;
}

bool SSTLearnedIndexManager::GetBlockPrediction(uint64_t key, uint32_t* block_index, 
                                               double* confidence) const {
    if (!block_index) {
        return false;
    }
    
    if (!IsModelReady()) {
        return false;
    }
    
    *block_index = PredictBlock(key, confidence);
    
    // Check if confidence meets threshold
    double conf = confidence ? *confidence : model_->GetConfidence(KeyToFeatures(key));
    
    return conf >= options_.confidence_threshold;
}

void SSTLearnedIndexManager::UpdateStats(uint64_t key, uint32_t actual_block, 
                                        uint32_t predicted_block, bool fallback_used) {
    stats_.total_queries++;
    
    if (fallback_used) {
        stats_.fallback_queries++;
    } else {
        double error = CalculatePredictionError(predicted_block, actual_block);
        
        if (error <= options_.max_prediction_error_bytes) {
            stats_.successful_predictions++;
        }
        
        // Update running average of prediction error
        double alpha = 0.1; // exponential moving average factor
        stats_.average_prediction_error = 
            alpha * error + (1.0 - alpha) * stats_.average_prediction_error;
    }
    
    stats_.update_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

ModelType SSTLearnedIndexManager::GetModelType() const {
    return model_ ? model_->GetType() : ModelType::LINEAR;
}

size_t SSTLearnedIndexManager::GetModelSize() const {
    if (!model_) {
        return 0;
    }
    
    // Estimate model size (parameters + overhead)
    return model_->GetParameterCount() * sizeof(double) + 
           block_predictions_.size() * sizeof(BlockPrediction) + 
           sizeof(ModelMetadata);
}

double SSTLearnedIndexManager::GetTrainingAccuracy() const {
    return model_ ? model_->GetTrainingAccuracy() : 0.0;
}

void SSTLearnedIndexManager::ClearPredictionCache() {
    prediction_cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
}

double SSTLearnedIndexManager::GetCacheHitRate() const {
    size_t total_requests = cache_hits_ + cache_misses_;
    return total_requests > 0 ? static_cast<double>(cache_hits_) / total_requests : 0.0;
}

bool SSTLearnedIndexManager::ValidateModel() const {
    if (!model_) {
        return false;
    }
    
    return model_->IsValid() && is_trained_;
}

std::string SSTLearnedIndexManager::GetDiagnosticsInfo() const {
    std::ostringstream oss;
    
    oss << "SST Learned Index Manager Diagnostics:\n";
    oss << "  File: " << sst_file_name_ << "\n";
    oss << "  File Size: " << file_size_ << " bytes\n";
    oss << "  Model Type: " << static_cast<int>(GetModelType()) << "\n";
    oss << "  Model Size: " << GetModelSize() << " bytes\n";
    oss << "  Is Trained: " << (is_trained_ ? "Yes" : "No") << "\n";
    oss << "  Training Accuracy: " << GetTrainingAccuracy() << "\n";
    oss << "  Total Queries: " << stats_.total_queries << "\n";
    oss << "  Success Rate: " << stats_.GetSuccessRate() * 100.0 << "%\n";
    oss << "  Fallback Rate: " << stats_.GetFallbackRate() * 100.0 << "%\n";
    oss << "  Average Error: " << stats_.average_prediction_error << " bytes\n";
    oss << "  Cache Hit Rate: " << GetCacheHitRate() * 100.0 << "%\n";
    oss << "  Cache Size: " << prediction_cache_.size() << " entries\n";
    
    return oss.str();
}

std::vector<double> SSTLearnedIndexManager::KeyToFeatures(uint64_t key) const {
    // Simple feature: normalized key position within file
    // More sophisticated features could include key hash, temporal features, etc.
    
    std::vector<double> features(1);
    
    if (file_size_ > 0) {
        // Normalize key to [0, 1] range based on file size
        features[0] = static_cast<double>(key) / static_cast<double>(file_size_);
    } else {
        features[0] = 0.0;
    }
    
    return features;
}

void SSTLearnedIndexManager::GenerateBlockPredictions() {
    if (!model_ || !is_trained_) {
        return;
    }
    
    block_predictions_.clear();
    
    // Generate predictions for a representative set of keys
    // This is a simplified implementation - a real implementation would
    // use the actual block structure of the SST file
    
    const size_t num_predictions = 100; // Sample 100 points
    for (size_t i = 0; i < num_predictions; ++i) {
        uint64_t sample_key = (file_size_ * i) / num_predictions;
        std::vector<double> features = KeyToFeatures(sample_key);
        
        uint64_t predicted_block = model_->Predict(features);
        double confidence = model_->GetConfidence(features);
        
        BlockPrediction prediction(
            static_cast<uint32_t>(predicted_block),
            sample_key,
            sample_key + (file_size_ / num_predictions),
            confidence
        );
        
        block_predictions_.push_back(prediction);
    }
}

void SSTLearnedIndexManager::UpdatePredictionCache(uint64_t key, uint32_t block_index) const {
    if (prediction_cache_.size() >= options_.max_cache_size) {
        // Simple cache eviction - remove oldest entry
        // In a production implementation, this could use LRU or other strategies
        auto oldest = prediction_cache_.begin();
        prediction_cache_.erase(oldest);
    }
    
    prediction_cache_[key] = block_index;
}

bool SSTLearnedIndexManager::GetFromCache(uint64_t key, uint32_t* block_index) const {
    auto it = prediction_cache_.find(key);
    if (it != prediction_cache_.end()) {
        *block_index = it->second;
        cache_hits_++;
        return true;
    }
    
    cache_misses_++;
    return false;
}

double SSTLearnedIndexManager::CalculatePredictionError(uint32_t predicted, uint32_t actual) const {
    // Calculate error in terms of block distance
    return std::abs(static_cast<double>(predicted) - static_cast<double>(actual));
}

} // namespace learned_index
} // namespace rocksdb