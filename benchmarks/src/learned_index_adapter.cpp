#include "../include/learned_index_adapter.h"
#include <algorithm>

namespace benchmark {

LearnedIndexAdapter::LearnedIndexAdapter() : sst_file_path_("benchmark_sst_file") {
    // Configure learned index options for benchmarking
    options_.enable_learned_index = true;
    options_.default_model_type = rocksdb::learned_index::ModelType::kLinear;
    options_.confidence_threshold = 0.8;
    options_.max_prediction_error_bytes = 4096;
    options_.cache_models = true;
    options_.max_cache_size = 100;
    
    manager_ = std::make_unique<rocksdb::learned_index::SSTLearnedIndexManager>(options_);
}

bool LearnedIndexAdapter::Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) {
    training_data_ = training_data;
    
    // Sort training data to simulate SST file organization
    std::sort(training_data_.begin(), training_data_.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });
    
    return manager_->TrainModel(sst_file_path_, training_data_);
}

uint32_t LearnedIndexAdapter::Lookup(uint64_t key) {
    // Use the learned index manager to predict the block
    uint32_t predicted_block = manager_->PredictBlockIndex(sst_file_path_, key);
    
    // In a real implementation, we would use this prediction to guide SST file block access
    // For benchmarking, we simulate the behavior by checking if the prediction is reasonable
    
    double confidence = manager_->GetPredictionConfidence(sst_file_path_, key);
    
    // If confidence is too low, we might fall back to binary search
    if (confidence < options_.confidence_threshold) {
        // Simulate fallback to traditional block index (binary search)
        auto it = std::lower_bound(training_data_.begin(), training_data_.end(),
                                  std::make_pair(key, 0U),
                                  [](const auto& a, const auto& b) {
                                      return a.first < b.first;
                                  });
        
        if (it != training_data_.end() && it->first == key) {
            return it->second;
        }
    }
    
    return predicted_block;
}

size_t LearnedIndexAdapter::GetMemoryUsage() const {
    size_t manager_memory = CalculateManagerMemoryUsage();
    size_t training_data_memory = training_data_.size() * sizeof(std::pair<uint64_t, uint32_t>);
    return manager_memory + training_data_memory + sizeof(LearnedIndexAdapter);
}

void LearnedIndexAdapter::GetStats(BenchmarkResult& result) const {
    const auto& stats = manager_->GetStats(sst_file_path_);
    
    result.successful_predictions = stats.successful_predictions;
    result.total_predictions = stats.total_queries;
    result.prediction_accuracy = stats.GetSuccessRate();
    result.fallback_rate = stats.GetFallbackRate();
}

size_t LearnedIndexAdapter::CalculateManagerMemoryUsage() const {
    // Estimate memory usage of the learned index manager
    // This includes the cached model and statistics
    
    size_t base_size = sizeof(rocksdb::learned_index::SSTLearnedIndexManager);
    
    // Estimate model size (approximately what we've seen in examples)
    size_t model_size = 200; // bytes per model
    
    // Estimate statistics size
    size_t stats_size = sizeof(rocksdb::learned_index::SSTIndexStats);
    
    // Estimate cache overhead
    size_t cache_overhead = 100; // bytes for cache management structures
    
    return base_size + model_size + stats_size + cache_overhead;
}

} // namespace benchmark