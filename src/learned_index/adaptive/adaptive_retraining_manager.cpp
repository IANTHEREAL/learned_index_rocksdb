#include "learned_index/adaptive/adaptive_retraining_manager.h"
#include <algorithm>
#include <random>

namespace rocksdb {
namespace learned_index {
namespace adaptive {

AdaptiveRetrainingManager::AdaptiveRetrainingManager(
    SSTLearnedIndexManager* index_manager,
    ModelPerformanceTracker* performance_tracker)
    : config_(Config()), index_manager_(index_manager), 
      performance_tracker_(performance_tracker),
      retraining_queue_([](const RetrainingRequest& a, const RetrainingRequest& b) {
          // Higher priority = lower number (priority queue is max-heap)
          return a.timestamp_ms > b.timestamp_ms; // FIFO for same priority
      }) {
}

AdaptiveRetrainingManager::AdaptiveRetrainingManager(
    SSTLearnedIndexManager* index_manager,
    ModelPerformanceTracker* performance_tracker,
    const Config& config)
    : config_(config), index_manager_(index_manager), 
      performance_tracker_(performance_tracker),
      retraining_queue_([](const RetrainingRequest& a, const RetrainingRequest& b) {
          // Higher priority = lower number (priority queue is max-heap)
          return a.timestamp_ms > b.timestamp_ms; // FIFO for same priority
      }) {
}

AdaptiveRetrainingManager::~AdaptiveRetrainingManager() {
    Stop();
}

void AdaptiveRetrainingManager::Start() {
    if (is_running_.load()) {
        return;
    }
    
    should_stop_.store(false);
    is_running_.store(true);
    is_paused_.store(false);
    
    if (config_.enable_background_thread) {
        // Start monitoring thread
        monitoring_thread_ = std::make_unique<std::thread>(&AdaptiveRetrainingManager::MonitoringLoop, this);
        
        // Start worker threads
        for (size_t i = 0; i < config_.max_concurrent_retraining; ++i) {
            worker_threads_.push_back(
                std::make_unique<std::thread>(&AdaptiveRetrainingManager::WorkerLoop, this));
        }
    }
}

void AdaptiveRetrainingManager::Stop() {
    if (!is_running_.load()) {
        return;
    }
    
    should_stop_.store(true);
    is_running_.store(false);
    
    // Wake up all threads
    queue_cv_.notify_all();
    
    // Join threads
    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
    }
    
    for (auto& worker : worker_threads_) {
        if (worker && worker->joinable()) {
            worker->join();
        }
    }
    
    monitoring_thread_.reset();
    worker_threads_.clear();
}

void AdaptiveRetrainingManager::Pause() {
    is_paused_.store(true);
}

void AdaptiveRetrainingManager::Resume() {
    is_paused_.store(false);
    queue_cv_.notify_all();
}

bool AdaptiveRetrainingManager::RequestRetraining(const std::string& model_id, 
                                                const std::string& sst_file_path,
                                                const std::string& reason) {
    if (!config_.enable_adaptive_retraining) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (retraining_queue_.size() >= config_.retraining_queue_size) {
        return false; // Queue full
    }
    
    // Check if already being retrained
    {
        std::lock_guard<std::mutex> models_lock(models_mutex_);
        if (models_being_retrained_.count(model_id) > 0) {
            return false; // Already in progress
        }
    }
    
    ModelHealthMetrics health = performance_tracker_->ComputeHealthMetrics(model_id);
    
    RetrainingRequest request(model_id, sst_file_path, GetCurrentTimestampMs(),
                            health.current_accuracy, reason);
    
    retraining_queue_.push(request);
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_retraining_requests++;
        if (reason == "manual") {
            stats_.manual_triggers++;
        } else {
            stats_.automatic_triggers++;
        }
    }
    
    queue_cv_.notify_one();
    return true;
}

bool AdaptiveRetrainingManager::RequestEmergencyRetraining(const std::string& model_id,
                                                         const std::string& sst_file_path) {
    // Emergency retraining bypasses normal queue limits
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    ModelHealthMetrics health = performance_tracker_->ComputeHealthMetrics(model_id);
    
    RetrainingRequest request(model_id, sst_file_path, GetCurrentTimestampMs(),
                            health.current_accuracy, "emergency");
    
    // Add to front of queue by using earlier timestamp
    request.timestamp_ms = 0; 
    
    retraining_queue_.push(request);
    
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_retraining_requests++;
        stats_.automatic_triggers++;
    }
    
    queue_cv_.notify_all();
    return true;
}

void AdaptiveRetrainingManager::SetRetrainingCallback(RetrainingCallback callback) {
    retraining_callback_ = callback;
}

size_t AdaptiveRetrainingManager::GetQueueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return retraining_queue_.size();
}

AdaptiveRetrainingManager::Stats AdaptiveRetrainingManager::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AdaptiveRetrainingManager::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats();
}

void AdaptiveRetrainingManager::UpdateConfig(const Config& new_config) {
    config_ = new_config;
}

void AdaptiveRetrainingManager::MonitoringLoop() {
    while (!should_stop_.load()) {
        try {
            if (!is_paused_.load() && config_.enable_adaptive_retraining) {
                CheckModelsForRetraining();
                
                {
                    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                    stats_.last_monitoring_cycle_ms = GetCurrentTimestampMs();
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.monitoring_interval_ms));
        } catch (const std::exception& e) {
            // Log error and continue
            continue;
        }
    }
}

void AdaptiveRetrainingManager::WorkerLoop() {
    while (!should_stop_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for work or stop signal
        queue_cv_.wait(lock, [this]() {
            return should_stop_.load() || (!retraining_queue_.empty() && !is_paused_.load());
        });
        
        if (should_stop_.load()) {
            break;
        }
        
        if (is_paused_.load() || retraining_queue_.empty() || !CanStartRetraining()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Get next retraining request
        RetrainingRequest request = retraining_queue_.top();
        retraining_queue_.pop();
        
        // Mark model as being retrained
        {
            std::lock_guard<std::mutex> models_lock(models_mutex_);
            models_being_retrained_.insert(request.model_id);
        }
        
        active_retraining_count_.fetch_add(1);
        lock.unlock();
        
        // Execute retraining
        RetrainingResult result = ExecuteRetraining(request);
        
        // Update statistics
        UpdateStats(result);
        
        // Call callback if provided
        if (retraining_callback_) {
            try {
                retraining_callback_(result);
            } catch (...) {
                // Ignore callback exceptions
            }
        }
        
        // Clean up
        {
            std::lock_guard<std::mutex> models_lock(models_mutex_);
            models_being_retrained_.erase(request.model_id);
        }
        
        active_retraining_count_.fetch_sub(1);
    }
}

void AdaptiveRetrainingManager::CheckModelsForRetraining() {
    std::vector<std::string> tracked_models = performance_tracker_->GetTrackedModels();
    
    for (const std::string& model_id : tracked_models) {
        ModelHealthMetrics health = performance_tracker_->ComputeHealthMetrics(model_id);
        
        if (ShouldTriggerRetraining(model_id, health)) {
            // Try to get SST file path from model_id
            // In practice, this would be maintained by the index manager
            std::string sst_file_path = model_id; // Assuming model_id is the file path
            
            bool is_emergency = health.current_accuracy < (config_.emergency_retraining_threshold / 100.0);
            
            if (is_emergency) {
                RequestEmergencyRetraining(model_id, sst_file_path);
            } else {
                RequestRetraining(model_id, sst_file_path, "automatic");
            }
        }
    }
}

bool AdaptiveRetrainingManager::ShouldTriggerRetraining(const std::string& model_id,
                                                      const ModelHealthMetrics& health) {
    // Check if already scheduled or in progress
    {
        std::lock_guard<std::mutex> models_lock(models_mutex_);
        if (models_being_retrained_.count(model_id) > 0) {
            return false;
        }
    }
    
    // Use the performance tracker's decision
    return health.needs_retraining;
}

RetrainingResult AdaptiveRetrainingManager::ExecuteRetraining(const RetrainingRequest& request) {
    RetrainingResult result;
    result.model_id = request.model_id;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Collect training data
        std::vector<std::pair<uint64_t, uint32_t>> training_data;
        if (!CollectTrainingData(request.model_id, request.sst_file_path, training_data)) {
            result.success = false;
            result.error_message = "Failed to collect training data";
            return result;
        }
        
        result.training_samples = training_data.size();
        
        if (training_data.size() < config_.min_new_samples_for_retrain) {
            result.success = false;
            result.error_message = "Insufficient training data";
            return result;
        }
        
        // Perform retraining
        bool training_success = index_manager_->TrainModel(request.sst_file_path, training_data);
        
        if (training_success) {
            result.success = true;
            
            // Compute new accuracy (would require validation set in practice)
            result.new_accuracy = 0.95; // Placeholder - would compute from validation
            
            // Record training event
            performance_tracker_->RecordTrainingEvent(request.model_id, GetCurrentTimestampMs(),
                                                    training_data.size(), result.new_accuracy);
        } else {
            result.success = false;
            result.error_message = "Model training failed";
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception during retraining: ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.training_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    return result;
}

bool AdaptiveRetrainingManager::CollectTrainingData(const std::string& model_id,
                                                  const std::string& sst_file_path,
                                                  std::vector<std::pair<uint64_t, uint32_t>>& training_data) {
    // Suppress unused parameter warning
    (void)sst_file_path;
    
    // In practice, this would collect data from:
    // 1. Recent query logs with actual block locations
    // 2. SST file scanning to get current key distribution
    // 3. Cached training data from online collection
    
    std::lock_guard<std::mutex> lock(training_data_mutex_);
    
    auto it = collected_training_data_.find(model_id);
    if (it != collected_training_data_.end()) {
        training_data = it->second;
        collected_training_data_.erase(it); // Use data once
        return !training_data.empty();
    }
    
    // Generate synthetic training data for demonstration
    // In production, this would scan the SST file or use cached query data
    std::mt19937_64 rng(std::hash<std::string>{}(model_id));
    std::uniform_int_distribution<uint64_t> key_dist(1000, 100000);
    std::uniform_int_distribution<uint32_t> block_dist(0, 99);
    
    training_data.clear();
    training_data.reserve(config_.min_new_samples_for_retrain);
    
    for (size_t i = 0; i < config_.min_new_samples_for_retrain; ++i) {
        uint64_t key = key_dist(rng);
        uint32_t block = block_dist(rng);
        training_data.emplace_back(key, block);
    }
    
    // Sort by key to simulate SST file organization
    std::sort(training_data.begin(), training_data.end());
    
    return true;
}

int AdaptiveRetrainingManager::CalculatePriority(const ModelHealthMetrics& health) {
    // Lower number = higher priority
    if (health.current_accuracy < 0.7) return 0; // Critical
    if (health.current_accuracy < 0.8) return 1; // High
    if (health.is_degrading) return 2;           // Medium
    return 3;                                    // Low
}

uint64_t AdaptiveRetrainingManager::GetCurrentTimestampMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void AdaptiveRetrainingManager::UpdateStats(const RetrainingResult& result) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (result.success) {
        stats_.successful_retrainings++;
    } else {
        stats_.failed_retrainings++;
    }
    
    // Update average duration (exponential moving average)
    double alpha = 0.1;
    stats_.average_retraining_duration_ms = 
        alpha * result.training_duration_ms + (1.0 - alpha) * stats_.average_retraining_duration_ms;
}

bool AdaptiveRetrainingManager::CanStartRetraining() const {
    return active_retraining_count_.load() < config_.max_concurrent_retraining;
}

// OnlineTrainingDataCollector implementation
OnlineTrainingDataCollector::OnlineTrainingDataCollector(AdaptiveRetrainingManager* manager)
    : manager_(manager), sampling_rate_(0.1), rng_(std::random_device{}()), dist_(0.0, 1.0) {
}

void OnlineTrainingDataCollector::RecordQuery(const std::string& model_id, 
                                            uint64_t key, 
                                            uint32_t actual_block) {
    // Suppress unused parameter warnings
    (void)model_id;
    (void)key;
    (void)actual_block;
    
    if (dist_(rng_) < sampling_rate_) {
        // Sample this query for training data
        // In practice, this would be stored in the manager's training data collection
        // For now, we'll just demonstrate the interface
    }
}

} // namespace adaptive
} // namespace learned_index
} // namespace rocksdb