#pragma once

#include "learned_index/adaptive/model_performance_tracker.h"
#include "learned_index/sst_learned_index_manager.h"
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <random>

namespace rocksdb {
namespace learned_index {
namespace adaptive {

struct RetrainingRequest {
    std::string model_id;
    std::string sst_file_path;
    uint64_t timestamp_ms;
    double current_accuracy;
    std::string trigger_reason;
    
    RetrainingRequest(const std::string& id, const std::string& path, 
                     uint64_t ts, double acc, const std::string& reason)
        : model_id(id), sst_file_path(path), timestamp_ms(ts), 
          current_accuracy(acc), trigger_reason(reason) {}
};

struct RetrainingResult {
    std::string model_id;
    bool success;
    double new_accuracy;
    size_t training_samples;
    uint64_t training_duration_ms;
    std::string error_message;
    
    RetrainingResult() : success(false), new_accuracy(0.0), training_samples(0),
                        training_duration_ms(0) {}
};

using RetrainingCallback = std::function<void(const RetrainingResult&)>;

class AdaptiveRetrainingManager {
public:
    struct Config {
        bool enable_adaptive_retraining = true;
        uint64_t monitoring_interval_ms = 30000;      // 30 seconds
        size_t max_concurrent_retraining = 2;          // Max parallel retraining jobs
        size_t retraining_queue_size = 100;
        bool enable_background_thread = true;
        bool enable_priority_retraining = true;        // Prioritize degraded models
        uint64_t emergency_retraining_threshold = 60000; // 1 minute for critical accuracy drops
        
        // Training data collection
        bool enable_online_data_collection = true;
        size_t min_new_samples_for_retrain = 1000;
        double sample_collection_ratio = 0.1;          // Collect 10% of queries as training data
        
        Config() = default;
    };

    AdaptiveRetrainingManager(SSTLearnedIndexManager* index_manager,
                            ModelPerformanceTracker* performance_tracker);
    AdaptiveRetrainingManager(SSTLearnedIndexManager* index_manager,
                            ModelPerformanceTracker* performance_tracker,
                            const Config& config);
    ~AdaptiveRetrainingManager();

    // Lifecycle management
    void Start();
    void Stop();
    void Pause();
    void Resume();
    
    // Manual retraining
    bool RequestRetraining(const std::string& model_id, const std::string& sst_file_path,
                          const std::string& reason = "manual");
    bool RequestEmergencyRetraining(const std::string& model_id, 
                                   const std::string& sst_file_path);
    
    // Callback management
    void SetRetrainingCallback(RetrainingCallback callback);
    
    // Status and monitoring
    bool IsRunning() const { return is_running_.load(); }
    size_t GetQueueSize() const;
    size_t GetActiveRetrainingCount() const { return active_retraining_count_.load(); }
    
    // Statistics
    struct Stats {
        uint64_t total_retraining_requests;
        uint64_t successful_retrainings;
        uint64_t failed_retrainings;
        uint64_t automatic_triggers;
        uint64_t manual_triggers;
        double average_retraining_duration_ms;
        uint64_t last_monitoring_cycle_ms;
        
        Stats() : total_retraining_requests(0), successful_retrainings(0),
                 failed_retrainings(0), automatic_triggers(0), manual_triggers(0),
                 average_retraining_duration_ms(0.0), last_monitoring_cycle_ms(0) {}
    };
    
    Stats GetStats() const;
    void ResetStats();
    
    // Configuration
    void UpdateConfig(const Config& new_config);
    const Config& GetConfig() const { return config_; }

private:
    Config config_;
    SSTLearnedIndexManager* index_manager_;
    ModelPerformanceTracker* performance_tracker_;
    
    // Threading
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_paused_{false};
    std::atomic<bool> should_stop_{false};
    std::unique_ptr<std::thread> monitoring_thread_;
    std::vector<std::unique_ptr<std::thread>> worker_threads_;
    
    // Synchronization
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    mutable std::mutex stats_mutex_;
    
    // Request queue
    std::priority_queue<RetrainingRequest, std::vector<RetrainingRequest>,
                       std::function<bool(const RetrainingRequest&, const RetrainingRequest&)>> 
        retraining_queue_;
    
    // State tracking
    std::atomic<size_t> active_retraining_count_{0};
    std::unordered_set<std::string> models_being_retrained_;
    std::mutex models_mutex_;
    
    // Callback
    RetrainingCallback retraining_callback_;
    
    // Statistics
    mutable Stats stats_;
    
    // Training data collection
    std::unordered_map<std::string, std::vector<std::pair<uint64_t, uint32_t>>> 
        collected_training_data_;
    std::mutex training_data_mutex_;
    
    // Main loop methods
    void MonitoringLoop();
    void WorkerLoop();
    
    // Monitoring logic
    void CheckModelsForRetraining();
    void ProcessRetrainingQueue();
    bool ShouldTriggerRetraining(const std::string& model_id, 
                                const ModelHealthMetrics& health);
    
    // Retraining execution
    RetrainingResult ExecuteRetraining(const RetrainingRequest& request);
    bool CollectTrainingData(const std::string& model_id, const std::string& sst_file_path,
                           std::vector<std::pair<uint64_t, uint32_t>>& training_data);
    
    // Priority calculation
    int CalculatePriority(const ModelHealthMetrics& health);
    
    // Utility methods
    uint64_t GetCurrentTimestampMs() const;
    void UpdateStats(const RetrainingResult& result);
    bool CanStartRetraining() const;
};

// Training data collector for online learning
class OnlineTrainingDataCollector {
public:
    OnlineTrainingDataCollector(AdaptiveRetrainingManager* manager);
    
    void RecordQuery(const std::string& model_id, uint64_t key, uint32_t actual_block);
    void SetSamplingRate(double rate) { sampling_rate_ = rate; }
    
private:
    AdaptiveRetrainingManager* manager_;
    double sampling_rate_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};

} // namespace adaptive
} // namespace learned_index
} // namespace rocksdb