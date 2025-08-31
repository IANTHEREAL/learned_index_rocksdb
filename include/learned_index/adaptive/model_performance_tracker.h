#pragma once

#include <chrono>
#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>

namespace rocksdb {
namespace learned_index {
namespace adaptive {

struct PredictionEvent {
    uint64_t timestamp_ms;
    uint64_t key;
    uint32_t predicted_block;
    uint32_t actual_block;
    double confidence;
    bool was_correct;
    double prediction_error_bytes;
    
    PredictionEvent() : timestamp_ms(0), key(0), predicted_block(0), 
                       actual_block(0), confidence(0.0), was_correct(false),
                       prediction_error_bytes(0.0) {}
};

struct WindowedMetrics {
    uint64_t window_start_ms;
    uint64_t window_end_ms;
    size_t total_predictions;
    size_t correct_predictions;
    double accuracy_rate;
    double average_confidence;
    double average_error_bytes;
    double p95_latency_us;
    double throughput_qps;
    
    WindowedMetrics() : window_start_ms(0), window_end_ms(0), total_predictions(0),
                       correct_predictions(0), accuracy_rate(0.0), average_confidence(0.0),
                       average_error_bytes(0.0), p95_latency_us(0.0), throughput_qps(0.0) {}
};

struct ModelHealthMetrics {
    std::string model_id;
    uint64_t last_training_timestamp_ms;
    uint64_t total_queries_served;
    double current_accuracy;
    double accuracy_trend_7d;      // 7-day accuracy trend
    double accuracy_trend_1h;      // 1-hour accuracy trend
    bool is_degrading;
    bool needs_retraining;
    uint64_t last_retrain_timestamp_ms;
    size_t retrain_count;
    
    ModelHealthMetrics() : last_training_timestamp_ms(0), total_queries_served(0),
                          current_accuracy(0.0), accuracy_trend_7d(0.0), accuracy_trend_1h(0.0),
                          is_degrading(false), needs_retraining(false), 
                          last_retrain_timestamp_ms(0), retrain_count(0) {}
};

class ModelPerformanceTracker {
public:
    struct Config {
        size_t max_events_per_window = 10000;
        uint64_t window_duration_ms = 60000;      // 1 minute windows
        size_t max_windows_stored = 1440;         // 24 hours of 1-min windows
        double accuracy_degradation_threshold = 0.05;  // 5% accuracy drop
        double minimum_accuracy_threshold = 0.85;      // 85% minimum accuracy
        size_t min_predictions_for_decision = 100;     // Min samples before retraining
        uint64_t min_time_between_retrains_ms = 300000; // 5 minutes between retrains
        bool enable_trend_analysis = true;
        
        Config() = default;
    };

    ModelPerformanceTracker();
    explicit ModelPerformanceTracker(const Config& config);
    ~ModelPerformanceTracker() = default;

    // Event recording
    void RecordPrediction(const std::string& model_id, const PredictionEvent& event);
    void RecordTrainingEvent(const std::string& model_id, uint64_t timestamp_ms, 
                           size_t training_samples, double training_accuracy);
    
    // Metrics computation
    WindowedMetrics ComputeCurrentMetrics(const std::string& model_id);
    WindowedMetrics ComputeWindowMetrics(const std::string& model_id, 
                                       uint64_t start_ms, uint64_t end_ms);
    ModelHealthMetrics ComputeHealthMetrics(const std::string& model_id);
    
    // Retraining decision
    bool ShouldRetrain(const std::string& model_id);
    std::vector<std::string> GetModelsNeedingRetrain();
    
    // Data access for dashboard
    std::vector<WindowedMetrics> GetHistoricalMetrics(const std::string& model_id, 
                                                     uint64_t start_ms, uint64_t end_ms);
    std::vector<std::string> GetTrackedModels() const;
    
    // Configuration
    void UpdateConfig(const Config& new_config);
    const Config& GetConfig() const { return config_; }
    
    // Cleanup
    void CleanupOldData();
    void Clear();

private:
    Config config_;
    mutable std::mutex mutex_;
    
    // Per-model data structures
    std::unordered_map<std::string, std::deque<PredictionEvent>> model_events_;
    std::unordered_map<std::string, std::deque<WindowedMetrics>> model_windows_;
    std::unordered_map<std::string, ModelHealthMetrics> model_health_;
    
    // Performance optimization
    std::unordered_map<std::string, uint64_t> last_window_computation_;
    
    // Helper methods
    void UpdateWindowedMetrics(const std::string& model_id);
    WindowedMetrics ComputeMetricsFromEvents(const std::deque<PredictionEvent>& events,
                                           uint64_t start_ms, uint64_t end_ms);
    double ComputeAccuracyTrend(const std::string& model_id, uint64_t duration_ms);
    bool IsAccuracyDegrading(const std::string& model_id);
    uint64_t GetCurrentTimestampMs() const;
    
    // Thread-safe accessors
    std::deque<PredictionEvent>& GetEventsForModel(const std::string& model_id);
    std::deque<WindowedMetrics>& GetWindowsForModel(const std::string& model_id);
    ModelHealthMetrics& GetHealthForModel(const std::string& model_id);
};

// Singleton manager for global access
class GlobalPerformanceTracker {
public:
    static ModelPerformanceTracker& GetInstance();
    static void Initialize(const ModelPerformanceTracker::Config& config);
    static void Shutdown();

private:
    static std::unique_ptr<ModelPerformanceTracker> instance_;
    static std::once_flag initialized_;
};

} // namespace adaptive
} // namespace learned_index
} // namespace rocksdb