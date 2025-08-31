#include "learned_index/adaptive/model_performance_tracker.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rocksdb {
namespace learned_index {
namespace adaptive {

ModelPerformanceTracker::ModelPerformanceTracker() 
    : config_(Config()) {
}

ModelPerformanceTracker::ModelPerformanceTracker(const Config& config) 
    : config_(config) {
}

void ModelPerformanceTracker::RecordPrediction(const std::string& model_id, 
                                             const PredictionEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& events = GetEventsForModel(model_id);
    events.push_back(event);
    
    // Limit event queue size
    while (events.size() > config_.max_events_per_window) {
        events.pop_front();
    }
    
    // Update health metrics
    auto& health = GetHealthForModel(model_id);
    health.total_queries_served++;
    
    // Trigger windowed metrics update if needed
    uint64_t current_time = GetCurrentTimestampMs();
    if (current_time - last_window_computation_[model_id] >= config_.window_duration_ms) {
        UpdateWindowedMetrics(model_id);
        last_window_computation_[model_id] = current_time;
    }
}

void ModelPerformanceTracker::RecordTrainingEvent(const std::string& model_id, 
                                                uint64_t timestamp_ms, 
                                                size_t training_samples, 
                                                double training_accuracy) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& health = GetHealthForModel(model_id);
    health.last_training_timestamp_ms = timestamp_ms;
    health.last_retrain_timestamp_ms = timestamp_ms;
    health.retrain_count++;
    
    // Use training parameters (avoid unused parameter warnings)
    (void)training_samples;
    (void)training_accuracy;
    
    // Reset degradation flags after successful retraining
    health.is_degrading = false;
    health.needs_retraining = false;
}

WindowedMetrics ModelPerformanceTracker::ComputeCurrentMetrics(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t current_time = GetCurrentTimestampMs();
    uint64_t window_start = current_time - config_.window_duration_ms;
    
    const auto& events = GetEventsForModel(model_id);
    return ComputeMetricsFromEvents(events, window_start, current_time);
}

WindowedMetrics ModelPerformanceTracker::ComputeWindowMetrics(const std::string& model_id,
                                                            uint64_t start_ms, 
                                                            uint64_t end_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    const auto& events = GetEventsForModel(model_id);
    return ComputeMetricsFromEvents(events, start_ms, end_ms);
}

ModelHealthMetrics ModelPerformanceTracker::ComputeHealthMetrics(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& health = GetHealthForModel(model_id);
    
    // Update current accuracy from recent events
    WindowedMetrics current = ComputeCurrentMetrics(model_id);
    health.current_accuracy = current.accuracy_rate;
    
    // Compute trends if enabled
    if (config_.enable_trend_analysis) {
        health.accuracy_trend_1h = ComputeAccuracyTrend(model_id, 3600000);  // 1 hour
        health.accuracy_trend_7d = ComputeAccuracyTrend(model_id, 604800000); // 7 days
        health.is_degrading = IsAccuracyDegrading(model_id);
    }
    
    // Determine if retraining is needed
    bool accuracy_below_threshold = health.current_accuracy < config_.minimum_accuracy_threshold;
    bool significant_degradation = health.is_degrading && 
                                 (health.accuracy_trend_1h < -config_.accuracy_degradation_threshold);
    
    uint64_t current_time = GetCurrentTimestampMs();
    bool enough_time_passed = (current_time - health.last_retrain_timestamp_ms) >= 
                             config_.min_time_between_retrains_ms;
    
    bool enough_samples = current.total_predictions >= config_.min_predictions_for_decision;
    
    health.needs_retraining = enough_samples && enough_time_passed && 
                             (accuracy_below_threshold || significant_degradation);
    
    return health;
}

bool ModelPerformanceTracker::ShouldRetrain(const std::string& model_id) {
    ModelHealthMetrics health = ComputeHealthMetrics(model_id);
    return health.needs_retraining;
}

std::vector<std::string> ModelPerformanceTracker::GetModelsNeedingRetrain() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> models_needing_retrain;
    
    for (const auto& pair : model_health_) {
        const std::string& model_id = pair.first;
        if (ShouldRetrain(model_id)) {
            models_needing_retrain.push_back(model_id);
        }
    }
    
    return models_needing_retrain;
}

std::vector<WindowedMetrics> ModelPerformanceTracker::GetHistoricalMetrics(
    const std::string& model_id, uint64_t start_ms, uint64_t end_ms) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    const auto& windows = GetWindowsForModel(model_id);
    std::vector<WindowedMetrics> result;
    
    for (const auto& window : windows) {
        if (window.window_start_ms >= start_ms && window.window_end_ms <= end_ms) {
            result.push_back(window);
        }
    }
    
    return result;
}

std::vector<std::string> ModelPerformanceTracker::GetTrackedModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> models;
    for (const auto& pair : model_health_) {
        models.push_back(pair.first);
    }
    
    return models;
}

void ModelPerformanceTracker::UpdateConfig(const Config& new_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = new_config;
}

void ModelPerformanceTracker::CleanupOldData() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t current_time = GetCurrentTimestampMs();
    uint64_t cutoff_time = current_time - (config_.max_windows_stored * config_.window_duration_ms);
    
    // Cleanup old events
    for (auto& pair : model_events_) {
        auto& events = pair.second;
        events.erase(
            std::remove_if(events.begin(), events.end(),
                          [cutoff_time](const PredictionEvent& event) {
                              return event.timestamp_ms < cutoff_time;
                          }),
            events.end()
        );
    }
    
    // Cleanup old windows
    for (auto& pair : model_windows_) {
        auto& windows = pair.second;
        while (!windows.empty() && windows.front().window_end_ms < cutoff_time) {
            windows.pop_front();
        }
    }
}

void ModelPerformanceTracker::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    model_events_.clear();
    model_windows_.clear();
    model_health_.clear();
    last_window_computation_.clear();
}

void ModelPerformanceTracker::UpdateWindowedMetrics(const std::string& model_id) {
    uint64_t current_time = GetCurrentTimestampMs();
    uint64_t window_start = current_time - config_.window_duration_ms;
    
    const auto& events = GetEventsForModel(model_id);
    WindowedMetrics metrics = ComputeMetricsFromEvents(events, window_start, current_time);
    
    auto& windows = GetWindowsForModel(model_id);
    windows.push_back(metrics);
    
    // Limit window storage
    while (windows.size() > config_.max_windows_stored) {
        windows.pop_front();
    }
}

WindowedMetrics ModelPerformanceTracker::ComputeMetricsFromEvents(
    const std::deque<PredictionEvent>& events, uint64_t start_ms, uint64_t end_ms) {
    
    WindowedMetrics metrics;
    metrics.window_start_ms = start_ms;
    metrics.window_end_ms = end_ms;
    
    std::vector<const PredictionEvent*> window_events;
    std::vector<double> confidences;
    std::vector<double> errors;
    
    for (const auto& event : events) {
        if (event.timestamp_ms >= start_ms && event.timestamp_ms <= end_ms) {
            window_events.push_back(&event);
            confidences.push_back(event.confidence);
            errors.push_back(event.prediction_error_bytes);
        }
    }
    
    metrics.total_predictions = window_events.size();
    
    if (metrics.total_predictions == 0) {
        return metrics;
    }
    
    // Calculate accuracy
    metrics.correct_predictions = std::count_if(window_events.begin(), window_events.end(),
                                              [](const PredictionEvent* event) {
                                                  return event->was_correct;
                                              });
    metrics.accuracy_rate = static_cast<double>(metrics.correct_predictions) / 
                           metrics.total_predictions;
    
    // Calculate average confidence
    metrics.average_confidence = std::accumulate(confidences.begin(), confidences.end(), 0.0) / 
                                confidences.size();
    
    // Calculate average error
    metrics.average_error_bytes = std::accumulate(errors.begin(), errors.end(), 0.0) / 
                                 errors.size();
    
    // Calculate throughput (queries per second)
    double window_duration_sec = static_cast<double>(end_ms - start_ms) / 1000.0;
    if (window_duration_sec > 0) {
        metrics.throughput_qps = metrics.total_predictions / window_duration_sec;
    }
    
    return metrics;
}

double ModelPerformanceTracker::ComputeAccuracyTrend(const std::string& model_id, 
                                                    uint64_t duration_ms) {
    uint64_t current_time = GetCurrentTimestampMs();
    uint64_t start_time = current_time - duration_ms;
    
    const auto& windows = GetWindowsForModel(model_id);
    
    std::vector<double> accuracies;
    for (const auto& window : windows) {
        if (window.window_start_ms >= start_time && window.total_predictions > 0) {
            accuracies.push_back(window.accuracy_rate);
        }
    }
    
    if (accuracies.size() < 2) {
        return 0.0; // Not enough data for trend analysis
    }
    
    // Simple linear regression to compute trend
    double n = static_cast<double>(accuracies.size());
    double sum_x = n * (n - 1) / 2; // Sum of indices
    double sum_y = std::accumulate(accuracies.begin(), accuracies.end(), 0.0);
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    
    for (size_t i = 0; i < accuracies.size(); ++i) {
        double x = static_cast<double>(i);
        sum_xy += x * accuracies[i];
        sum_xx += x * x;
    }
    
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    return slope;
}

bool ModelPerformanceTracker::IsAccuracyDegrading(const std::string& model_id) {
    double trend_1h = ComputeAccuracyTrend(model_id, 3600000); // 1 hour
    return trend_1h < -config_.accuracy_degradation_threshold;
}

uint64_t ModelPerformanceTracker::GetCurrentTimestampMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::deque<PredictionEvent>& ModelPerformanceTracker::GetEventsForModel(const std::string& model_id) {
    if (model_events_.find(model_id) == model_events_.end()) {
        model_events_[model_id] = std::deque<PredictionEvent>();
    }
    return model_events_[model_id];
}

std::deque<WindowedMetrics>& ModelPerformanceTracker::GetWindowsForModel(const std::string& model_id) {
    if (model_windows_.find(model_id) == model_windows_.end()) {
        model_windows_[model_id] = std::deque<WindowedMetrics>();
    }
    return model_windows_[model_id];
}

ModelHealthMetrics& ModelPerformanceTracker::GetHealthForModel(const std::string& model_id) {
    if (model_health_.find(model_id) == model_health_.end()) {
        model_health_[model_id] = ModelHealthMetrics();
        model_health_[model_id].model_id = model_id;
    }
    return model_health_[model_id];
}

// Singleton implementation
std::unique_ptr<ModelPerformanceTracker> GlobalPerformanceTracker::instance_ = nullptr;
std::once_flag GlobalPerformanceTracker::initialized_;

ModelPerformanceTracker& GlobalPerformanceTracker::GetInstance() {
    std::call_once(initialized_, []() {
        instance_ = std::make_unique<ModelPerformanceTracker>();
    });
    return *instance_;
}

void GlobalPerformanceTracker::Initialize(const ModelPerformanceTracker::Config& config) {
    instance_ = std::make_unique<ModelPerformanceTracker>(config);
}

void GlobalPerformanceTracker::Shutdown() {
    instance_.reset();
}

} // namespace adaptive
} // namespace learned_index
} // namespace rocksdb