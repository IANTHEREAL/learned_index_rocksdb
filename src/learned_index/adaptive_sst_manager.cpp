#include "learned_index/adaptive_sst_manager.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace rocksdb {
namespace learned_index {

AdaptiveSSTLearnedIndexManager::AdaptiveSSTLearnedIndexManager(
    const SSTLearnedIndexOptions& sst_options)
    : SSTLearnedIndexManager(sst_options), adaptive_config_(AdaptiveConfig()) {
    
    if (adaptive_config_.enable_performance_tracking) {
        performance_tracker_ = std::make_unique<adaptive::ModelPerformanceTracker>(
            adaptive_config_.tracker_config);
    }
    
    if (adaptive_config_.enable_adaptive_retraining && performance_tracker_) {
        retraining_manager_ = std::make_unique<adaptive::AdaptiveRetrainingManager>(
            this, performance_tracker_.get(), adaptive_config_.retraining_config);
        
        // Set retraining callback
        retraining_manager_->SetRetrainingCallback(
            [this](const adaptive::RetrainingResult& result) {
                OnRetrainingComplete(result);
            });
    }
}

AdaptiveSSTLearnedIndexManager::AdaptiveSSTLearnedIndexManager(
    const SSTLearnedIndexOptions& sst_options,
    const AdaptiveConfig& adaptive_config)
    : SSTLearnedIndexManager(sst_options), adaptive_config_(adaptive_config) {
    
    if (adaptive_config_.enable_performance_tracking) {
        performance_tracker_ = std::make_unique<adaptive::ModelPerformanceTracker>(
            adaptive_config_.tracker_config);
    }
    
    if (adaptive_config_.enable_adaptive_retraining && performance_tracker_) {
        retraining_manager_ = std::make_unique<adaptive::AdaptiveRetrainingManager>(
            this, performance_tracker_.get(), adaptive_config_.retraining_config);
        
        // Set retraining callback
        retraining_manager_->SetRetrainingCallback(
            [this](const adaptive::RetrainingResult& result) {
                OnRetrainingComplete(result);
            });
    }
}

uint32_t AdaptiveSSTLearnedIndexManager::PredictBlockIndex(const std::string& sst_file_path, 
                                                         uint64_t key) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get prediction from base class
    uint32_t predicted_block = SSTLearnedIndexManager::PredictBlockIndex(sst_file_path, key);
    double confidence = GetPredictionConfidence(sst_file_path, key);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    (void)start_time; (void)end_time; // Suppress unused warnings
    
    // For now, we don't know the actual block, so we'll record it as unknown
    // The actual block will be recorded later via RecordActualBlock
    if (performance_tracker_ && adaptive_monitoring_active_.load()) {
        adaptive::PredictionEvent event;
        event.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        event.key = key;
        event.predicted_block = predicted_block;
        event.actual_block = predicted_block; // Will be updated when actual is known
        event.confidence = confidence;
        event.was_correct = true; // Will be updated when actual is known
        event.prediction_error_bytes = 0.0; // Will be calculated when actual is known
        
        // Store temporarily - in practice, we'd have a mechanism to match predictions with actuals
        performance_tracker_->RecordPrediction(sst_file_path, event);
    }
    
    return predicted_block;
}

bool AdaptiveSSTLearnedIndexManager::TrainModel(
    const std::string& sst_file_path,
    const std::vector<std::pair<uint64_t, uint32_t>>& key_block_pairs) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = SSTLearnedIndexManager::TrainModel(sst_file_path, key_block_pairs);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (success && performance_tracker_) {
        // Record training event
        uint64_t timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Estimate training accuracy (in practice, this would use a validation set)
        double estimated_accuracy = 0.95; // Placeholder
        
        performance_tracker_->RecordTrainingEvent(sst_file_path, timestamp_ms, 
                                                 key_block_pairs.size(), estimated_accuracy);
    }
    
    return success;
}

void AdaptiveSSTLearnedIndexManager::RecordActualBlock(const std::string& sst_file_path, 
                                                     uint64_t key, 
                                                     uint32_t actual_block) {
    if (!performance_tracker_ || !adaptive_monitoring_active_.load()) {
        return;
    }
    
    // In practice, we would match this with a previous prediction
    // For now, we'll make a new prediction to compare
    uint32_t predicted_block = SSTLearnedIndexManager::PredictBlockIndex(sst_file_path, key);
    double confidence = GetPredictionConfidence(sst_file_path, key);
    bool was_correct = (predicted_block == actual_block);
    
    // Calculate prediction error (simplified - would be based on byte offset in practice)
    double prediction_error = was_correct ? 0.0 : std::abs(static_cast<double>(predicted_block - actual_block)) * 4096.0;
    (void)prediction_error; // Suppress unused variable warning
    
    RecordPredictionEvent(sst_file_path, key, predicted_block, actual_block, 
                         confidence, was_correct);
}

bool AdaptiveSSTLearnedIndexManager::RequestModelRetraining(const std::string& sst_file_path,
                                                          const std::string& reason) {
    if (!retraining_manager_) {
        return false;
    }
    
    return retraining_manager_->RequestRetraining(sst_file_path, sst_file_path, reason);
}

adaptive::ModelHealthMetrics AdaptiveSSTLearnedIndexManager::GetModelHealth(
    const std::string& sst_file_path) const {
    
    if (!performance_tracker_) {
        return adaptive::ModelHealthMetrics();
    }
    
    return performance_tracker_->ComputeHealthMetrics(sst_file_path);
}

adaptive::WindowedMetrics AdaptiveSSTLearnedIndexManager::GetCurrentMetrics(
    const std::string& sst_file_path) const {
    
    if (!performance_tracker_) {
        return adaptive::WindowedMetrics();
    }
    
    return performance_tracker_->ComputeCurrentMetrics(sst_file_path);
}

std::vector<std::string> AdaptiveSSTLearnedIndexManager::GetModelsNeedingRetrain() const {
    if (!retraining_manager_) {
        return {};
    }
    
    return performance_tracker_->GetModelsNeedingRetrain();
}

void AdaptiveSSTLearnedIndexManager::UpdateAdaptiveConfig(const AdaptiveConfig& new_config) {
    adaptive_config_ = new_config;
    
    if (performance_tracker_) {
        performance_tracker_->UpdateConfig(adaptive_config_.tracker_config);
    }
    
    if (retraining_manager_) {
        retraining_manager_->UpdateConfig(adaptive_config_.retraining_config);
    }
}

void AdaptiveSSTLearnedIndexManager::StartAdaptiveMonitoring() {
    if (adaptive_monitoring_active_.load()) {
        return;
    }
    
    adaptive_monitoring_active_.store(true);
    
    if (retraining_manager_ && adaptive_config_.enable_adaptive_retraining) {
        retraining_manager_->Start();
    }
}

void AdaptiveSSTLearnedIndexManager::StopAdaptiveMonitoring() {
    adaptive_monitoring_active_.store(false);
    
    if (retraining_manager_) {
        retraining_manager_->Stop();
    }
}

bool AdaptiveSSTLearnedIndexManager::IsAdaptiveMonitoringActive() const {
    return adaptive_monitoring_active_.load();
}

bool AdaptiveSSTLearnedIndexManager::ExportMetrics(const std::string& format) const {
    if (!adaptive_config_.enable_metrics_export || !performance_tracker_) {
        return false;
    }
    
    try {
        std::string metrics_data;
        if (format == "json") {
            metrics_data = ExportMetricsAsJSON();
        } else if (format == "csv") {
            metrics_data = ExportMetricsAsCSV();
        } else {
            return false;
        }
        
        // Write to file
        std::string filename = adaptive_config_.metrics_export_path + "_" + 
                              std::to_string(std::time(nullptr)) + "." + format;
        std::ofstream file(filename);
        file << metrics_data;
        file.close();
        
        // Call export callback if set
        if (metrics_export_callback_) {
            metrics_export_callback_(filename);
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void AdaptiveSSTLearnedIndexManager::SetMetricsExportCallback(
    std::function<void(const std::string&)> callback) {
    metrics_export_callback_ = callback;
}

void AdaptiveSSTLearnedIndexManager::RecordPredictionEvent(const std::string& sst_file_path,
                                                         uint64_t key,
                                                         uint32_t predicted_block,
                                                         uint32_t actual_block,
                                                         double confidence,
                                                         bool was_correct) {
    if (!performance_tracker_) {
        return;
    }
    
    adaptive::PredictionEvent event;
    event.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    event.key = key;
    event.predicted_block = predicted_block;
    event.actual_block = actual_block;
    event.confidence = confidence;
    event.was_correct = was_correct;
    event.prediction_error_bytes = was_correct ? 0.0 : 
        std::abs(static_cast<double>(predicted_block - actual_block)) * 4096.0;
    
    performance_tracker_->RecordPrediction(sst_file_path, event);
}

void AdaptiveSSTLearnedIndexManager::OnRetrainingComplete(const adaptive::RetrainingResult& result) {
    // Log retraining result (in practice, this might update monitoring systems)
    if (result.success) {
        // Retraining successful - metrics will be automatically updated by the performance tracker
    } else {
        // Retraining failed - might need to alert or fallback to traditional index
    }
}

std::string AdaptiveSSTLearnedIndexManager::ExportMetricsAsJSON() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"timestamp\": " << std::time(nullptr) << ",\n";
    ss << "  \"models\": [\n";
    
    auto tracked_models = performance_tracker_->GetTrackedModels();
    for (size_t i = 0; i < tracked_models.size(); ++i) {
        const std::string& model_id = tracked_models[i];
        auto health = performance_tracker_->ComputeHealthMetrics(model_id);
        auto current_metrics = performance_tracker_->ComputeCurrentMetrics(model_id);
        
        ss << "    {\n";
        ss << "      \"model_id\": \"" << model_id << "\",\n";
        ss << "      \"current_accuracy\": " << health.current_accuracy << ",\n";
        ss << "      \"total_queries\": " << health.total_queries_served << ",\n";
        ss << "      \"accuracy_trend_1h\": " << health.accuracy_trend_1h << ",\n";
        ss << "      \"accuracy_trend_7d\": " << health.accuracy_trend_7d << ",\n";
        ss << "      \"is_degrading\": " << (health.is_degrading ? "true" : "false") << ",\n";
        ss << "      \"needs_retraining\": " << (health.needs_retraining ? "true" : "false") << ",\n";
        ss << "      \"retrain_count\": " << health.retrain_count << ",\n";
        ss << "      \"current_throughput_qps\": " << current_metrics.throughput_qps << "\n";
        ss << "    }";
        if (i < tracked_models.size() - 1) ss << ",";
        ss << "\n";
    }
    
    ss << "  ]\n";
    ss << "}";
    
    return ss.str();
}

std::string AdaptiveSSTLearnedIndexManager::ExportMetricsAsCSV() const {
    std::stringstream ss;
    ss << "model_id,current_accuracy,total_queries,accuracy_trend_1h,accuracy_trend_7d,";
    ss << "is_degrading,needs_retraining,retrain_count,current_throughput_qps\n";
    
    auto tracked_models = performance_tracker_->GetTrackedModels();
    for (const std::string& model_id : tracked_models) {
        auto health = performance_tracker_->ComputeHealthMetrics(model_id);
        auto current_metrics = performance_tracker_->ComputeCurrentMetrics(model_id);
        
        ss << model_id << ","
           << health.current_accuracy << ","
           << health.total_queries_served << ","
           << health.accuracy_trend_1h << ","
           << health.accuracy_trend_7d << ","
           << (health.is_degrading ? "1" : "0") << ","
           << (health.needs_retraining ? "1" : "0") << ","
           << health.retrain_count << ","
           << current_metrics.throughput_qps << "\n";
    }
    
    return ss.str();
}

// Factory implementations
std::unique_ptr<AdaptiveSSTLearnedIndexManager> AdaptiveSSTManagerFactory::CreateDefault() {
    SSTLearnedIndexOptions sst_options;
    AdaptiveSSTLearnedIndexManager::AdaptiveConfig adaptive_config;
    
    return std::make_unique<AdaptiveSSTLearnedIndexManager>(sst_options, adaptive_config);
}

std::unique_ptr<AdaptiveSSTLearnedIndexManager> AdaptiveSSTManagerFactory::CreateForProduction() {
    SSTLearnedIndexOptions sst_options;
    sst_options.enable_learned_index = true;
    sst_options.confidence_threshold = 0.85;
    sst_options.cache_models = true;
    sst_options.max_cache_size = 1000;
    
    AdaptiveSSTLearnedIndexManager::AdaptiveConfig adaptive_config;
    adaptive_config.enable_performance_tracking = true;
    adaptive_config.enable_adaptive_retraining = true;
    adaptive_config.enable_metrics_export = true;
    
    // Production-tuned tracker config
    adaptive_config.tracker_config.window_duration_ms = 60000; // 1 minute windows
    adaptive_config.tracker_config.max_windows_stored = 1440; // 24 hours
    adaptive_config.tracker_config.minimum_accuracy_threshold = 0.85;
    adaptive_config.tracker_config.accuracy_degradation_threshold = 0.05;
    
    // Production-tuned retraining config
    adaptive_config.retraining_config.monitoring_interval_ms = 60000; // 1 minute
    adaptive_config.retraining_config.max_concurrent_retraining = 1;
    adaptive_config.retraining_config.min_new_samples_for_retrain = 5000;
    
    return std::make_unique<AdaptiveSSTLearnedIndexManager>(sst_options, adaptive_config);
}

std::unique_ptr<AdaptiveSSTLearnedIndexManager> AdaptiveSSTManagerFactory::CreateForTesting() {
    SSTLearnedIndexOptions sst_options;
    AdaptiveSSTLearnedIndexManager::AdaptiveConfig adaptive_config;
    
    // Testing-optimized config (faster intervals, smaller thresholds)
    adaptive_config.tracker_config.window_duration_ms = 5000; // 5 second windows
    adaptive_config.tracker_config.max_windows_stored = 100;
    adaptive_config.tracker_config.minimum_accuracy_threshold = 0.7;
    adaptive_config.tracker_config.min_predictions_for_decision = 10;
    
    adaptive_config.retraining_config.monitoring_interval_ms = 5000; // 5 seconds
    adaptive_config.retraining_config.min_new_samples_for_retrain = 50;
    
    return std::make_unique<AdaptiveSSTLearnedIndexManager>(sst_options, adaptive_config);
}

std::unique_ptr<AdaptiveSSTLearnedIndexManager> AdaptiveSSTManagerFactory::CreateWithConfig(
    const SSTLearnedIndexOptions& sst_options,
    const AdaptiveSSTLearnedIndexManager::AdaptiveConfig& adaptive_config) {
    
    return std::make_unique<AdaptiveSSTLearnedIndexManager>(sst_options, adaptive_config);
}

} // namespace learned_index
} // namespace rocksdb