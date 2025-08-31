#pragma once

#include "learned_index/sst_learned_index_manager.h"
#include "learned_index/adaptive/model_performance_tracker.h"
#include "learned_index/adaptive/adaptive_retraining_manager.h"
#include <memory>
#include <atomic>

namespace rocksdb {
namespace learned_index {

// Enhanced SST manager with adaptive retraining capabilities
class AdaptiveSSTLearnedIndexManager : public SSTLearnedIndexManager {
public:
    struct AdaptiveConfig {
        bool enable_performance_tracking = true;
        bool enable_adaptive_retraining = true;
        bool enable_metrics_export = true;
        std::string metrics_export_path = "/tmp/learned_index_metrics";
        
        // Performance tracking config
        adaptive::ModelPerformanceTracker::Config tracker_config;
        
        // Retraining config  
        adaptive::AdaptiveRetrainingManager::Config retraining_config;
        
        AdaptiveConfig() = default;
    };

    explicit AdaptiveSSTLearnedIndexManager(const SSTLearnedIndexOptions& sst_options);
    explicit AdaptiveSSTLearnedIndexManager(const SSTLearnedIndexOptions& sst_options,
                                          const AdaptiveConfig& adaptive_config);
    ~AdaptiveSSTLearnedIndexManager() = default;

    // Enhanced prediction with performance tracking
    uint32_t PredictBlockIndex(const std::string& sst_file_path, uint64_t key);
    
    // Enhanced training with performance monitoring
    bool TrainModel(const std::string& sst_file_path,
                   const std::vector<std::pair<uint64_t, uint32_t>>& key_block_pairs);
    
    // New methods for adaptive functionality
    void RecordActualBlock(const std::string& sst_file_path, uint64_t key, uint32_t actual_block);
    bool RequestModelRetraining(const std::string& sst_file_path, const std::string& reason = "manual");
    
    // Performance monitoring
    adaptive::ModelHealthMetrics GetModelHealth(const std::string& sst_file_path) const;
    adaptive::WindowedMetrics GetCurrentMetrics(const std::string& sst_file_path) const;
    std::vector<std::string> GetModelsNeedingRetrain() const;
    
    // Configuration management
    void UpdateAdaptiveConfig(const AdaptiveConfig& new_config);
    const AdaptiveConfig& GetAdaptiveConfig() const { return adaptive_config_; }
    
    // Lifecycle management
    void StartAdaptiveMonitoring();
    void StopAdaptiveMonitoring();
    bool IsAdaptiveMonitoringActive() const;
    
    // Metrics export
    bool ExportMetrics(const std::string& format = "json") const;
    void SetMetricsExportCallback(std::function<void(const std::string&)> callback);

private:
    AdaptiveConfig adaptive_config_;
    
    // Adaptive components
    std::unique_ptr<adaptive::ModelPerformanceTracker> performance_tracker_;
    std::unique_ptr<adaptive::AdaptiveRetrainingManager> retraining_manager_;
    
    // State tracking
    std::atomic<bool> adaptive_monitoring_active_{false};
    std::function<void(const std::string&)> metrics_export_callback_;
    
    // Performance tracking helpers
    void RecordPredictionEvent(const std::string& sst_file_path, uint64_t key, 
                             uint32_t predicted_block, uint32_t actual_block, 
                             double confidence, bool was_correct);
    
    // Retraining callback
    void OnRetrainingComplete(const adaptive::RetrainingResult& result);
    
    // Metrics export helpers
    std::string ExportMetricsAsJSON() const;
    std::string ExportMetricsAsCSV() const;
};

// Factory for creating adaptive managers with different configurations
class AdaptiveSSTManagerFactory {
public:
    static std::unique_ptr<AdaptiveSSTLearnedIndexManager> CreateDefault();
    static std::unique_ptr<AdaptiveSSTLearnedIndexManager> CreateForProduction();
    static std::unique_ptr<AdaptiveSSTLearnedIndexManager> CreateForTesting();
    static std::unique_ptr<AdaptiveSSTLearnedIndexManager> CreateWithConfig(
        const SSTLearnedIndexOptions& sst_options,
        const AdaptiveSSTLearnedIndexManager::AdaptiveConfig& adaptive_config);
};

} // namespace learned_index
} // namespace rocksdb