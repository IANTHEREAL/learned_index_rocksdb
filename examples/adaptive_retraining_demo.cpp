#include "learned_index/adaptive_sst_manager.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>

using namespace rocksdb::learned_index;

class RetrainingDemo {
public:
    RetrainingDemo() {
        // Create adaptive manager with testing configuration for faster demo
        manager_ = AdaptiveSSTManagerFactory::CreateForTesting();
        
        // Configure for demonstration
        auto config = manager_->GetAdaptiveConfig();
        config.tracker_config.window_duration_ms = 10000; // 10 second windows
        config.tracker_config.minimum_accuracy_threshold = 0.75; // Lower threshold for demo
        config.retraining_config.monitoring_interval_ms = 5000; // 5 second monitoring
        config.retraining_config.min_new_samples_for_retrain = 100; // Smaller sample size
        manager_->UpdateAdaptiveConfig(config);
        
        sst_file_path_ = "demo_sst_file.sst";
    }
    
    void RunDemo() {
        std::cout << "=== Adaptive Retraining Demonstration ===" << std::endl;
        std::cout << "This demo shows how learned indexes adapt to changing workloads" << std::endl;
        std::cout << std::string(60, '=') << std::endl << std::endl;
        
        // Step 1: Train initial model
        std::cout << "Step 1: Training initial model with sequential data..." << std::endl;
        TrainInitialModel();
        PrintModelHealth();
        std::cout << std::endl;
        
        // Step 2: Start adaptive monitoring
        std::cout << "Step 2: Starting adaptive monitoring..." << std::endl;
        manager_->StartAdaptiveMonitoring();
        std::cout << "Adaptive monitoring started." << std::endl << std::endl;
        
        // Step 3: Simulate good performance period
        std::cout << "Step 3: Simulating good performance period (30 seconds)..." << std::endl;
        SimulateGoodPerformance(30);
        PrintModelHealth();
        std::cout << std::endl;
        
        // Step 4: Introduce workload shift
        std::cout << "Step 4: Introducing workload shift to random access pattern..." << std::endl;
        std::cout << "This should cause accuracy degradation and trigger retraining." << std::endl;
        SimulateWorkloadShift(60);
        PrintModelHealth();
        std::cout << std::endl;
        
        // Step 5: Show recovery after retraining
        std::cout << "Step 5: Continuing with new pattern to show recovery..." << std::endl;
        SimulateRecoveryPeriod(30);
        PrintModelHealth();
        std::cout << std::endl;
        
        // Step 6: Export metrics
        std::cout << "Step 6: Exporting performance metrics..." << std::endl;
        ExportMetrics();
        
        manager_->StopAdaptiveMonitoring();
        std::cout << std::endl << "Demo completed!" << std::endl;
    }

private:
    std::unique_ptr<AdaptiveSSTLearnedIndexManager> manager_;
    std::string sst_file_path_;
    std::mt19937 rng_{42};
    
    void TrainInitialModel() {
        // Generate sequential training data
        std::vector<std::pair<uint64_t, uint32_t>> training_data;
        for (uint64_t key = 1000; key < 11000; key += 10) {
            uint32_t block = static_cast<uint32_t>((key - 1000) / 1000); // 10 blocks
            training_data.emplace_back(key, block);
        }
        
        bool success = manager_->TrainModel(sst_file_path_, training_data);
        std::cout << "  Initial training: " << (success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "  Training samples: " << training_data.size() << std::endl;
    }
    
    void SimulateGoodPerformance(int duration_seconds) {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);
        
        std::uniform_int_distribution<uint64_t> key_dist(1000, 10900); // Within training range
        
        int query_count = 0;
        while (std::chrono::steady_clock::now() < end_time) {
            uint64_t key = key_dist(rng_);
            uint32_t predicted_block = manager_->PredictBlockIndex(sst_file_path_, key);
            
            // Simulate mostly correct actual blocks (good performance)
            uint32_t actual_block = predicted_block;
            if (rng_() % 100 < 10) { // 10% error rate
                actual_block = (predicted_block + 1) % 10;
            }
            
            manager_->RecordActualBlock(sst_file_path_, key, actual_block);
            
            query_count++;
            if (query_count % 100 == 0) {
                std::cout << "\r  Processed " << query_count << " queries..." << std::flush;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "\r  Completed " << query_count << " queries with good performance." << std::endl;
    }
    
    void SimulateWorkloadShift(int duration_seconds) {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);
        
        std::uniform_int_distribution<uint64_t> key_dist(20000, 30000); // Outside training range
        
        int query_count = 0;
        while (std::chrono::steady_clock::now() < end_time) {
            uint64_t key = key_dist(rng_);
            uint32_t predicted_block = manager_->PredictBlockIndex(sst_file_path_, key);
            
            // Simulate poor predictions due to workload shift
            uint32_t actual_block = rng_() % 10; // Random actual blocks
            
            manager_->RecordActualBlock(sst_file_path_, key, actual_block);
            
            query_count++;
            if (query_count % 100 == 0) {
                std::cout << "\r  Processed " << query_count << " queries (workload shifted)..." << std::flush;
                
                // Check if retraining was triggered
                auto models_needing_retrain = manager_->GetModelsNeedingRetrain();
                if (!models_needing_retrain.empty()) {
                    std::cout << std::endl << "  ⚠️  Retraining triggered due to performance degradation!" << std::endl;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::cout << "\r  Completed " << query_count << " queries with shifted workload." << std::endl;
    }
    
    void SimulateRecoveryPeriod(int duration_seconds) {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);
        
        std::uniform_int_distribution<uint64_t> key_dist(20000, 30000); // New pattern
        
        int query_count = 0;
        while (std::chrono::steady_clock::now() < end_time) {
            uint64_t key = key_dist(rng_);
            uint32_t predicted_block = manager_->PredictBlockIndex(sst_file_path_, key);
            
            // Simulate improved predictions after retraining
            uint32_t actual_block = (key - 20000) / 1000; // New pattern that should be learnable
            if (rng_() % 100 < 20) { // 20% error rate (better than during shift)
                actual_block = rng_() % 10;
            }
            
            manager_->RecordActualBlock(sst_file_path_, key, actual_block);
            
            query_count++;
            if (query_count % 50 == 0) {
                std::cout << "\r  Recovery progress: " << query_count << " queries..." << std::flush;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        std::cout << "\r  Completed " << query_count << " queries in recovery period." << std::endl;
    }
    
    void PrintModelHealth() {
        auto health = manager_->GetModelHealth(sst_file_path_);
        auto metrics = manager_->GetCurrentMetrics(sst_file_path_);
        
        std::cout << "📊 Model Health Report:" << std::endl;
        std::cout << "  Current Accuracy: " << std::fixed << std::setprecision(1) 
                  << (health.current_accuracy * 100) << "%" << std::endl;
        std::cout << "  Total Queries: " << health.total_queries_served << std::endl;
        std::cout << "  1H Accuracy Trend: " << std::setprecision(3) 
                  << (health.accuracy_trend_1h * 100) << "%" << std::endl;
        std::cout << "  Is Degrading: " << (health.is_degrading ? "YES ⚠️" : "NO ✅") << std::endl;
        std::cout << "  Needs Retraining: " << (health.needs_retraining ? "YES 🔄" : "NO ✅") << std::endl;
        std::cout << "  Retraining Count: " << health.retrain_count << std::endl;
        std::cout << "  Current Throughput: " << std::setprecision(1) 
                  << metrics.throughput_qps << " QPS" << std::endl;
    }
    
    void ExportMetrics() {
        bool json_success = manager_->ExportMetrics("json");
        bool csv_success = manager_->ExportMetrics("csv");
        
        std::cout << "  JSON export: " << (json_success ? "SUCCESS" : "FAILED") << std::endl;
        std::cout << "  CSV export: " << (csv_success ? "SUCCESS" : "FAILED") << std::endl;
        
        if (json_success || csv_success) {
            std::cout << "  Metrics exported to /tmp/learned_index_metrics_*" << std::endl;
        }
    }
};

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [option]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --demo       Run interactive adaptive retraining demo" << std::endl;
    std::cout << "  --dashboard  Print dashboard startup instructions" << std::endl;
    std::cout << "  --help       Show this help message" << std::endl;
}

void PrintDashboardInstructions() {
    std::cout << "=== Performance Dashboard Instructions ===" << std::endl;
    std::cout << std::endl;
    std::cout << "To start the performance dashboard:" << std::endl;
    std::cout << std::endl;
    std::cout << "1. Install Python dependencies:" << std::endl;
    std::cout << "   pip install flask plotly sqlite3" << std::endl;
    std::cout << std::endl;
    std::cout << "2. Navigate to the dashboard directory:" << std::endl;
    std::cout << "   cd dashboard/" << std::endl;
    std::cout << std::endl;
    std::cout << "3. Start the dashboard server:" << std::endl;
    std::cout << "   python3 dashboard_server.py" << std::endl;
    std::cout << std::endl;
    std::cout << "4. Open your browser and visit:" << std::endl;
    std::cout << "   http://localhost:5000" << std::endl;
    std::cout << std::endl;
    std::cout << "5. Click 'Start Demo Data' to begin generating sample metrics" << std::endl;
    std::cout << std::endl;
    std::cout << "The dashboard provides:" << std::endl;
    std::cout << "  • Real-time accuracy and throughput charts" << std::endl;
    std::cout << "  • Model health monitoring" << std::endl;
    std::cout << "  • Retraining event tracking" << std::endl;
    std::cout << "  • Performance trend analysis" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::string option = (argc > 1) ? argv[1] : "--demo";
    
    if (option == "--help" || option == "-h") {
        PrintUsage(argv[0]);
        return 0;
    } else if (option == "--dashboard") {
        PrintDashboardInstructions();
        return 0;
    } else if (option == "--demo") {
        try {
            RetrainingDemo demo;
            demo.RunDemo();
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error running demo: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown option: " << option << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
}