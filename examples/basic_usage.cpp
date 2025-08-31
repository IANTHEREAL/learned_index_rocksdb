#include "learned_index/learned_index_block.h"
#include "learned_index/ml_model.h"
#include "learned_index/sst_learned_index_manager.h"
#include <iostream>
#include <vector>
#include <random>

using namespace rocksdb::learned_index;

void demonstrate_learned_index_block() {
    std::cout << "=== Learned Index Block Demo ===" << std::endl;
    
    // Create a learned index block
    LearnedIndexBlock block;
    block.model_type = ModelType::LINEAR;
    block.feature_dimensions = 1;
    block.parameters = {1.5, 0.8}; // bias and weight for linear model
    block.parameter_count = 2;
    
    // Add some block predictions
    block.block_predictions.push_back(BlockPrediction(0, 1000, 2000, 0.95));
    block.block_predictions.push_back(BlockPrediction(1, 2000, 3000, 0.88));
    block.block_predictions.push_back(BlockPrediction(2, 3000, 4000, 0.92));
    
    // Update metadata
    block.metadata.training_samples = 1000;
    block.metadata.training_accuracy = 0.85;
    block.metadata.validation_accuracy = 0.83;
    
    // Update checksum
    block.UpdateChecksum();
    
    std::cout << "Block valid: " << (block.IsValid() ? "Yes" : "No") << std::endl;
    std::cout << "Serialized size: " << block.GetSerializedSize() << " bytes" << std::endl;
    
    // Test serialization
    std::string serialized = block.Serialize();
    std::cout << "Serialization successful: " << (!serialized.empty() ? "Yes" : "No") << std::endl;
    
    // Test deserialization
    LearnedIndexBlock deserialized_block;
    bool deserialize_success = deserialized_block.Deserialize(serialized);
    std::cout << "Deserialization successful: " << (deserialize_success ? "Yes" : "No") << std::endl;
    
    if (deserialize_success) {
        std::cout << "Deserialized block valid: " << (deserialized_block.IsValid() ? "Yes" : "No") << std::endl;
        std::cout << "Parameters match: " << (block.parameters == deserialized_block.parameters ? "Yes" : "No") << std::endl;
    }
    
    std::cout << std::endl;
}

void demonstrate_linear_model() {
    std::cout << "=== Linear Model Demo ===" << std::endl;
    
    // Create a linear model
    LinearModel model(1); // 1-dimensional features
    
    // Generate synthetic training data
    std::vector<std::vector<double>> features;
    std::vector<uint64_t> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for (int i = 0; i < 100; ++i) {
        double x = dis(gen);
        features.push_back({x});
        // Target is roughly 2*x + noise
        targets.push_back(static_cast<uint64_t>(2.0 * x + dis(gen) * 0.1));
    }
    
    // Train the model
    bool training_success = model.Train(features, targets);
    std::cout << "Training successful: " << (training_success ? "Yes" : "No") << std::endl;
    
    if (training_success) {
        std::cout << "Training accuracy: " << model.GetTrainingAccuracy() << std::endl;
        std::cout << "Parameter count: " << model.GetParameterCount() << std::endl;
        
        // Test predictions
        std::vector<double> test_features = {50.0};
        uint64_t prediction = model.Predict(test_features);
        double confidence = model.GetConfidence(test_features);
        
        std::cout << "Prediction for x=50: " << prediction << std::endl;
        std::cout << "Confidence: " << confidence << std::endl;
        
        // Test parameter serialization
        auto parameters = model.GetParameters();
        std::cout << "Model parameters: [";
        for (size_t i = 0; i < parameters.size(); ++i) {
            std::cout << parameters[i];
            if (i < parameters.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << std::endl;
}

void demonstrate_sst_manager() {
    std::cout << "=== SST Learned Index Manager Demo ===" << std::endl;
    
    // Create manager with options
    SSTLearnedIndexOptions options;
    options.model_type = ModelType::LINEAR;
    options.confidence_threshold = 0.7;
    options.max_prediction_error_bytes = 1024;
    
    SSTLearnedIndexManager manager(options);
    
    // Initialize with fake SST file info
    bool init_success = manager.Initialize("test.sst", 10000);
    std::cout << "Initialization successful: " << (init_success ? "Yes" : "No") << std::endl;
    
    if (init_success) {
        // Create synthetic key ranges for training
        std::vector<KeyRange> key_ranges;
        for (uint32_t i = 0; i < 10; ++i) {
            uint64_t start = i * 1000;
            uint64_t end = (i + 1) * 1000;
            key_ranges.emplace_back(start, end, i, 100);
        }
        
        // Train the model
        bool training_success = manager.TrainModel(key_ranges);
        std::cout << "Model training successful: " << (training_success ? "Yes" : "No") << std::endl;
        
        if (training_success) {
            std::cout << "Model ready: " << (manager.IsModelReady() ? "Yes" : "No") << std::endl;
            std::cout << "Model type: " << static_cast<int>(manager.GetModelType()) << std::endl;
            std::cout << "Model size: " << manager.GetModelSize() << " bytes" << std::endl;
            std::cout << "Training accuracy: " << manager.GetTrainingAccuracy() << std::endl;
            
            // Test predictions
            for (uint64_t test_key : {500, 1500, 2500, 5500, 9500}) {
                double confidence;
                uint32_t predicted_block = manager.PredictBlock(test_key, &confidence);
                std::cout << "Key " << test_key << " -> Block " << predicted_block 
                         << " (confidence: " << confidence << ")" << std::endl;
            }
            
            // Test model save/load
            LearnedIndexBlock saved_block;
            bool save_success = manager.SaveModel(&saved_block);
            std::cout << "Model save successful: " << (save_success ? "Yes" : "No") << std::endl;
            
            if (save_success) {
                SSTLearnedIndexManager new_manager(options);
                new_manager.Initialize("test2.sst", 10000);
                bool load_success = new_manager.LoadModel(saved_block);
                std::cout << "Model load successful: " << (load_success ? "Yes" : "No") << std::endl;
            }
        }
    }
    
    std::cout << std::endl;
    
    // Print diagnostics
    std::cout << manager.GetDiagnosticsInfo() << std::endl;
}

int main() {
    std::cout << "Learned Index RocksDB - Basic Usage Example" << std::endl;
    std::cout << "===========================================" << std::endl << std::endl;
    
    try {
        demonstrate_learned_index_block();
        demonstrate_linear_model();
        demonstrate_sst_manager();
        
        std::cout << "All demonstrations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}