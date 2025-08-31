#pragma once

#include <vector>
#include <memory>
#include <string>
#include "learned_index_block.h"

namespace rocksdb {
namespace learned_index {

// Abstract base class for ML models
class MLModel {
public:
    virtual ~MLModel() = default;
    
    // Train the model with input features and target positions
    virtual bool Train(const std::vector<std::vector<double>>& features,
                      const std::vector<uint64_t>& targets) = 0;
    
    // Make prediction for given features
    virtual uint64_t Predict(const std::vector<double>& features) const = 0;
    
    // Get prediction confidence (0.0 to 1.0)
    virtual double GetConfidence(const std::vector<double>& features) const = 0;
    
    // Serialize model parameters
    virtual std::vector<double> GetParameters() const = 0;
    
    // Load model from parameters
    virtual bool LoadParameters(const std::vector<double>& params) = 0;
    
    // Get model type
    virtual ModelType GetType() const = 0;
    
    // Get number of parameters
    virtual size_t GetParameterCount() const = 0;
    
    // Get feature dimensions required
    virtual size_t GetFeatureDimensions() const = 0;
    
    // Get training accuracy
    virtual double GetTrainingAccuracy() const = 0;
    
    // Validate model state
    virtual bool IsValid() const = 0;
};

// Linear regression model implementation
class LinearModel : public MLModel {
private:
    std::vector<double> weights_;
    double bias_;
    double training_accuracy_;
    size_t feature_dimensions_;
    bool is_trained_;

public:
    LinearModel(size_t feature_dims = 1);
    
    bool Train(const std::vector<std::vector<double>>& features,
              const std::vector<uint64_t>& targets) override;
              
    uint64_t Predict(const std::vector<double>& features) const override;
    
    double GetConfidence(const std::vector<double>& features) const override;
    
    std::vector<double> GetParameters() const override;
    
    bool LoadParameters(const std::vector<double>& params) override;
    
    ModelType GetType() const override { return ModelType::LINEAR; }
    
    size_t GetParameterCount() const override;
    
    size_t GetFeatureDimensions() const override { return feature_dimensions_; }
    
    double GetTrainingAccuracy() const override { return training_accuracy_; }
    
    bool IsValid() const override { return is_trained_; }

private:
    // Helper function for least squares regression
    bool SolveLeastSquares(const std::vector<std::vector<double>>& X,
                          const std::vector<uint64_t>& y);
                          
    // Solve linear system Ax = b using Gaussian elimination
    bool SolveLinearSystem(std::vector<std::vector<double>>& A,
                          std::vector<double>& b,
                          std::vector<double>& x) const;
                          
    // Calculate mean squared error
    double CalculateMSE(const std::vector<std::vector<double>>& features,
                       const std::vector<uint64_t>& targets) const;
                       
    // Calculate variance of values
    double CalculateVariance(const std::vector<double>& values) const;
};

// Factory class for creating ML models
class MLModelFactory {
public:
    static std::unique_ptr<MLModel> CreateModel(ModelType type, 
                                               size_t feature_dimensions = 1);
    
    static std::unique_ptr<MLModel> LoadModel(const LearnedIndexBlock& block);
};

} // namespace learned_index
} // namespace rocksdb