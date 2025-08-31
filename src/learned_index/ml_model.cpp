#include "learned_index/ml_model.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace rocksdb {
namespace learned_index {

// LinearModel implementation
LinearModel::LinearModel(size_t feature_dims)
    : feature_dimensions_(feature_dims)
    , bias_(0.0)
    , training_accuracy_(0.0)
    , is_trained_(false) {
    weights_.resize(feature_dimensions_, 0.0);
}

bool LinearModel::Train(const std::vector<std::vector<double>>& features,
                       const std::vector<uint64_t>& targets) {
    if (features.empty() || targets.empty() || features.size() != targets.size()) {
        return false;
    }
    
    // Validate feature dimensions
    for (const auto& feature_vec : features) {
        if (feature_vec.size() != feature_dimensions_) {
            return false;
        }
    }
    
    // Convert targets to double for calculations
    std::vector<double> y_double(targets.begin(), targets.end());
    
    // Solve least squares regression
    bool success = SolveLeastSquares(features, targets);
    if (!success) {
        return false;
    }
    
    // Calculate training accuracy (R-squared)
    training_accuracy_ = 1.0 - (CalculateMSE(features, targets) / 
                               CalculateVariance(y_double));
    
    is_trained_ = true;
    return true;
}

uint64_t LinearModel::Predict(const std::vector<double>& features) const {
    if (!is_trained_ || features.size() != feature_dimensions_) {
        return 0;
    }
    
    double prediction = bias_;
    for (size_t i = 0; i < feature_dimensions_; ++i) {
        prediction += weights_[i] * features[i];
    }
    
    // Ensure non-negative prediction
    return static_cast<uint64_t>(std::max(0.0, prediction));
}

double LinearModel::GetConfidence(const std::vector<double>& features) const {
    if (!is_trained_ || features.size() != feature_dimensions_) {
        return 0.0;
    }
    
    // For linear model, confidence is based on training accuracy
    // In a more sophisticated implementation, this could consider
    // the distance from training data distribution
    return training_accuracy_;
}

std::vector<double> LinearModel::GetParameters() const {
    std::vector<double> params;
    params.reserve(weights_.size() + 1);
    
    // First parameter is bias, followed by weights
    params.push_back(bias_);
    params.insert(params.end(), weights_.begin(), weights_.end());
    
    return params;
}

bool LinearModel::LoadParameters(const std::vector<double>& params) {
    if (params.size() != feature_dimensions_ + 1) {
        return false;
    }
    
    bias_ = params[0];
    weights_.assign(params.begin() + 1, params.end());
    
    is_trained_ = true;
    return true;
}

size_t LinearModel::GetParameterCount() const {
    return feature_dimensions_ + 1; // weights + bias
}

bool LinearModel::SolveLeastSquares(const std::vector<std::vector<double>>& X,
                                   const std::vector<uint64_t>& y) {
    size_t n = X.size();
    size_t m = feature_dimensions_;
    
    if (n == 0 || m == 0) {
        return false;
    }
    
    // Create design matrix with bias column (X_augmented = [1, X])
    std::vector<std::vector<double>> X_aug(n, std::vector<double>(m + 1));
    for (size_t i = 0; i < n; ++i) {
        X_aug[i][0] = 1.0; // bias column
        for (size_t j = 0; j < m; ++j) {
            X_aug[i][j + 1] = X[i][j];
        }
    }
    
    // Compute X^T * X
    std::vector<std::vector<double>> XTX(m + 1, std::vector<double>(m + 1, 0.0));
    for (size_t i = 0; i < m + 1; ++i) {
        for (size_t j = 0; j < m + 1; ++j) {
            for (size_t k = 0; k < n; ++k) {
                XTX[i][j] += X_aug[k][i] * X_aug[k][j];
            }
        }
    }
    
    // Compute X^T * y
    std::vector<double> XTy(m + 1, 0.0);
    for (size_t i = 0; i < m + 1; ++i) {
        for (size_t k = 0; k < n; ++k) {
            XTy[i] += X_aug[k][i] * static_cast<double>(y[k]);
        }
    }
    
    // Solve XTX * params = XTy using Gaussian elimination
    std::vector<double> params(m + 1);
    std::vector<std::vector<double>> XTX_copy = XTX;
    std::vector<double> XTy_copy = XTy;
    if (!SolveLinearSystem(XTX_copy, XTy_copy, params)) {
        return false;
    }
    
    // Extract bias and weights
    bias_ = params[0];
    for (size_t i = 0; i < m; ++i) {
        weights_[i] = params[i + 1];
    }
    
    return true;
}

bool LinearModel::SolveLinearSystem(std::vector<std::vector<double>>& A,
                                   std::vector<double>& b,
                                   std::vector<double>& x) const {
    size_t n = A.size();
    
    // Gaussian elimination with partial pivoting
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        size_t max_row = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > std::abs(A[max_row][i])) {
                max_row = k;
            }
        }
        
        // Swap rows
        if (max_row != i) {
            std::swap(A[i], A[max_row]);
            std::swap(b[i], b[max_row]);
        }
        
        // Check for singular matrix
        if (std::abs(A[i][i]) < 1e-10) {
            return false;
        }
        
        // Eliminate
        for (size_t k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (size_t j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // Back substitution
    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    
    return true;
}

double LinearModel::CalculateMSE(const std::vector<std::vector<double>>& features,
                                const std::vector<uint64_t>& targets) const {
    if (features.size() != targets.size()) {
        return std::numeric_limits<double>::max();
    }
    
    double mse = 0.0;
    for (size_t i = 0; i < features.size(); ++i) {
        uint64_t predicted = Predict(features[i]);
        double error = static_cast<double>(predicted) - static_cast<double>(targets[i]);
        mse += error * error;
    }
    
    return mse / features.size();
}

double LinearModel::CalculateVariance(const std::vector<double>& values) const {
    if (values.empty()) {
        return 0.0;
    }
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0.0;
    
    for (double value : values) {
        double diff = value - mean;
        variance += diff * diff;
    }
    
    return variance / values.size();
}

// MLModelFactory implementation
std::unique_ptr<MLModel> MLModelFactory::CreateModel(ModelType type, 
                                                    size_t feature_dimensions) {
    switch (type) {
        case ModelType::LINEAR:
            return std::make_unique<LinearModel>(feature_dimensions);
        case ModelType::NEURAL_NET:
            // TODO: Implement neural network model
            return nullptr;
        case ModelType::POLYNOMIAL:
            // TODO: Implement polynomial model
            return nullptr;
        default:
            return nullptr;
    }
}

std::unique_ptr<MLModel> MLModelFactory::LoadModel(const LearnedIndexBlock& block) {
    if (!block.IsValid()) {
        return nullptr;
    }
    
    auto model = CreateModel(block.model_type, block.feature_dimensions);
    if (!model) {
        return nullptr;
    }
    
    if (!model->LoadParameters(block.parameters)) {
        return nullptr;
    }
    
    return model;
}

} // namespace learned_index
} // namespace rocksdb