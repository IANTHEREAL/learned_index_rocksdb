#include <gtest/gtest.h>
#include "learned_index/ml_model.h"
#include <vector>
#include <random>

using namespace rocksdb::learned_index;

class LinearModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_ = std::make_unique<LinearModel>(1);
    }
    
    std::unique_ptr<LinearModel> model_;
};

TEST_F(LinearModelTest, Constructor) {
    EXPECT_EQ(model_->GetFeatureDimensions(), 1U);
    EXPECT_EQ(model_->GetType(), ModelType::LINEAR);
    EXPECT_FALSE(model_->IsValid()); // Not trained yet
    EXPECT_EQ(model_->GetParameterCount(), 2U); // bias + 1 weight
}

TEST_F(LinearModelTest, TrainingSimpleLinearData) {
    // Create simple linear training data: y = 2x + 1
    std::vector<std::vector<double>> features;
    std::vector<uint64_t> targets;
    
    for (int i = 0; i < 100; ++i) {
        double x = static_cast<double>(i);
        features.push_back({x});
        targets.push_back(static_cast<uint64_t>(2.0 * x + 1.0));
    }
    
    EXPECT_TRUE(model_->Train(features, targets));
    EXPECT_TRUE(model_->IsValid());
    EXPECT_GT(model_->GetTrainingAccuracy(), 0.99); // Should be very accurate for perfect linear data
}

TEST_F(LinearModelTest, TrainingWithNoise) {
    // Create linear data with noise
    std::vector<std::vector<double>> features;
    std::vector<uint64_t> targets;
    
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<> noise(0.0, 1.0);
    
    for (int i = 0; i < 200; ++i) {
        double x = static_cast<double>(i);
        features.push_back({x});
        double y = 3.0 * x + 5.0 + noise(gen);
        targets.push_back(static_cast<uint64_t>(std::max(0.0, y)));
    }
    
    EXPECT_TRUE(model_->Train(features, targets));
    EXPECT_TRUE(model_->IsValid());
    EXPECT_GT(model_->GetTrainingAccuracy(), 0.8); // Should be reasonably accurate despite noise
}

TEST_F(LinearModelTest, PredictionAfterTraining) {
    // Train with simple data
    std::vector<std::vector<double>> features = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    std::vector<uint64_t> targets = {10, 20, 30, 40, 50};
    
    EXPECT_TRUE(model_->Train(features, targets));
    
    // Test predictions
    uint64_t pred1 = model_->Predict({6.0});
    uint64_t pred2 = model_->Predict({7.0});
    
    EXPECT_GT(pred1, 50U); // Should be greater than largest training target
    EXPECT_GT(pred2, pred1); // Should be monotonically increasing
}

TEST_F(LinearModelTest, ParameterSerialization) {
    // Train the model
    std::vector<std::vector<double>> features = {{1.0}, {2.0}, {3.0}};
    std::vector<uint64_t> targets = {2, 4, 6};
    
    EXPECT_TRUE(model_->Train(features, targets));
    
    // Get parameters
    auto params = model_->GetParameters();
    EXPECT_EQ(params.size(), 2U); // bias + weight
    
    // Create new model and load parameters
    LinearModel new_model(1);
    EXPECT_TRUE(new_model.LoadParameters(params));
    EXPECT_TRUE(new_model.IsValid());
    
    // Predictions should be identical
    std::vector<double> test_features = {5.0};
    EXPECT_EQ(model_->Predict(test_features), new_model.Predict(test_features));
}

TEST_F(LinearModelTest, InvalidTrainingData) {
    // Empty data
    EXPECT_FALSE(model_->Train({}, {}));
    
    // Mismatched sizes
    std::vector<std::vector<double>> features = {{1.0}, {2.0}};
    std::vector<uint64_t> targets = {1, 2, 3};
    EXPECT_FALSE(model_->Train(features, targets));
    
    // Wrong feature dimensions
    std::vector<std::vector<double>> wrong_features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<uint64_t> valid_targets = {1, 2};
    EXPECT_FALSE(model_->Train(wrong_features, valid_targets));
}

TEST_F(LinearModelTest, PredictionWithoutTraining) {
    // Should return 0 for untrained model
    EXPECT_EQ(model_->Predict({1.0}), 0U);
    
    // Should return 0 confidence
    EXPECT_DOUBLE_EQ(model_->GetConfidence({1.0}), 0.0);
}

TEST_F(LinearModelTest, InvalidPredictionInput) {
    // Train the model first
    std::vector<std::vector<double>> features = {{1.0}, {2.0}};
    std::vector<uint64_t> targets = {1, 2};
    EXPECT_TRUE(model_->Train(features, targets));
    
    // Wrong feature dimensions
    EXPECT_EQ(model_->Predict({1.0, 2.0}), 0U);
    EXPECT_DOUBLE_EQ(model_->GetConfidence({1.0, 2.0}), 0.0);
}

TEST_F(LinearModelTest, LoadInvalidParameters) {
    // Wrong number of parameters
    EXPECT_FALSE(model_->LoadParameters({1.0})); // Need 2 parameters
    EXPECT_FALSE(model_->LoadParameters({1.0, 2.0, 3.0})); // Too many parameters
    
    // Valid parameters should work
    EXPECT_TRUE(model_->LoadParameters({1.0, 2.0}));
    EXPECT_TRUE(model_->IsValid());
}

class MLModelFactoryTest : public ::testing::Test {};

TEST_F(MLModelFactoryTest, CreateLinearModel) {
    auto model = MLModelFactory::CreateModel(ModelType::LINEAR, 1);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->GetType(), ModelType::LINEAR);
    EXPECT_EQ(model->GetFeatureDimensions(), 1U);
}

TEST_F(MLModelFactoryTest, CreateUnsupportedModel) {
    auto neural_model = MLModelFactory::CreateModel(ModelType::NEURAL_NET, 1);
    EXPECT_EQ(neural_model, nullptr); // Not implemented yet
    
    auto poly_model = MLModelFactory::CreateModel(ModelType::POLYNOMIAL, 1);
    EXPECT_EQ(poly_model, nullptr); // Not implemented yet
}

TEST_F(MLModelFactoryTest, LoadModelFromBlock) {
    // Create a valid learned index block
    LearnedIndexBlock block;
    block.model_type = ModelType::LINEAR;
    block.feature_dimensions = 1;
    block.parameters = {1.0, 2.0}; // bias and weight
    block.parameter_count = 2;
    
    auto model = MLModelFactory::LoadModel(block);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->GetType(), ModelType::LINEAR);
    EXPECT_TRUE(model->IsValid());
}

TEST_F(MLModelFactoryTest, LoadModelFromInvalidBlock) {
    // Invalid block (unsupported model type)
    LearnedIndexBlock invalid_block;
    invalid_block.model_type = ModelType::NEURAL_NET;
    invalid_block.feature_dimensions = 1;
    invalid_block.parameters = {1.0, 2.0};
    invalid_block.parameter_count = 2;
    
    auto model = MLModelFactory::LoadModel(invalid_block);
    EXPECT_EQ(model, nullptr);
}