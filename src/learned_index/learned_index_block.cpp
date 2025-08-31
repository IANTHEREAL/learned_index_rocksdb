#include "learned_index/learned_index_block.h"
#include <cstring>
#include <crc32c/crc32c.h>
#include <sstream>

namespace rocksdb {
namespace learned_index {

bool LearnedIndexBlock::IsValid() const {
    if (magic_number != LEARNED_INDEX_MAGIC) {
        return false;
    }
    
    if (version != LEARNED_INDEX_VERSION) {
        return false;
    }
    
    if (parameter_count != parameters.size()) {
        return false;
    }
    
    if (feature_dimensions == 0) {
        return false;
    }
    
    // Validate model type
    switch (model_type) {
        case ModelType::LINEAR:
        case ModelType::NEURAL_NET:
        case ModelType::POLYNOMIAL:
            break;
        default:
            return false;
    }
    
    return true;
}

void LearnedIndexBlock::UpdateChecksum() {
    // Create a temporary copy without checksum for calculation
    std::string data = Serialize();
    
    // Remove the last 4 bytes (checksum) for calculation
    if (data.size() >= 4) {
        data.resize(data.size() - 4);
        checksum = crc32c::Crc32c(data.data(), data.size());
    }
}

std::string LearnedIndexBlock::Serialize() const {
    std::ostringstream oss;
    
    // Write fixed-size header
    oss.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));
    oss.write(reinterpret_cast<const char*>(&version), sizeof(version));
    oss.write(reinterpret_cast<const char*>(&model_type), sizeof(model_type));
    oss.write(reinterpret_cast<const char*>(&feature_dimensions), sizeof(feature_dimensions));
    oss.write(reinterpret_cast<const char*>(&parameter_count), sizeof(parameter_count));
    
    // Write parameters
    for (const auto& param : parameters) {
        oss.write(reinterpret_cast<const char*>(&param), sizeof(param));
    }
    
    // Write metadata
    oss.write(reinterpret_cast<const char*>(&metadata.training_samples), 
              sizeof(metadata.training_samples));
    oss.write(reinterpret_cast<const char*>(&metadata.training_accuracy), 
              sizeof(metadata.training_accuracy));
    oss.write(reinterpret_cast<const char*>(&metadata.validation_accuracy), 
              sizeof(metadata.validation_accuracy));
    oss.write(reinterpret_cast<const char*>(&metadata.training_timestamp), 
              sizeof(metadata.training_timestamp));
    oss.write(reinterpret_cast<const char*>(&metadata.update_at), 
              sizeof(metadata.update_at));
    
    // Write block predictions count
    uint32_t prediction_count = static_cast<uint32_t>(block_predictions.size());
    oss.write(reinterpret_cast<const char*>(&prediction_count), sizeof(prediction_count));
    
    // Write block predictions
    for (const auto& prediction : block_predictions) {
        oss.write(reinterpret_cast<const char*>(&prediction.block_index), 
                  sizeof(prediction.block_index));
        oss.write(reinterpret_cast<const char*>(&prediction.predicted_start_key), 
                  sizeof(prediction.predicted_start_key));
        oss.write(reinterpret_cast<const char*>(&prediction.predicted_end_key), 
                  sizeof(prediction.predicted_end_key));
        oss.write(reinterpret_cast<const char*>(&prediction.confidence), 
                  sizeof(prediction.confidence));
    }
    
    // Write checksum
    oss.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    
    return oss.str();
}

bool LearnedIndexBlock::Deserialize(const std::string& data) {
    if (data.size() < sizeof(magic_number) + sizeof(version) + 
                     sizeof(model_type) + sizeof(feature_dimensions) + 
                     sizeof(parameter_count) + sizeof(checksum)) {
        return false;
    }
    
    size_t offset = 0;
    
    // Read fixed-size header
    std::memcpy(&magic_number, data.data() + offset, sizeof(magic_number));
    offset += sizeof(magic_number);
    
    std::memcpy(&version, data.data() + offset, sizeof(version));
    offset += sizeof(version);
    
    std::memcpy(&model_type, data.data() + offset, sizeof(model_type));
    offset += sizeof(model_type);
    
    std::memcpy(&feature_dimensions, data.data() + offset, sizeof(feature_dimensions));
    offset += sizeof(feature_dimensions);
    
    std::memcpy(&parameter_count, data.data() + offset, sizeof(parameter_count));
    offset += sizeof(parameter_count);
    
    // Validate basic structure
    if (magic_number != LEARNED_INDEX_MAGIC || version != LEARNED_INDEX_VERSION) {
        return false;
    }
    
    // Read parameters
    parameters.resize(parameter_count);
    for (uint32_t i = 0; i < parameter_count; ++i) {
        if (offset + sizeof(double) > data.size()) {
            return false;
        }
        std::memcpy(&parameters[i], data.data() + offset, sizeof(double));
        offset += sizeof(double);
    }
    
    // Read metadata
    if (offset + sizeof(ModelMetadata) > data.size()) {
        return false;
    }
    
    std::memcpy(&metadata.training_samples, data.data() + offset, 
                sizeof(metadata.training_samples));
    offset += sizeof(metadata.training_samples);
    
    std::memcpy(&metadata.training_accuracy, data.data() + offset, 
                sizeof(metadata.training_accuracy));
    offset += sizeof(metadata.training_accuracy);
    
    std::memcpy(&metadata.validation_accuracy, data.data() + offset, 
                sizeof(metadata.validation_accuracy));
    offset += sizeof(metadata.validation_accuracy);
    
    std::memcpy(&metadata.training_timestamp, data.data() + offset, 
                sizeof(metadata.training_timestamp));
    offset += sizeof(metadata.training_timestamp);
    
    std::memcpy(&metadata.update_at, data.data() + offset, 
                sizeof(metadata.update_at));
    offset += sizeof(metadata.update_at);
    
    // Read block predictions
    if (offset + sizeof(uint32_t) > data.size()) {
        return false;
    }
    
    uint32_t prediction_count;
    std::memcpy(&prediction_count, data.data() + offset, sizeof(prediction_count));
    offset += sizeof(prediction_count);
    
    block_predictions.resize(prediction_count);
    for (uint32_t i = 0; i < prediction_count; ++i) {
        if (offset + sizeof(BlockPrediction) > data.size()) {
            return false;
        }
        
        std::memcpy(&block_predictions[i].block_index, data.data() + offset, 
                    sizeof(block_predictions[i].block_index));
        offset += sizeof(block_predictions[i].block_index);
        
        std::memcpy(&block_predictions[i].predicted_start_key, data.data() + offset, 
                    sizeof(block_predictions[i].predicted_start_key));
        offset += sizeof(block_predictions[i].predicted_start_key);
        
        std::memcpy(&block_predictions[i].predicted_end_key, data.data() + offset, 
                    sizeof(block_predictions[i].predicted_end_key));
        offset += sizeof(block_predictions[i].predicted_end_key);
        
        std::memcpy(&block_predictions[i].confidence, data.data() + offset, 
                    sizeof(block_predictions[i].confidence));
        offset += sizeof(block_predictions[i].confidence);
    }
    
    // Read checksum
    if (offset + sizeof(checksum) > data.size()) {
        return false;
    }
    
    std::memcpy(&checksum, data.data() + offset, sizeof(checksum));
    offset += sizeof(checksum);
    
    // Verify checksum
    std::string data_for_checksum = data.substr(0, data.size() - sizeof(checksum));
    uint32_t calculated_checksum = crc32c::Crc32c(data_for_checksum.data(), 
                                                  data_for_checksum.size());
    
    if (calculated_checksum != checksum) {
        return false;
    }
    
    return IsValid();
}

size_t LearnedIndexBlock::GetSerializedSize() const {
    size_t size = 0;
    
    // Fixed header
    size += sizeof(magic_number) + sizeof(version) + sizeof(model_type) + 
            sizeof(feature_dimensions) + sizeof(parameter_count);
    
    // Parameters
    size += parameters.size() * sizeof(double);
    
    // Metadata
    size += sizeof(ModelMetadata);
    
    // Block predictions
    size += sizeof(uint32_t); // prediction count
    size += block_predictions.size() * sizeof(BlockPrediction);
    
    // Checksum
    size += sizeof(checksum);
    
    return size;
}

} // namespace learned_index
} // namespace rocksdb