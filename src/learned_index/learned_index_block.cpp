#include "learned_index/learned_index_block.h"
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdint>

#ifdef HAVE_CRC32C_LIB
#include <crc32c/crc32c.h>
#else
// Fallback CRC32 implementation
namespace {
uint32_t crc32_fallback(const void* data, size_t length) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    
    // Simple CRC32 implementation (not optimized)
    static const uint32_t crc_table[256] = {
        0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
        0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
        0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
        0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
        0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
        0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
        0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
        0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
        0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423,
        0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
        0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
        0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
        0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
        0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
        0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
        0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
        0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
        0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
        0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA,
        0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
        0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
        0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
        0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84,
        0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
        0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB,
        0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
        0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
        0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
        0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55,
        0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
        0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28,
        0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
        0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
        0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
        0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
        0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
        0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
        0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
        0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
        0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
        0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693,
        0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
        0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
    };
    
    for (size_t i = 0; i < length; ++i) {
        crc = crc_table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
    }
    
    return crc ^ 0xFFFFFFFF;
}
}
#endif

namespace learned_index {

std::string LearnedIndexBlock::Serialize() const {
  std::ostringstream oss;
  
  // Write header
  oss.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));
  oss.write(reinterpret_cast<const char*>(&version), sizeof(version));
  oss.write(reinterpret_cast<const char*>(&model_type), sizeof(model_type));
  oss.write(reinterpret_cast<const char*>(&feature_dimensions), sizeof(feature_dimensions));
  oss.write(reinterpret_cast<const char*>(&parameter_count), sizeof(parameter_count));
  
  // Write model parameters
  for (const auto& param : parameters) {
    oss.write(reinterpret_cast<const char*>(&param), sizeof(param));
  }
  
  // Write metadata
  oss.write(reinterpret_cast<const char*>(&metadata.training_samples), sizeof(metadata.training_samples));
  oss.write(reinterpret_cast<const char*>(&metadata.training_accuracy), sizeof(metadata.training_accuracy));
  oss.write(reinterpret_cast<const char*>(&metadata.validation_accuracy), sizeof(metadata.validation_accuracy));
  oss.write(reinterpret_cast<const char*>(&metadata.training_timestamp), sizeof(metadata.training_timestamp));
  oss.write(reinterpret_cast<const char*>(&metadata.last_update_timestamp), sizeof(metadata.last_update_timestamp));
  
  // Write block predictions
  uint32_t prediction_count = static_cast<uint32_t>(block_predictions.size());
  oss.write(reinterpret_cast<const char*>(&prediction_count), sizeof(prediction_count));
  
  for (const auto& prediction : block_predictions) {
    oss.write(reinterpret_cast<const char*>(&prediction.block_index), sizeof(prediction.block_index));
    oss.write(reinterpret_cast<const char*>(&prediction.predicted_start_key), sizeof(prediction.predicted_start_key));
    oss.write(reinterpret_cast<const char*>(&prediction.predicted_end_key), sizeof(prediction.predicted_end_key));
    oss.write(reinterpret_cast<const char*>(&prediction.confidence), sizeof(prediction.confidence));
  }
  
  // Write checksum (placeholder, will be calculated separately)
  oss.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
  
  return oss.str();
}

bool LearnedIndexBlock::Deserialize(const std::string& data) {
  if (data.size() < sizeof(magic_number) + sizeof(version)) {
    return false;
  }
  
  std::istringstream iss(data);
  
  // Read header
  iss.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
  if (magic_number != LEARNED_INDEX_MAGIC_NUMBER) {
    return false;
  }
  
  iss.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != LEARNED_INDEX_VERSION) {
    return false;
  }
  
  iss.read(reinterpret_cast<char*>(&model_type), sizeof(model_type));
  iss.read(reinterpret_cast<char*>(&feature_dimensions), sizeof(feature_dimensions));
  iss.read(reinterpret_cast<char*>(&parameter_count), sizeof(parameter_count));
  
  // Read model parameters
  parameters.clear();
  parameters.reserve(parameter_count);
  for (uint32_t i = 0; i < parameter_count; ++i) {
    double param;
    iss.read(reinterpret_cast<char*>(&param), sizeof(param));
    parameters.push_back(param);
  }
  
  // Read metadata
  iss.read(reinterpret_cast<char*>(&metadata.training_samples), sizeof(metadata.training_samples));
  iss.read(reinterpret_cast<char*>(&metadata.training_accuracy), sizeof(metadata.training_accuracy));
  iss.read(reinterpret_cast<char*>(&metadata.validation_accuracy), sizeof(metadata.validation_accuracy));
  iss.read(reinterpret_cast<char*>(&metadata.training_timestamp), sizeof(metadata.training_timestamp));
  iss.read(reinterpret_cast<char*>(&metadata.last_update_timestamp), sizeof(metadata.last_update_timestamp));
  
  // Read block predictions
  uint32_t prediction_count;
  iss.read(reinterpret_cast<char*>(&prediction_count), sizeof(prediction_count));
  
  block_predictions.clear();
  block_predictions.reserve(prediction_count);
  for (uint32_t i = 0; i < prediction_count; ++i) {
    BlockPrediction prediction;
    iss.read(reinterpret_cast<char*>(&prediction.block_index), sizeof(prediction.block_index));
    iss.read(reinterpret_cast<char*>(&prediction.predicted_start_key), sizeof(prediction.predicted_start_key));
    iss.read(reinterpret_cast<char*>(&prediction.predicted_end_key), sizeof(prediction.predicted_end_key));
    iss.read(reinterpret_cast<char*>(&prediction.confidence), sizeof(prediction.confidence));
    block_predictions.push_back(prediction);
  }
  
  // Read checksum
  iss.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
  
  return !iss.fail();
}

void LearnedIndexBlock::UpdateChecksum() {
  std::string data = Serialize();
  // Calculate checksum excluding the checksum field itself
  size_t data_size = data.size() - sizeof(checksum);
#ifdef HAVE_CRC32C_LIB
  checksum = crc32c::Crc32c(data.data(), data_size);
#else
  checksum = crc32_fallback(data.data(), data_size);
#endif
}

bool LearnedIndexBlock::VerifyChecksum() const {
  std::string data = Serialize();
  size_t data_size = data.size() - sizeof(checksum);
#ifdef HAVE_CRC32C_LIB
  uint32_t calculated_checksum = crc32c::Crc32c(data.data(), data_size);
#else
  uint32_t calculated_checksum = crc32_fallback(data.data(), data_size);
#endif
  return calculated_checksum == checksum;
}

int LearnedIndexBlock::PredictBlockIndex(uint64_t key) const {
  if (!IsValid() || block_predictions.empty()) {
    return -1; // Invalid prediction
  }
  
  // For linear model, use simple linear interpolation
  if (model_type == ModelType::LINEAR && parameters.size() >= 2) {
    double slope = parameters[0];
    double intercept = parameters[1];
    double predicted_position = slope * static_cast<double>(key) + intercept;
    
    // Clamp to valid block range
    int predicted_block = static_cast<int>(predicted_position);
    predicted_block = std::max(0, std::min(predicted_block, static_cast<int>(block_predictions.size() - 1)));
    
    return predicted_block;
  }
  
  // Fallback: binary search through block predictions
  auto it = std::lower_bound(block_predictions.begin(), block_predictions.end(), key,
    [](const BlockPrediction& pred, uint64_t search_key) {
      return pred.predicted_end_key < search_key;
    });
  
  if (it != block_predictions.end()) {
    return static_cast<int>(it - block_predictions.begin());
  }
  
  return static_cast<int>(block_predictions.size() - 1);
}

double LearnedIndexBlock::GetPredictionConfidence(uint64_t key) const {
  int block_index = PredictBlockIndex(key);
  if (block_index >= 0 && block_index < static_cast<int>(block_predictions.size())) {
    return block_predictions[block_index].confidence;
  }
  return 0.0;
}

void LearnedIndexBlock::AddBlockPrediction(const BlockPrediction& prediction) {
  block_predictions.push_back(prediction);
  
  // Keep predictions sorted by start key for efficient lookups
  std::sort(block_predictions.begin(), block_predictions.end(),
    [](const BlockPrediction& a, const BlockPrediction& b) {
      return a.predicted_start_key < b.predicted_start_key;
    });
}

void LearnedIndexBlock::UpdateModelParameters(const std::vector<double>& new_parameters) {
  parameters = new_parameters;
  parameter_count = static_cast<uint32_t>(parameters.size());
  
  // Update timestamp
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  metadata.last_update_timestamp = static_cast<uint64_t>(now);
  
  // Recalculate checksum
  UpdateChecksum();
}

bool LearnedIndexBlock::IsValid() const {
  return magic_number == LEARNED_INDEX_MAGIC_NUMBER &&
         version == LEARNED_INDEX_VERSION &&
         parameter_count == parameters.size() &&
         VerifyChecksum();
}

} // namespace learned_index