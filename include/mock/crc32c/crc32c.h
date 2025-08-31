#pragma once

#include <cstdint>
#include <cstddef>

namespace crc32c {

// Simple CRC32 implementation for testing
// In production, this would use RocksDB's existing CRC implementation
inline uint32_t Crc32c(const void* data, size_t length) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0x82F63B78;
            } else {
                crc >>= 1;
            }
        }
    }
    
    return crc ^ 0xFFFFFFFF;
}

inline uint32_t Crc32c(const char* data, size_t length) {
    return Crc32c(static_cast<const void*>(data), length);
}

} // namespace crc32c