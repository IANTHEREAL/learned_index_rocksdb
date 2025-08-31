#pragma once

#include "benchmark_framework.h"
#include "../../include/learned_index/sst_learned_index_manager.h"

namespace benchmark {

class LearnedIndexAdapter : public IndexInterface {
public:
    LearnedIndexAdapter();
    ~LearnedIndexAdapter() override = default;
    
    bool Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) override;
    uint32_t Lookup(uint64_t key) override;
    size_t GetMemoryUsage() const override;
    std::string GetIndexType() const override { return "LearnedIndex"; }
    void GetStats(BenchmarkResult& result) const override;

private:
    std::unique_ptr<rocksdb::learned_index::SSTLearnedIndexManager> manager_;
    std::string sst_file_path_;
    rocksdb::learned_index::SSTLearnedIndexOptions options_;
    
    // Fallback index for keys not in learned index
    std::vector<std::pair<uint64_t, uint32_t>> training_data_;
    
    size_t CalculateManagerMemoryUsage() const;
};

} // namespace benchmark