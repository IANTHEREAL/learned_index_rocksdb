#pragma once

#include "benchmark_framework.h"
#include <map>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace benchmark {

// Simple B+ tree implementation for baseline comparison
class BTreeIndex : public IndexInterface {
public:
    BTreeIndex();
    ~BTreeIndex() override = default;
    
    bool Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) override;
    uint32_t Lookup(uint64_t key) override;
    size_t GetMemoryUsage() const override;
    std::string GetIndexType() const override { return "B+Tree"; }
    void GetStats(BenchmarkResult& result) const override;

private:
    struct BTreeNode {
        std::vector<uint64_t> keys;
        std::vector<uint32_t> values;
        std::vector<std::unique_ptr<BTreeNode>> children;
        bool is_leaf;
        
        BTreeNode(bool leaf = true) : is_leaf(leaf) {}
    };
    
    std::unique_ptr<BTreeNode> root_;
    size_t node_capacity_;
    size_t total_nodes_;
    mutable size_t lookup_count_;
    mutable size_t total_comparisons_;
    
    void InsertIntoNode(BTreeNode* node, uint64_t key, uint32_t value);
    BTreeNode* SplitNode(BTreeNode* node);
    uint32_t SearchInNode(const BTreeNode* node, uint64_t key) const;
    size_t CalculateNodeMemory(const BTreeNode* node) const;
};

// Alternative: Simple sorted array with binary search (more realistic baseline)
class SortedArrayIndex : public IndexInterface {
public:
    SortedArrayIndex();
    ~SortedArrayIndex() override = default;
    
    bool Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) override;
    uint32_t Lookup(uint64_t key) override;
    size_t GetMemoryUsage() const override;
    std::string GetIndexType() const override { return "SortedArray"; }
    void GetStats(BenchmarkResult& result) const override;

private:
    std::vector<std::pair<uint64_t, uint32_t>> sorted_data_;
    mutable size_t lookup_count_;
    mutable size_t total_comparisons_;
};

// Hash table baseline for comparison
class HashIndex : public IndexInterface {
public:
    HashIndex();
    ~HashIndex() override = default;
    
    bool Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) override;
    uint32_t Lookup(uint64_t key) override;
    size_t GetMemoryUsage() const override;
    std::string GetIndexType() const override { return "HashTable"; }
    void GetStats(BenchmarkResult& result) const override;

private:
    std::unordered_map<uint64_t, uint32_t> hash_table_;
    mutable size_t lookup_count_;
    mutable size_t hash_collisions_;
};

} // namespace benchmark