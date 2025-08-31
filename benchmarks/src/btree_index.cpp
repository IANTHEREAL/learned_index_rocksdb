#include "../include/btree_index.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <cmath>

namespace benchmark {

// BTreeIndex Implementation
BTreeIndex::BTreeIndex() : node_capacity_(64), total_nodes_(0), lookup_count_(0), total_comparisons_(0) {
    root_ = std::make_unique<BTreeNode>(true);
    total_nodes_ = 1;
}

bool BTreeIndex::Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) {
    // Reset the tree
    root_ = std::make_unique<BTreeNode>(true);
    total_nodes_ = 1;
    lookup_count_ = 0;
    total_comparisons_ = 0;
    
    // Insert all key-value pairs
    for (const auto& pair : training_data) {
        InsertIntoNode(root_.get(), pair.first, pair.second);
    }
    
    return true;
}

uint32_t BTreeIndex::Lookup(uint64_t key) {
    lookup_count_++;
    return SearchInNode(root_.get(), key);
}

size_t BTreeIndex::GetMemoryUsage() const {
    return CalculateNodeMemory(root_.get());
}

void BTreeIndex::GetStats(BenchmarkResult& result) const {
    result.successful_predictions = lookup_count_;
    result.total_predictions = lookup_count_;
    result.prediction_accuracy = 1.0; // B+ tree is always accurate
    result.fallback_rate = 0.0;
}

void BTreeIndex::InsertIntoNode(BTreeNode* node, uint64_t key, uint32_t value) {
    if (node->is_leaf) {
        // Find insertion position
        auto it = std::lower_bound(node->keys.begin(), node->keys.end(), key);
        size_t pos = it - node->keys.begin();
        
        // Insert key and value
        node->keys.insert(it, key);
        node->values.insert(node->values.begin() + pos, value);
        
        // Check if node needs to be split
        if (node->keys.size() > node_capacity_) {
            BTreeNode* new_node = SplitNode(node);
            if (node == root_.get()) {
                // Create new root
                auto new_root = std::make_unique<BTreeNode>(false);
                new_root->keys.push_back(new_node->keys[0]);
                new_root->children.push_back(std::move(root_));
                new_root->children.push_back(std::unique_ptr<BTreeNode>(new_node));
                root_ = std::move(new_root);
                total_nodes_++;
            }
        }
    } else {
        // Find child to insert into
        auto it = std::upper_bound(node->keys.begin(), node->keys.end(), key);
        size_t child_index = it - node->keys.begin();
        
        if (child_index < node->children.size()) {
            InsertIntoNode(node->children[child_index].get(), key, value);
        }
    }
}

BTreeIndex::BTreeNode* BTreeIndex::SplitNode(BTreeNode* node) {
    size_t mid = node->keys.size() / 2;
    
    BTreeNode* new_node = new BTreeNode(node->is_leaf);
    total_nodes_++;
    
    // Move half the keys and values to new node
    new_node->keys.assign(node->keys.begin() + mid, node->keys.end());
    node->keys.resize(mid);
    
    if (node->is_leaf) {
        new_node->values.assign(node->values.begin() + mid, node->values.end());
        node->values.resize(mid);
    } else {
        new_node->children.assign(
            std::make_move_iterator(node->children.begin() + mid),
            std::make_move_iterator(node->children.end())
        );
        node->children.resize(mid);
    }
    
    return new_node;
}

uint32_t BTreeIndex::SearchInNode(const BTreeNode* node, uint64_t key) const {
    if (node->is_leaf) {
        // Binary search in leaf node
        auto it = std::lower_bound(node->keys.begin(), node->keys.end(), key);
        total_comparisons_ += std::log2(static_cast<double>(node->keys.size())) + 1;
        
        if (it != node->keys.end() && *it == key) {
            size_t index = it - node->keys.begin();
            return node->values[index];
        }
        return 0; // Key not found
    } else {
        // Find child to search in
        auto it = std::upper_bound(node->keys.begin(), node->keys.end(), key);
        total_comparisons_ += std::log2(static_cast<double>(node->keys.size())) + 1;
        
        size_t child_index = it - node->keys.begin();
        if (child_index < node->children.size()) {
            return SearchInNode(node->children[child_index].get(), key);
        }
        return 0; // Key not found
    }
}

size_t BTreeIndex::CalculateNodeMemory(const BTreeNode* node) const {
    size_t memory = sizeof(BTreeNode);
    memory += node->keys.size() * sizeof(uint64_t);
    memory += node->values.size() * sizeof(uint32_t);
    memory += node->children.size() * sizeof(std::unique_ptr<BTreeNode>);
    
    for (const auto& child : node->children) {
        memory += CalculateNodeMemory(child.get());
    }
    
    return memory;
}

// SortedArrayIndex Implementation
SortedArrayIndex::SortedArrayIndex() : lookup_count_(0), total_comparisons_(0) {}

bool SortedArrayIndex::Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) {
    sorted_data_ = training_data;
    
    // Sort by key
    std::sort(sorted_data_.begin(), sorted_data_.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });
    
    lookup_count_ = 0;
    total_comparisons_ = 0;
    
    return true;
}

uint32_t SortedArrayIndex::Lookup(uint64_t key) {
    lookup_count_++;
    
    // Binary search
    auto it = std::lower_bound(sorted_data_.begin(), sorted_data_.end(), 
                              std::make_pair(key, 0U),
                              [](const auto& a, const auto& b) {
                                  return a.first < b.first;
                              });
    
    total_comparisons_ += std::log2(static_cast<double>(sorted_data_.size())) + 1;
    
    if (it != sorted_data_.end() && it->first == key) {
        return it->second;
    }
    
    return 0; // Key not found, return default block
}

size_t SortedArrayIndex::GetMemoryUsage() const {
    return sorted_data_.size() * sizeof(std::pair<uint64_t, uint32_t>) + sizeof(SortedArrayIndex);
}

void SortedArrayIndex::GetStats(BenchmarkResult& result) const {
    result.successful_predictions = lookup_count_;
    result.total_predictions = lookup_count_;
    result.prediction_accuracy = 1.0; // Sorted array is always accurate
    result.fallback_rate = 0.0;
}

// HashIndex Implementation
HashIndex::HashIndex() : lookup_count_(0), hash_collisions_(0) {}

bool HashIndex::Train(const std::vector<std::pair<uint64_t, uint32_t>>& training_data) {
    hash_table_.clear();
    hash_table_.reserve(training_data.size() * 2); // Load factor ~0.5
    
    for (const auto& pair : training_data) {
        auto result = hash_table_.insert(pair);
        if (!result.second) {
            hash_collisions_++;
        }
    }
    
    lookup_count_ = 0;
    
    return true;
}

uint32_t HashIndex::Lookup(uint64_t key) {
    lookup_count_++;
    
    auto it = hash_table_.find(key);
    if (it != hash_table_.end()) {
        return it->second;
    }
    
    return 0; // Key not found
}

size_t HashIndex::GetMemoryUsage() const {
    // Approximate hash table memory usage
    size_t bucket_count = hash_table_.bucket_count();
    size_t entry_size = sizeof(std::pair<uint64_t, uint32_t>);
    return bucket_count * sizeof(void*) + hash_table_.size() * entry_size + sizeof(HashIndex);
}

void HashIndex::GetStats(BenchmarkResult& result) const {
    result.successful_predictions = lookup_count_;
    result.total_predictions = lookup_count_;
    result.prediction_accuracy = 1.0; // Hash table is always accurate for existing keys
    result.fallback_rate = 0.0;
}

} // namespace benchmark