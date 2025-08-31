#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Latency Comparison Chart
workloads = ['Sequential (dataset=10000, queries=5000)', 'Sequential (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)']

learned_index_latency = [0.124789, 0.051308, 0.14745, 0.0592082, 0.143219, 0.0515074, 0.121381, 0.0602586, 0.121719, 0.0525218]

btree_latency = [0.0683002, 0.0352736, 0.0957334, 0.0382832, 0.091421, 0.0372678, 0.0452416, 0.0340618, 0.067564, 0.0388098]


x = np.arange(len(workloads))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, learned_index_latency, width, label='Learned Index', color='skyblue')
bars2 = ax.bar(x + width/2, btree_latency, width, label='B+ Tree', color='lightcoral')

ax.set_xlabel('Workload Type')
ax.set_ylabel('Average Lookup Latency (μs)')
ax.set_title('Lookup Latency Comparison: Learned Index vs B+ Tree')
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
