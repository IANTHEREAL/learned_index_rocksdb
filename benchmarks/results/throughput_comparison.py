#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Throughput Comparison Chart
workloads = ['Sequential (dataset=10000, queries=5000)', 'Sequential (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)']

learned_index_throughput = [6.12745e+06, 1.11359e+07, 5.35332e+06, 1.02249e+07, 5.51876e+06, 1.11857e+07, 6.27353e+06, 1.01626e+07, 6.26566e+06, 1.10619e+07]

btree_throughput = [9.32836e+06, 1.36612e+07, 7.48503e+06, 1.31234e+07, 7.72798e+06, 1.32979e+07, 1.19048e+07, 1.36612e+07, 9.48767e+06, 1.30548e+07]


x = np.arange(len(workloads))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, learned_index_throughput, width, label='Learned Index', color='gold')
bars2 = ax.bar(x + width/2, btree_throughput, width, label='B+ Tree', color='purple')

ax.set_xlabel('Workload Type')
ax.set_ylabel('Throughput (Queries Per Second)')
ax.set_title('Throughput Comparison: Learned Index vs B+ Tree')
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
