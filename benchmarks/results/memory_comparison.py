#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Memory Usage Comparison Chart
workloads = ['Sequential (dataset=10000, queries=5000)', 'Sequential (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)']

learned_index_memory = [156.91, 1.04297, 156.91, 1.04297, 156.91, 1.04297, 156.91, 1.04297, 156.91, 1.04297]

btree_memory = [156.297, 318.461, 156.297, 310.773, 156.297, 315.055, 156.297, 162.227, 156.297, 318.461]


x = np.arange(len(workloads))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, learned_index_memory, width, label='Learned Index', color='lightgreen')
bars2 = ax.bar(x + width/2, btree_memory, width, label='B+ Tree', color='orange')

ax.set_xlabel('Workload Type')
ax.set_ylabel('Index Memory Usage (KB)')
ax.set_title('Memory Usage Comparison: Learned Index vs B+ Tree')
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend()

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
