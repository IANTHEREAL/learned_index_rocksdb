#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Accuracy Analysis Chart (Learned Index Only)
workloads = ['Sequential (dataset=10000, queries=5000)', 'Sequential (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Random (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Mixed (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Zipfian (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)', 'Temporal (dataset=10000, queries=5000)']

accuracy = [88.9, 100, 94.45, 100, 96.2933, 100, 97.22, 100, 97.776, 100]

fallback_rate = [11.1, 0, 5.55, 0, 3.70667, 0, 2.78, 0, 2.224, 0]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
bars1 = ax1.bar(workloads, accuracy, color='mediumseagreen')
ax1.set_xlabel('Workload Type')
ax1.set_ylabel('Prediction Accuracy (%)')
ax1.set_title('Learned Index Prediction Accuracy')
ax1.set_ylim(0, 100)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Fallback rate plot
bars2 = ax2.bar(workloads, fallback_rate, color='salmon')
ax2.set_xlabel('Workload Type')
ax2.set_ylabel('Fallback Rate (%)')
ax2.set_title('Learned Index Fallback Rate')
ax2.set_ylim(0, 100)

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
