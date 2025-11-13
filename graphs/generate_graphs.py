import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

output_dir = Path('images')
output_dir.mkdir(exist_ok=True)

processes = [1, 2, 3, 4, 5, 6, 7, 8]
serial_time = 5.276

execution_times = {
    1: 5.269,
    2: 2.767,
    3: 1.912,
    4: 1.440,
    5: 1.296,
    6: 1.164,
    7: 1.064,
    8: 0.975
}

throughput = {
    1: 1898.05,
    2: 3613.99,
    3: 5230.17,
    4: 6942.93,
    5: 7716.50,
    6: 8590.98,
    7: 9394.28,
    8: 10252.38
}

speedup = {
    1: 1.00,
    2: 1.91,
    3: 2.76,
    4: 3.66,
    5: 4.07,
    6: 4.53,
    7: 4.96,
    8: 5.41
}

efficiency = {
    1: 100.0,
    2: 95.3,
    3: 92.0,
    4: 91.6,
    5: 81.4,
    6: 75.5,
    7: 70.8,
    8: 67.6
}

memory_usage = {
    1: 21.95,
    2: 22.39,
    3: 22.56,
    4: 22.59,
    5: 22.58,
    6: 22.53,
    7: 22.67,
    8: 22.69
}

communication_overhead = {
    2: 0.0,
    3: 0.0,
    4: 0.3,
    5: 4.7,
    6: 3.6,
    7: 1.9,
    8: 2.3
}

load_imbalance = {
    1: 0.00,
    2: 0.08,
    3: 0.09,
    4: 0.31,
    5: 4.51,
    6: 3.65,
    7: 6.87,
    8: 4.93
}

latency_avg = {
    1: 0.527,
    2: 0.277,
    3: 0.191,
    4: 0.144,
    5: 0.130,
    6: 0.116,
    7: 0.106,
    8: 0.098
}

latency_min = {
    1: 0.520,
    2: 0.548,
    3: 0.569,
    4: 0.569,
    5: 0.569,
    6: 0.569,
    7: 0.569,
    8: 0.569
}

latency_max = {
    1: 0.816,
    2: 0.977,
    3: 1.121,
    4: 1.678,
    5: 4.138,
    6: 6.324,
    7: 5.330,
    8: 16.672
}

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(processes, [execution_times[p] for p in processes], 'o-', linewidth=2.5, markersize=10, label='Actual', color='#2E86AB')
ax.plot(processes, [serial_time / p for p in processes], '--', linewidth=2, label='Ideal (Linear)', color='#A23B72', alpha=0.7)
ax.axhline(y=serial_time, color='#F18F01', linestyle=':', linewidth=2, label='Serial Baseline', alpha=0.8)
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax.set_title('Execution Time vs Number of Processes (Data Parallel)', fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xticks(processes)
for i, p in enumerate(processes):
    ax.annotate(f'{execution_times[p]:.3f}s', 
                xy=(p, execution_times[p]), 
                xytext=(0, -20), 
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
plt.tight_layout()
plt.savefig(output_dir / '01_execution_time.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(processes, [speedup[p] for p in processes], 'o-', linewidth=2.5, markersize=10, label='Actual Speedup', color='#06A77D')
ax.plot(processes, processes, '--', linewidth=2, label='Ideal (Linear) Speedup', color='#D62246', alpha=0.7)
ax.fill_between(processes, [speedup[p] for p in processes], processes, alpha=0.2, color='red', label='Scaling Gap')
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Speedup', fontweight='bold')
ax.set_title('Speedup vs Number of Processes (Data Parallel)', fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(processes)
for i, p in enumerate(processes):
    ax.annotate(f'{speedup[p]:.2f}x', 
                xy=(p, speedup[p]), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4))
plt.tight_layout()
plt.savefig(output_dir / '02_speedup.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#06A77D' if e >= 90 else '#F18F01' if e >= 75 else '#D62246' for e in [efficiency[p] for p in processes]]
bars = ax.bar(processes, [efficiency[p] for p in processes], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect Efficiency', alpha=0.6)
ax.axhline(y=75, color='orange', linestyle='--', linewidth=1.5, label='Good Threshold (75%)', alpha=0.6)
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax.set_title('Parallel Efficiency vs Number of Processes', fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(processes)
ax.set_ylim(0, 110)
for i, (p, bar) in enumerate(zip(processes, bars)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{efficiency[p]:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / '03_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(processes, [throughput[p] for p in processes], 'o-', linewidth=2.5, markersize=10, color='#7209B7')
ax.axhline(y=throughput[1], color='#F18F01', linestyle=':', linewidth=2, label='Serial Baseline', alpha=0.8)
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Throughput (images/second)', fontweight='bold')
ax.set_title('Throughput vs Number of Processes', fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(processes)
for i, p in enumerate(processes):
    ax.annotate(f'{throughput[p]:.0f}', 
                xy=(p, throughput[p]), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.5))
plt.tight_layout()
plt.savefig(output_dir / '04_throughput.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(processes, [latency_avg[p] for p in processes], 'o-', linewidth=2.5, markersize=10, label='Average Latency', color='#3A86FF')
ax.fill_between(processes, 
                [latency_min[p] for p in processes], 
                [latency_max[p] for p in processes], 
                alpha=0.2, 
                color='#3A86FF',
                label='Min-Max Range')
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Latency (milliseconds)', fontweight='bold')
ax.set_title('Latency Analysis: Average with Min-Max Range', fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(processes)
ax.set_yscale('log')
for i, p in enumerate(processes):
    ax.annotate(f'{latency_avg[p]:.3f}ms', 
                xy=(p, latency_avg[p]), 
                xytext=(0, -20), 
                textcoords='offset points',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.3))
plt.tight_layout()
plt.savefig(output_dir / '05_latency.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
proc_list = [2, 3, 4, 5, 6, 7, 8]
comm_values = [communication_overhead[p] for p in proc_list]
comp_values = [100 - communication_overhead[p] for p in proc_list]
width = 0.6
p1 = ax.bar(proc_list, comp_values, width, label='Computation', color='#06A77D', alpha=0.8)
p2 = ax.bar(proc_list, comm_values, width, bottom=comp_values, label='Communication', color='#D62246', alpha=0.8)
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Time Distribution (%)', fontweight='bold')
ax.set_title('Computation vs Communication Overhead', fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(proc_list)
ax.set_ylim(0, 105)
for i, p in enumerate(proc_list):
    if comm_values[i] > 0:
        ax.text(p, comp_values[i] + comm_values[i]/2, 
                f'{comm_values[i]:.1f}%',
                ha='center', va='center', fontweight='bold', fontsize=9, color='white')
plt.tight_layout()
plt.savefig(output_dir / '06_communication_overhead.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
colors_imbalance = ['#06A77D' if lb < 5 else '#F18F01' if lb < 15 else '#D62246' for lb in [load_imbalance[p] for p in processes]]
bars = ax.bar(processes, [load_imbalance[p] for p in processes], color=colors_imbalance, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Excellent (< 5%)', alpha=0.6)
ax.axhline(y=15, color='orange', linestyle='--', linewidth=1.5, label='Good (< 15%)', alpha=0.6)
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Load Imbalance (%)', fontweight='bold')
ax.set_title('Load Imbalance Factor vs Number of Processes', fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(processes)
for i, (p, bar) in enumerate(zip(processes, bars)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{load_imbalance[p]:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / '07_load_imbalance.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(processes, [memory_usage[p] for p in processes], 'o-', linewidth=2.5, markersize=10, color='#F72585')
ax.axhline(y=11.77, color='#4CC9F0', linestyle='--', linewidth=2, label='Serial Baseline (11.77 MB)', alpha=0.8)
ax.set_xlabel('Number of Processes', fontweight='bold')
ax.set_ylabel('Peak Memory Usage (MB)', fontweight='bold')
ax.set_title('Memory Usage vs Number of Processes', fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(processes)
for i, p in enumerate(processes):
    ax.annotate(f'{memory_usage[p]:.2f} MB', 
                xy=(p, memory_usage[p]), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', alpha=0.3))
plt.tight_layout()
plt.savefig(output_dir / '08_memory_usage.png', dpi=300, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

incremental_speedup = [1.0]
for i in range(1, len(processes)):
    inc_speedup = speedup[processes[i]] / speedup[processes[i-1]]
    incremental_speedup.append(inc_speedup)

ax1.plot(processes, incremental_speedup, 'o-', linewidth=2.5, markersize=10, color='#F15BB5')
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.set_xlabel('Number of Processes', fontweight='bold')
ax1.set_ylabel('Incremental Speedup', fontweight='bold')
ax1.set_title('Incremental Speedup (N vs N-1 processes)', fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(processes)
for i, (p, inc) in enumerate(zip(processes, incremental_speedup)):
    ax1.annotate(f'{inc:.2f}x', 
                xy=(p, inc), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9)

scaling_loss = [0]
for i in range(1, len(processes)):
    ideal_speedup = processes[i]
    actual_speedup = speedup[processes[i]]
    loss = ((ideal_speedup - actual_speedup) / ideal_speedup) * 100
    scaling_loss.append(loss)

ax2.plot(processes, scaling_loss, 'o-', linewidth=2.5, markersize=10, color='#9B5DE5')
ax2.fill_between(processes, 0, scaling_loss, alpha=0.2, color='#9B5DE5')
ax2.set_xlabel('Number of Processes', fontweight='bold')
ax2.set_ylabel('Scaling Loss (%)', fontweight='bold')
ax2.set_title('Scaling Loss from Ideal Linear Speedup', fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(processes)
for i, (p, loss) in enumerate(zip(processes, scaling_loss)):
    ax2.annotate(f'{loss:.1f}%', 
                xy=(p, loss), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '09_scaling_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 7))

implementations = ['Serial\n(1P)', 'Data Parallel\n(2P)', 'Data Parallel\n(3P)', 
                   'Data Parallel\n(4P)', 'Data Parallel\n(5P)', 'Data Parallel\n(6P)',
                   'Data Parallel\n(7P)', 'Data Parallel\n(8P)', 'Pipeline\n(5P)']
times = [5.276, 2.767, 1.912, 1.440, 1.296, 1.164, 1.064, 0.975, 3.368]
speedups = [1.0, 1.91, 2.76, 3.66, 4.07, 4.53, 4.96, 5.41, 1.57]
colors_impl = ['#F18F01', '#06A77D', '#06A77D', '#06A77D', '#06A77D', '#06A77D', '#06A77D', '#06A77D', '#3A86FF']

x_pos = np.arange(len(implementations))
bars = ax.bar(x_pos, times, color=colors_impl, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (bar, speedup_val) in enumerate(zip(bars, speedups)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}s\n({speedup_val:.2f}x)',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_xlabel('Implementation', fontweight='bold')
ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax.set_title('Performance Comparison: All Implementations', fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(implementations, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '10_implementation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
ax1.plot(processes, [execution_times[p] for p in processes], 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_title('Execution Time', fontweight='bold')
ax1.set_xlabel('Processes')
ax1.set_ylabel('Time (s)')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(processes, [speedup[p] for p in processes], 'o-', linewidth=2, markersize=8, color='#06A77D')
ax2.plot(processes, processes, '--', linewidth=1.5, color='gray', alpha=0.5, label='Ideal')
ax2.set_title('Speedup', fontweight='bold')
ax2.set_xlabel('Processes')
ax2.set_ylabel('Speedup')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(processes, [efficiency[p] for p in processes], 'o-', linewidth=2, markersize=8, color='#F18F01')
ax3.set_title('Efficiency', fontweight='bold')
ax3.set_xlabel('Processes')
ax3.set_ylabel('Efficiency (%)')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
ax4.plot(processes, [throughput[p] for p in processes], 'o-', linewidth=2, markersize=8, color='#7209B7')
ax4.set_title('Throughput', fontweight='bold')
ax4.set_xlabel('Processes')
ax4.set_ylabel('Images/sec')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
proc_comm = [2, 3, 4, 5, 6, 7, 8]
ax5.plot(proc_comm, [communication_overhead[p] for p in proc_comm], 'o-', linewidth=2, markersize=8, color='#D62246')
ax5.set_title('Communication Overhead', fontweight='bold')
ax5.set_xlabel('Processes')
ax5.set_ylabel('Overhead (%)')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
ax6.plot(processes, [load_imbalance[p] for p in processes], 'o-', linewidth=2, markersize=8, color='#F72585')
ax6.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax6.set_title('Load Imbalance', fontweight='bold')
ax6.set_xlabel('Processes')
ax6.set_ylabel('Imbalance (%)')
ax6.grid(True, alpha=0.3)

plt.suptitle('CNN Inference Performance Dashboard', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / '11_performance_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ All 11 graphs generated successfully!")
print("\nGenerated files:")
for i in range(1, 12):
    filename = f"{i:02d}_*.png"
    print(f"  - {filename}")

