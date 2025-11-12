#!/usr/bin/env python3

import re
import sys
from pathlib import Path

class PerformanceAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.serial_metrics = {}
        self.data_parallel_metrics = []
        self.pipeline_metrics = {}
        
    def parse_results(self):
        with open(self.results_file, 'r') as f:
            content = f.read()
        
        serial_section = re.search(r'SERIAL INFERENCE.*?Performance Baseline', content, re.DOTALL)
        if serial_section:
            self.serial_metrics = self._extract_metrics(serial_section.group())
            self.serial_metrics['type'] = 'Serial'
            self.serial_metrics['processes'] = 1
        
        data_parallel_sections = re.findall(r'DATA PARALLEL INFERENCE.*?(?=(?:DATA PARALLEL|PIPELINE|$))', content, re.DOTALL)
        for section in data_parallel_sections:
            metrics = self._extract_metrics(section)
            if metrics:
                metrics['type'] = 'Data Parallel'
                procs = re.search(r'Number of Processes:\s+(\d+)', section)
                if procs:
                    metrics['processes'] = int(procs.group(1))
                    self.data_parallel_metrics.append(metrics)
        
        pipeline_section = re.search(r'PIPELINE.*?Total execution time:.*?\n', content, re.DOTALL)
        if pipeline_section:
            self.pipeline_metrics = self._extract_pipeline_metrics(pipeline_section.group())
            self.pipeline_metrics['type'] = 'Pipeline'
            self.pipeline_metrics['processes'] = 5
        
        # Recalculate speedup and efficiency based on serial baseline
        if self.serial_metrics and 'inference_time' in self.serial_metrics:
            serial_time = self.serial_metrics['inference_time']
            
            for metrics in self.data_parallel_metrics:
                if 'inference_time' in metrics and metrics['inference_time'] > 0:
                    metrics['speedup'] = serial_time / metrics['inference_time']
                    metrics['efficiency'] = (metrics['speedup'] / metrics['processes']) * 100.0
            
            if self.pipeline_metrics and 'inference_time' in self.pipeline_metrics and self.pipeline_metrics['inference_time'] > 0:
                self.pipeline_metrics['speedup'] = serial_time / self.pipeline_metrics['inference_time']
                self.pipeline_metrics['efficiency'] = (self.pipeline_metrics['speedup'] / self.pipeline_metrics['processes']) * 100.0
    
    def _extract_metrics(self, section):
        metrics = {}
        
        patterns = {
            'total_time': r'Total Time:\s+([\d.]+)',
            'inference_time': r'Inference Time:\s+([\d.]+)',
            'load_model_time': r'Model Load Time:\s+([\d.]+)',
            'load_data_time': r'Data Load Time:\s+([\d.]+)',
            'communication_time': r'Communication Time:\s+([\d.]+)',
            'throughput': r'Throughput:\s+([\d.]+)',
            'avg_latency': r'Avg Latency per Image:\s+([\d.]+)',
            'min_latency': r'Min Latency:\s+([\d.]+)',
            'max_latency': r'Max Latency:\s+([\d.]+)',
            'peak_memory_mb': r'Peak Memory:\s+([\d.]+)',
            'speedup': r'Speedup:\s+([\d.]+)x',
            'efficiency': r'Parallel Efficiency:\s+([\d.]+)%',
            'load_imbalance': r'Load Imbalance Factor:\s+([\d.]+)%',
            'accuracy': r'Accuracy:\s+([\d.]+)%',
            'conv1_time': r'Conv1 Layer:\s+([\d.]+)',
            'conv2_time': r'Conv2 Layer:\s+([\d.]+)',
            'fc1_time': r'FC1 Layer:\s+([\d.]+)',
            'fc2_time': r'FC2 Layer:\s+([\d.]+)',
            'output_time': r'Output Layer:\s+([\d.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, section)
            if match:
                metrics[key] = float(match.group(1))
        
        return metrics
    
    def _extract_pipeline_metrics(self, section):
        metrics = {}
        
        time_match = re.search(r'Total execution time:\s+([\d.]+)', section)
        if time_match:
            metrics['inference_time'] = float(time_match.group(1))
        
        correct_match = re.search(r'Total correct predictions:\s+(\d+)', section)
        if correct_match:
            correct = int(correct_match.group(1))
            metrics['accuracy'] = (correct / 10000.0) * 100.0
        
        return metrics
    
    def print_summary_table(self):
        print("\n" + "="*100)
        print(" " * 30 + "PERFORMANCE METRICS SUMMARY")
        print("="*100)
        
        # Check if we have valid serial baseline
        if not self.serial_metrics or self.serial_metrics.get('inference_time', 0) == 0:
            print("\n⚠ WARNING: Serial baseline metrics not found or invalid!")
            print("Speedup and efficiency calculations require serial baseline.\n")
        
        print(f"{'Implementation':<25} {'Processes':<10} {'Time(s)':<10} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<12} {'Memory(MB)':<12}")
        print("-"*100)
        
        if self.serial_metrics:
            self._print_row(self.serial_metrics)
        
        for metrics in self.data_parallel_metrics:
            self._print_row(metrics)
        
        if self.pipeline_metrics:
            self._print_row(self.pipeline_metrics)
        
        print("="*100)
    
    def _print_row(self, metrics):
        impl_name = f"{metrics['type']}"
        if metrics['processes'] > 1:
            impl_name = f"{metrics['type']} ({metrics['processes']}P)"
        
        time_str = f"{metrics.get('inference_time', 0):.3f}"
        throughput_str = f"{metrics.get('throughput', 0):.1f} img/s" if 'throughput' in metrics else "N/A"
        speedup_str = f"{metrics.get('speedup', 1.0):.2f}x"
        efficiency_str = f"{metrics.get('efficiency', 100.0):.1f}%" if 'efficiency' in metrics else "100.0%"
        memory_str = f"{metrics.get('peak_memory_mb', 0):.1f}"
        
        print(f"{impl_name:<25} {metrics['processes']:<10} {time_str:<10} {throughput_str:<15} {speedup_str:<10} {efficiency_str:<12} {memory_str:<12}")
    
    def print_detailed_analysis(self):
        print("\n" + "="*100)
        print(" " * 35 + "DETAILED ANALYSIS")
        print("="*100)
        
        print("\n1. SCALING EFFICIENCY ANALYSIS")
        print("-" * 50)
        if self.data_parallel_metrics:
            for i, metrics in enumerate(self.data_parallel_metrics):
                procs = metrics['processes']
                speedup = metrics.get('speedup', 0)
                efficiency = metrics.get('efficiency', 0)
                
                print(f"  {procs} Processes: Speedup={speedup:.2f}x, Efficiency={efficiency:.1f}%")
                
                if efficiency >= 90:
                    status = "✓ Excellent"
                elif efficiency >= 75:
                    status = "✓ Good"
                elif efficiency >= 60:
                    status = "⚠ Fair"
                else:
                    status = "✗ Poor"
                print(f"    Status: {status}")
                
                if i > 0:
                    prev = self.data_parallel_metrics[i-1]
                    prev_speedup = prev.get('speedup', 0)
                    if prev_speedup > 0:
                        scaling_factor = speedup / prev_speedup
                        print(f"    Scaling from {prev['processes']}P to {procs}P: {scaling_factor:.2f}x")
                print()
        
        print("\n2. LATENCY ANALYSIS")
        print("-" * 50)
        if self.serial_metrics:
            self._print_latency_stats("Serial", self.serial_metrics)
        for metrics in self.data_parallel_metrics:
            self._print_latency_stats(f"Data Parallel ({metrics['processes']}P)", metrics)
        
        print("\n3. COMMUNICATION OVERHEAD ANALYSIS")
        print("-" * 50)
        for metrics in self.data_parallel_metrics:
            if 'communication_time' in metrics and 'inference_time' in metrics:
                comm_pct = (metrics['communication_time'] / metrics['inference_time']) * 100
                comp_pct = 100 - comm_pct
                print(f"  Data Parallel ({metrics['processes']}P):")
                print(f"    Computation: {comp_pct:.1f}%")
                print(f"    Communication: {comm_pct:.1f}%")
                if comm_pct < 5:
                    print(f"    → Excellent (<5% overhead)")
                elif comm_pct < 15:
                    print(f"    → Good (<15% overhead)")
                else:
                    print(f"    → Significant overhead (>{comm_pct:.1f}%)")
                print()
        
        print("\n4. LOAD BALANCING ANALYSIS")
        print("-" * 50)
        for metrics in self.data_parallel_metrics:
            if 'load_imbalance' in metrics:
                imbalance = metrics['load_imbalance']
                print(f"  Data Parallel ({metrics['processes']}P): {imbalance:.1f}% imbalance")
                if imbalance < 5:
                    print(f"    ✓ Excellent load distribution")
                elif imbalance < 15:
                    print(f"    ✓ Good load distribution")
                else:
                    print(f"    ⚠ Poor load distribution - consider rebalancing")
                print()
        
        print("\n5. MEMORY EFFICIENCY ANALYSIS")
        print("-" * 50)
        if self.serial_metrics and 'peak_memory_mb' in self.serial_metrics:
            base_mem = self.serial_metrics['peak_memory_mb']
            print(f"  Serial baseline: {base_mem:.1f} MB")
            
            for metrics in self.data_parallel_metrics:
                if 'peak_memory_mb' in metrics:
                    mem = metrics['peak_memory_mb']
                    overhead = ((mem / base_mem) - 1) * 100
                    mem_per_proc = mem / metrics['processes']
                    print(f"  Data Parallel ({metrics['processes']}P): {mem:.1f} MB total, {mem_per_proc:.1f} MB/process")
                    print(f"    Memory overhead vs serial: {overhead:.1f}%")
        
        print("\n6. LAYER-WISE COMPUTATION BREAKDOWN")
        print("-" * 50)
        if self.serial_metrics:
            self._print_layer_breakdown("Serial", self.serial_metrics)
    
    def _print_latency_stats(self, name, metrics):
        if all(k in metrics for k in ['min_latency', 'max_latency', 'avg_latency']):
            print(f"  {name}:")
            print(f"    Min: {metrics['min_latency']:.3f} ms")
            print(f"    Avg: {metrics['avg_latency']:.3f} ms")
            print(f"    Max: {metrics['max_latency']:.3f} ms")
            variance = metrics['max_latency'] - metrics['min_latency']
            print(f"    Variance: {variance:.3f} ms")
            print()
    
    def _print_layer_breakdown(self, name, metrics):
        layers = ['conv1_time', 'conv2_time', 'fc1_time', 'fc2_time', 'output_time']
        layer_names = ['Conv1', 'Conv2', 'FC1', 'FC2', 'Output']
        
        total_layer_time = sum(metrics.get(l, 0) for l in layers)
        
        if total_layer_time > 0:
            print(f"  {name}:")
            for layer, layer_name in zip(layers, layer_names):
                if layer in metrics:
                    time = metrics[layer]
                    pct = (time / total_layer_time) * 100
                    print(f"    {layer_name:8s}: {time:.3f}s ({pct:.1f}%)")
            print()
    
    def generate_recommendations(self):
        print("\n" + "="*100)
        print(" " * 35 + "OPTIMIZATION RECOMMENDATIONS")
        print("="*100)
        print()
        
        if self.data_parallel_metrics:
            best = max(self.data_parallel_metrics, key=lambda x: x.get('speedup', 0))
            print(f"✓ Best configuration: Data Parallel with {best['processes']} processes")
            print(f"  Speedup: {best.get('speedup', 0):.2f}x")
            print(f"  Efficiency: {best.get('efficiency', 0):.1f}%")
            print()
            
            last = self.data_parallel_metrics[-1]
            if last.get('efficiency', 0) < 70:
                print("⚠ Efficiency drops significantly at higher process counts")
                print("  Recommendation: Use fewer processes for better efficiency")
                print(f"  Optimal: {best['processes']} processes")
                print()
        
        for metrics in self.data_parallel_metrics:
            if 'load_imbalance' in metrics and metrics['load_imbalance'] > 10:
                print(f"⚠ Load imbalance detected in {metrics['processes']}P configuration")
                print("  Recommendation: Improve workload distribution algorithm")
                print()
        
        if self.pipeline_metrics:
            if self.pipeline_metrics.get('efficiency', 0) < 50:
                print("⚠ Pipeline parallel shows low efficiency")
                print("  Reason: High communication overhead from layer-to-layer transfers")
                print("  Recommendation: Use data parallel for inference workloads")
                print()
        
        print("="*100)

def main():
    if len(sys.argv) < 2:
        results_file = Path("results/benchmark_results_detailed.txt")
    else:
        results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run the benchmark first: ./scripts/run_benchmarks_detailed.sh")
        sys.exit(1)
    
    analyzer = PerformanceAnalyzer(results_file)
    analyzer.parse_results()
    analyzer.print_summary_table()
    analyzer.print_detailed_analysis()
    analyzer.generate_recommendations()
    
    print("\nAnalysis complete!")
    print(f"Full results available in: {results_file}")

if __name__ == "__main__":
    main()

