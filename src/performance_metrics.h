#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <stdint.h>
#include <time.h>

typedef struct {
    double total_time;
    double load_model_time;
    double load_data_time;
    double inference_time;
    double communication_time;
    
    double conv1_time;
    double conv2_time;
    double fc1_time;
    double fc2_time;
    double output_time;
    
    uint64_t memory_used_bytes;
    uint64_t peak_memory_bytes;
    
    double throughput_images_per_sec;
    double avg_latency_per_image_ms;
    double min_latency_ms;
    double max_latency_ms;
    
    int num_processes;
    double parallel_efficiency;
    double speedup;
    
    double mpi_wait_time;
    double mpi_send_time;
    double mpi_recv_time;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    
    double cpu_utilization;
    double load_imbalance;
    
    int correct_predictions;
    int total_images;
    double accuracy;
} PerformanceMetrics;

void metrics_init(PerformanceMetrics* metrics);
void metrics_print(const PerformanceMetrics* metrics, const char* implementation_name);
void metrics_print_detailed(const PerformanceMetrics* metrics, const char* implementation_name);
void metrics_calculate_derived(PerformanceMetrics* metrics, double serial_time);
double get_current_time_sec(void);
uint64_t get_memory_usage_bytes(void);
void print_comparison_table(PerformanceMetrics* serial, PerformanceMetrics* data_parallel[], int num_data_parallel, PerformanceMetrics* pipeline);

#endif

