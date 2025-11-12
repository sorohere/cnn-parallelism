#include "performance_metrics.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

void metrics_init(PerformanceMetrics* metrics) {
    memset(metrics, 0, sizeof(PerformanceMetrics));
    metrics->min_latency_ms = 1e9;
    metrics->max_latency_ms = 0.0;
}

double get_current_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

uint64_t get_memory_usage_bytes(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return (uint64_t)usage.ru_maxrss;
#else
    return (uint64_t)usage.ru_maxrss * 1024;
#endif
}

void metrics_calculate_derived(PerformanceMetrics* metrics, double serial_time) {
    if (metrics->total_images > 0) {
        metrics->accuracy = (metrics->correct_predictions * 100.0) / metrics->total_images;
        metrics->throughput_images_per_sec = metrics->total_images / metrics->inference_time;
        metrics->avg_latency_per_image_ms = (metrics->inference_time * 1000.0) / metrics->total_images;
    }
    
    if (serial_time > 0 && metrics->inference_time > 0) {
        metrics->speedup = serial_time / metrics->inference_time;
        if (metrics->num_processes > 0) {
            metrics->parallel_efficiency = (metrics->speedup / metrics->num_processes) * 100.0;
        }
    }
    
    metrics->peak_memory_bytes = get_memory_usage_bytes();
}

void metrics_print(const PerformanceMetrics* metrics, const char* implementation_name) {
    printf("\n");
    printf("========================================================================\n");
    printf("  %s - PERFORMANCE SUMMARY\n", implementation_name);
    printf("========================================================================\n");
    printf("  Execution Metrics:\n");
    printf("    Total Time:              %.3f seconds\n", metrics->total_time);
    printf("    Inference Time:          %.3f seconds\n", metrics->inference_time);
    printf("    Model Load Time:         %.3f seconds\n", metrics->load_model_time);
    printf("    Data Load Time:          %.3f seconds\n", metrics->load_data_time);
    
    if (metrics->communication_time > 0) {
        printf("    Communication Time:      %.3f seconds (%.1f%%)\n", 
               metrics->communication_time,
               (metrics->communication_time / metrics->total_time) * 100.0);
        printf("      - MPI Send Time:       %.3f seconds\n", metrics->mpi_send_time);
        printf("      - MPI Recv Time:       %.3f seconds\n", metrics->mpi_recv_time);
        printf("      - MPI Wait Time:       %.3f seconds\n", metrics->mpi_wait_time);
    }
    
    printf("\n  Layer-wise Timing:\n");
    if (metrics->conv1_time > 0) {
        printf("    Conv1 Layer:             %.3f seconds (%.1f%%)\n", 
               metrics->conv1_time, (metrics->conv1_time / metrics->inference_time) * 100.0);
    }
    if (metrics->conv2_time > 0) {
        printf("    Conv2 Layer:             %.3f seconds (%.1f%%)\n", 
               metrics->conv2_time, (metrics->conv2_time / metrics->inference_time) * 100.0);
    }
    if (metrics->fc1_time > 0) {
        printf("    FC1 Layer:               %.3f seconds (%.1f%%)\n", 
               metrics->fc1_time, (metrics->fc1_time / metrics->inference_time) * 100.0);
    }
    if (metrics->fc2_time > 0) {
        printf("    FC2 Layer:               %.3f seconds (%.1f%%)\n", 
               metrics->fc2_time, (metrics->fc2_time / metrics->inference_time) * 100.0);
    }
    if (metrics->output_time > 0) {
        printf("    Output Layer:            %.3f seconds (%.1f%%)\n", 
               metrics->output_time, (metrics->output_time / metrics->inference_time) * 100.0);
    }
    
    printf("\n  Throughput & Latency:\n");
    printf("    Throughput:              %.2f images/second\n", metrics->throughput_images_per_sec);
    printf("    Avg Latency per Image:   %.3f ms\n", metrics->avg_latency_per_image_ms);
    printf("    Min Latency:             %.3f ms\n", metrics->min_latency_ms);
    printf("    Max Latency:             %.3f ms\n", metrics->max_latency_ms);
    
    printf("\n  Memory Usage:\n");
    printf("    Peak Memory:             %.2f MB\n", metrics->peak_memory_bytes / (1024.0 * 1024.0));
    
    if (metrics->num_processes > 1) {
        printf("\n  Parallelization Metrics:\n");
        printf("    Number of Processes:     %d\n", metrics->num_processes);
        printf("    Speedup:                 %.2fx\n", metrics->speedup);
        printf("    Parallel Efficiency:     %.2f%%\n", metrics->parallel_efficiency);
        
        if (metrics->load_imbalance > 0) {
            printf("    Load Imbalance:          %.2f%%\n", metrics->load_imbalance * 100.0);
        }
    }
    
    if (metrics->bytes_sent > 0 || metrics->bytes_received > 0) {
        printf("\n  Communication Volume:\n");
        printf("    Data Sent:               %.2f MB\n", metrics->bytes_sent / (1024.0 * 1024.0));
        printf("    Data Received:           %.2f MB\n", metrics->bytes_received / (1024.0 * 1024.0));
        printf("    Total Data Transfer:     %.2f MB\n", 
               (metrics->bytes_sent + metrics->bytes_received) / (1024.0 * 1024.0));
    }
    
    printf("\n  Accuracy:\n");
    printf("    Correct Predictions:     %d / %d\n", metrics->correct_predictions, metrics->total_images);
    printf("    Accuracy:                %.2f%%\n", metrics->accuracy);
    printf("========================================================================\n\n");
}

void metrics_print_detailed(const PerformanceMetrics* metrics, const char* implementation_name) {
    metrics_print(metrics, implementation_name);
    
    printf("DETAILED ANALYSIS:\n");
    printf("------------------\n");
    
    if (metrics->num_processes > 1) {
        double computation_time = metrics->inference_time - metrics->communication_time;
        printf("  Computation Time:        %.3f seconds (%.1f%%)\n", 
               computation_time, (computation_time / metrics->inference_time) * 100.0);
        printf("  Communication Overhead:  %.3f seconds (%.1f%%)\n", 
               metrics->communication_time, (metrics->communication_time / metrics->inference_time) * 100.0);
        
        double ideal_time = metrics->inference_time / metrics->speedup * metrics->num_processes;
        printf("  Ideal Time (Perfect Scaling): %.3f seconds\n", ideal_time);
        printf("  Scaling Loss:            %.3f seconds\n", metrics->inference_time - ideal_time);
    }
    
    printf("\n  Time Distribution:\n");
    double layer_total = metrics->conv1_time + metrics->conv2_time + 
                         metrics->fc1_time + metrics->fc2_time + metrics->output_time;
    if (layer_total > 0) {
        printf("    Convolutional Layers:    %.1f%%\n", 
               ((metrics->conv1_time + metrics->conv2_time) / layer_total) * 100.0);
        printf("    Fully Connected Layers:  %.1f%%\n", 
               ((metrics->fc1_time + metrics->fc2_time + metrics->output_time) / layer_total) * 100.0);
    }
    
    printf("\n");
}

void print_comparison_table(PerformanceMetrics* serial, PerformanceMetrics* data_parallel[], 
                            int num_data_parallel, PerformanceMetrics* pipeline) {
    printf("\n");
    printf("================================================================================\n");
    printf("                    COMPREHENSIVE PERFORMANCE COMPARISON                       \n");
    printf("================================================================================\n\n");
    
    printf("%-20s | %10s | %12s | %10s | %12s | %10s\n", 
           "Implementation", "Time (s)", "Throughput", "Speedup", "Efficiency", "Memory (MB)");
    printf("--------------------------------------------------------------------------------\n");
    
    printf("%-20s | %10.3f | %9.2f/s | %10.2fx | %11.1f%% | %10.2f\n",
           "Serial (Baseline)", 
           serial->inference_time,
           serial->throughput_images_per_sec,
           1.0,
           100.0,
           serial->peak_memory_bytes / (1024.0 * 1024.0));
    
    for (int i = 0; i < num_data_parallel; i++) {
        char name[50];
        snprintf(name, sizeof(name), "Data Parallel (%dP)", data_parallel[i]->num_processes);
        printf("%-20s | %10.3f | %9.2f/s | %10.2fx | %11.1f%% | %10.2f\n",
               name,
               data_parallel[i]->inference_time,
               data_parallel[i]->throughput_images_per_sec,
               data_parallel[i]->speedup,
               data_parallel[i]->parallel_efficiency,
               data_parallel[i]->peak_memory_bytes / (1024.0 * 1024.0));
    }
    
    if (pipeline) {
        printf("%-20s | %10.3f | %9.2f/s | %10.2fx | %11.1f%% | %10.2f\n",
               "Pipeline (5P)",
               pipeline->inference_time,
               pipeline->throughput_images_per_sec,
               pipeline->speedup,
               pipeline->parallel_efficiency,
               pipeline->peak_memory_bytes / (1024.0 * 1024.0));
    }
    
    printf("================================================================================\n\n");
    
    printf("KEY INSIGHTS:\n");
    printf("-------------\n");
    
    if (num_data_parallel > 0) {
        PerformanceMetrics* best_dp = data_parallel[0];
        for (int i = 1; i < num_data_parallel; i++) {
            if (data_parallel[i]->speedup > best_dp->speedup) {
                best_dp = data_parallel[i];
            }
        }
        printf("  • Best Data Parallel: %dP with %.2fx speedup (%.1f%% efficiency)\n",
               best_dp->num_processes, best_dp->speedup, best_dp->parallel_efficiency);
    }
    
    if (num_data_parallel > 1) {
        double efficiency_drop = data_parallel[0]->parallel_efficiency - 
                                 data_parallel[num_data_parallel-1]->parallel_efficiency;
        printf("  • Efficiency drops by %.1f%% as process count increases\n", efficiency_drop);
    }
    
    if (pipeline) {
        printf("  • Pipeline has %.1f%% communication overhead\n", 
               (pipeline->communication_time / pipeline->inference_time) * 100.0);
        printf("  • Pipeline efficiency limited to %.1f%% due to sequential dependencies\n",
               pipeline->parallel_efficiency);
    }
    
    printf("\n");
}

