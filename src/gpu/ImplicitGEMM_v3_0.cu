#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../../include/verify.h"
#include "../../include/conv_2d.h"

__global__ void implgemm(param_t param)
{
    uint32_t tx = threadIdx.x;
    int bx      = blockIdx.x;
    int by      = blockIdx.y;

    // warp tile
    const uint32_t lane_id   = tx % 32;
    const uint32_t warp_id   = tx / 32;
    const uint32_t mma_tid_x = lane_id % 8;
    const uint32_t mma_tid_y = lane_id / 8;
    // lds addr
    uint32_t weight_lds_addr = (warp_id / 2) * 16 + mma_tid_y * 4;
    uint32_t input_lds_addr  = (warp_id % 2) * 32 + mma_tid_x * 4;

    int x = bx * 64 + input_lds_addr;
    int y = by * 64 + weight_lds_addr;
    int z = blockIdx.z;

    __shared__ float smem_input[4 * 64];
    __shared__ float smem_weight[4 * 64];

    int pos_oh = (bx * 64 + tx % 64) / param.Ow;
    int pos_ow = (bx * 64 + tx % 64) % param.Ow;

    int pos_h = pos_oh * param.u - param.p;
    int pos_w = pos_ow * param.v - param.q;

    int in_offset     = z * param.c * param.h * param.w;
    int weight_offset = (by * 64 + tx / 4) * param.c * param.r * param.s;

    int in_channel_offset     = param.h * param.w;

    // sts addr(ld：取数据单元，st：存数据单元)
    uint32_t weight_sts_addr = (tx % 4) * 64 + (tx / 4);
    uint32_t input_sts_addr  = (tx / 64) * 64 + (tx % 64);

    float output_frag[4][4];
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
#pragma unroll
        for(int j = 0; j < 4; j++)
        {
            output_frag[i][j] = 0.0f;
        }
    }

    for(int crs = 0; crs < param.c * param.r * param.s; crs += 4)
    {
        int weight_offset_tmp        = crs + tx % 4;
        smem_weight[weight_sts_addr] = param.weight[weight_offset + weight_offset_tmp];

        int cur_c         = (crs + tx / 64) / (param.r * param.s);
        int cur_r         = ((crs + tx / 64) % (param.r * param.s)) / param.s;
        int cur_s         = ((crs + tx / 64) % (param.r * param.s)) % param.s;
        int cur_h         = pos_h + cur_r;
        int cur_w         = pos_w + cur_s;
        int in_offset_tmp = cur_c * in_channel_offset + cur_h * param.w + cur_w;
        if(cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w)
        {
            smem_input[input_sts_addr] = param.input[in_offset + in_offset_tmp];
        }
        else
        {
            smem_input[input_sts_addr] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for(int i = 0; i < 4; i++)
        {
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
#pragma unroll
                for(int subcrs = 0; subcrs < 4; subcrs++)
                {
                    output_frag[i][j] += smem_weight[weight_lds_addr + subcrs * 64 + i] *
                                         smem_input[input_lds_addr + subcrs * 64 + j];
                }
            }
        }
        __syncthreads();
    }

    int out_offset;
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
#pragma unroll
        for(int j = 0; j < 4; j++)
        {
            out_offset = z * param.k * param.Oh * param.Ow + (y + i) * param.Oh * param.Ow + x + j;
            if(x + j < param.Oh * param.Ow && y + i < param.k)
            {
                param.output[out_offset] = output_frag[i][j];
            }
        }
    }
}

void launch_implgemm(param_t param)
{
    unsigned int n = param.n;
    unsigned int h = param.h;
    unsigned int w = param.w;
    unsigned int k = param.k;
    unsigned int r = param.r;
    unsigned int s = param.s;
    unsigned int u = param.u;
    unsigned int v = param.v;
    unsigned int p = param.p;
    unsigned int q = param.q;

    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;

    param.Oh = out_h;
    param.Ow = out_w;

    int block_x  = ((out_h * out_w + 63) / 64);
    int block_y  = (k + 63) / 64;
    int block_z  = n;
    int thread_x = 256;
    int thread_y = 1;
    int thread_z = 1;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(thread_x, thread_y, thread_z);

    implgemm<<<grid, block>>>(param);
}

int main(int argc, char** argv)
{
    unsigned int n = atoi(argv[1]);
    unsigned int c = atoi(argv[2]);
    unsigned int h = atoi(argv[3]);
    unsigned int w = atoi(argv[4]);
    unsigned int k = atoi(argv[5]);
    unsigned int r = atoi(argv[6]);
    unsigned int s = atoi(argv[7]);
    unsigned int u = atoi(argv[8]);
    unsigned int v = atoi(argv[9]);
    unsigned int p = atoi(argv[10]);
    unsigned int q = atoi(argv[11]);

    int out_h = (h - r + 2 * p) / u + 1;
    int out_w = (w - s + 2 * q) / v + 1;

    double M              = k;
    double N              = n * out_h * out_w;
    double K              = c * r * s;
    double tmp            = n * out_h * out_w * 1e-9f;
    double flops_per_conv = tmp * M * K * 2.0;

    float* input       = (float*)malloc(n * c * h * w * sizeof(float));
    float* weight      = (float*)malloc(k * c * r * s * sizeof(float));
    float* bias        = (float*)malloc(k * sizeof(float));
    float* output      = (float*)malloc(n * k * out_h * out_w * sizeof(float));
    float* output_host = (float*)malloc(n * k * out_h * out_w * sizeof(float));

    float *input_device, *weight_device, *bias_device, *output_device;
    cudaMalloc((void**)&input_device, n * c * h * w * sizeof(float));
    cudaMalloc((void**)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void**)&bias_device, k * sizeof(float));
    cudaMalloc((void**)&output_device, n * k * out_h * out_w * sizeof(float));

    for(int i = 0; i < n * c * h * w; i++)
    {
        input[i] = (rand() % 255) / 255.0;
    }

    for(int i = 0; i < k * c * r * s; i++)
    {
        weight[i] = (rand() % 255) / 255.0;
    }

    for(int i = 0; i < k; i++)
    {
        bias[i] = 0.0f;
    }

    for(int i = 0; i < n * k * out_h * out_w; i++)
    {
        output[i]      = 0.0;
        output_host[i] = 0.0;
    }

    cudaMemcpy(input_device, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_device, bias, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        output_device, output, n * k * out_h * out_w * sizeof(float), cudaMemcpyHostToDevice);

    param_t param;

    param.input  = input_device;
    param.weight = weight_device;
    param.bias   = bias_device;
    param.output = output_device;
    param.n      = n;
    param.c      = c;
    param.h      = h;
    param.w      = w;
    param.k      = k;
    param.r      = r;
    param.s      = s;
    param.u      = u;
    param.v      = v;
    param.p      = p;
    param.q      = q;
    param.Oh     = out_h;
    param.Ow     = out_w;

    launch_implgemm(param);
    cudaMemcpy(
        output_host, output_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float time_elapsed = 0.0;

    int iternum = 10;
    for(int i = 0; i < iternum; i++)
    {
        launch_implgemm(param);
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("===================start verfiy===================\n");
    direct_conv2dcpu(input, weight, bias, output, n, c, h, w, k, r, s, u, v, p, q);

    int error = 0;
    for(int i = 0; i < n * k * out_h * out_w; i++)
    {
        if(abs(output_host[i] - output[i]) > getPrecision(output[i]))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, output_host[i], output[i]);
            error++;
            break;
        }
    }
    printf("================finish,error:%d=========================\n", error);

    float timePerConv = time_elapsed / iternum;
    double gflops     = flops_per_conv / (timePerConv / 1000.0f);
    printf("%2d %2d %2d %2d %d %d %2d\n", n, h, w, c, r, s, k);
    printf("time: %f ms\n", timePerConv);
    printf("Performance :%f GFlops\n", gflops);

    cudaFree(input_device);
    cudaFree(weight_device);
    cudaFree(output_device);

    free(input);
    free(weight);
    free(output);
    free(output_host);

    return 0;
}