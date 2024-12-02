#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../../include/verify.h"
#include "../../include/conv_2d.h"

__global__ void implgemm(param_t param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if(x >= param.Oh * param.Ow || y >= param.k || z >= param.n)
    {
        return;
    }

    int pos_oh = x / param.Ow;
    int pos_ow = x % param.Oh;

    int pos_h = pos_oh * param.u - param.p;
    int pos_w = pos_ow * param.v - param.q;

    float sum = 0.0f;

    //定位输入矩阵中当前线程需要访问的第一个元素的地址
    int in_offset  = z * param.c * param.h * param.w + pos_h * param.w + pos_w;
    int wei_offset = y * param.c * param.r * param.s;
    //计算通道的偏移
    int in_channel_offset  = param.h * param.w;
    int wei_channel_offset = param.r * param.s;

    for(int i = 0; i < param.r; i++)
    {
        for(int j = 0; j < param.s; j++)
        {
            int real_pos_h = pos_h + i;
            int real_pos_w = pos_w + j;
            if(real_pos_h >= 0 && real_pos_w >= 0 && real_pos_h < param.h && real_pos_w < param.w)
            {
                int in_offset_tmp  = in_offset;
                int wei_offset_tmp = wei_offset;
                for(int channel = 0; channel < param.c; channel++)
                {
                    sum += param.input[in_offset_tmp + i * param.w + j] *
                           param.weight[wei_offset_tmp + i * param.s + j];
                    in_offset_tmp += in_channel_offset;
                    wei_offset_tmp += wei_channel_offset;
                }
            }
        }
    }

    int out_offset           = z * param.k * param.Oh * param.Ow + y * param.Oh * param.Ow + x;
    param.output[out_offset] = sum;
}

void launch_implgemm(param_t param)
{
    unsigned int n = param.n;
    unsigned int c = param.c;
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

    int block_x  = ((out_h * out_w + 15) / 16);
    int block_y  = (k + 15) / 16;
    int block_z  = n;
    int thread_x = 16;
    int thread_y = 16;
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