#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include "verify.h"
#include "conv_2d.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void implgemm(param_t param)
{
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float* smem_weight = reinterpret_cast<float*>(smem);
    float* smem_input  = reinterpret_cast<float*>(smem + 16 * 1024);

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // warp tile
    const int lane_id   = tx % 32;
    const int warp_id   = tx / 32;
    const int mma_tid_x = lane_id % 8;
    const int mma_tid_y = lane_id / 8;

    // lds addr
    int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int input_lds_addr  = (warp_id % 2) * 64 + mma_tid_x * 4;

    int z = blockIdx.z;

    float weight_ldg_reg[4];
    float input_ldg_reg[4];

    int pos_h[4];
    int pos_w[4];

#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        pos_h[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.u - param.p;
        pos_w[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.v - param.q;
    }

    int input_offset         = z * param.c * param.h * param.w;
    int weight_offset        = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int input_channel_offset = param.h * param.w;
    int weight_k_offset      = param.c * param.r * param.s;

    // sts addr
    int weight_sts_addr = (tx % 8) * 132 + (tx / 8) * 4;
    int input_sts_addr  = (tx / 32) * 128 + (tx % 32);

    int write_flag = 1;
    float weight_frag[2][8];
    float input_frag[2][8];
    float output_frag[8][8];
#pragma unroll
    for(int i = 0; i < 8; i++)
    {
#pragma unroll
        for(int j = 0; j < 8; j++)
        {
            output_frag[i][j] = 0.0f;
        }
    }

// ldg
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        if(tx % 8 < weight_k_offset && by * 128 + tx / 8 * 4 + i < param.k)
        {
            weight_ldg_reg[i] = param.weight[weight_offset + i * weight_k_offset + tx % 8];
        }
        else
        {
            weight_ldg_reg[i] = 0.0f;
        }
    }
    int cur_c = (tx / 32) / (param.r * param.s);
    int cur_r = ((tx / 32) % (param.r * param.s)) / param.s;
    int cur_s = ((tx / 32) % (param.r * param.s)) % param.s;
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        int cur_h            = pos_h[i] + cur_r;
        int cur_w            = pos_w[i] + cur_s;
        int input_offset_tmp = cur_c * input_channel_offset + cur_h * param.w + cur_w;
        if(cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w)
        {
            input_ldg_reg[i] = param.input[input_offset + input_offset_tmp];
        }
        else
        {
            input_ldg_reg[i] = 0.0f;
        }
    }
    // sts
    FLOAT4(smem_weight[weight_sts_addr]) = FLOAT4(weight_ldg_reg[0]);
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        smem_input[input_sts_addr + i * 32] = input_ldg_reg[i];
    }
    __syncthreads();

    // lds
    FLOAT4(weight_frag[0][0]) = FLOAT4(smem_weight[weight_lds_addr]);
    FLOAT4(weight_frag[0][4]) = FLOAT4(smem_weight[weight_lds_addr + 16]);

    FLOAT4(input_frag[0][0]) = FLOAT4(smem_input[input_lds_addr]);
    FLOAT4(input_frag[0][4]) = FLOAT4(smem_input[input_lds_addr + 32]);

    for(int crs = 0; crs < param.c * param.r * param.s; crs += 8)
    {
        // ldg
        int weight_offset_tmp = crs + 8 + tx % 8;
#pragma unroll
        for(int i = 0; i < 4; i++)
        {
            if(weight_offset_tmp < weight_k_offset && by * 128 + tx / 8 * 4 + i < param.k)
            {
                weight_ldg_reg[i] =
                    param.weight[weight_offset + weight_offset_tmp + i * weight_k_offset];
            }
            else
            {
                weight_ldg_reg[i] = 0.0f;
            }
        }
        cur_c = (tx / 32 + crs + 8) / (param.r * param.s);
        cur_r = ((tx / 32 + crs + 8) % (param.r * param.s)) / param.s;
        cur_s = ((tx / 32 + crs + 8) % (param.r * param.s)) % param.s;
#pragma unroll
        for(int i = 0; i < 4; i++)
        {
            int cur_h            = pos_h[i] + cur_r;
            int cur_w            = pos_w[i] + cur_s;
            int input_offset_tmp = cur_c * input_channel_offset + cur_h * param.w + cur_w;
            if(cur_h >= 0 && cur_w >= 0 && cur_h < param.h && cur_w < param.w)
            {
                input_ldg_reg[i] = param.input[input_offset + input_offset_tmp];
            }
            else
            {
                input_ldg_reg[i] = 0.0f;
            }
        }
        int load_flag = write_flag ^ 1; // 0
#pragma unroll
        for(int subcrs = 0; subcrs < 8 - 1; subcrs++)
        {
            // lds
            FLOAT4(weight_frag[(subcrs + 1) % 2][0]) =
                FLOAT4(smem_weight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132]);
            FLOAT4(weight_frag[(subcrs + 1) % 2][4]) = FLOAT4(
                smem_weight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + 16]);

            FLOAT4(input_frag[(subcrs + 1) % 2][0]) =
                FLOAT4(smem_input[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128]);
            FLOAT4(input_frag[(subcrs + 1) % 2][4]) =
                FLOAT4(smem_input[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + 32]);
// GEMM
#pragma unroll
            for(int i = 0; i < 8; i++)
            {
#pragma unroll
                for(int j = 0; j < 8; j++)
                {
                    output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                }
            }
        }
        // sts
        FLOAT4(smem_weight[write_flag * 132 * 8 + weight_sts_addr]) = FLOAT4(weight_ldg_reg[0]);

#pragma unroll
        for(int i = 0; i < 4; i++)
        {
            smem_input[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();

        write_flag ^= 1; // 0
        // lds
        FLOAT4(weight_frag[0][0]) =
            FLOAT4(smem_weight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr]);
        FLOAT4(weight_frag[0][4]) =
            FLOAT4(smem_weight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + 16]);

        FLOAT4(input_frag[0][0]) = FLOAT4(smem_input[(load_flag ^ 1) * 128 * 8 + input_lds_addr]);
        FLOAT4(input_frag[0][4]) =
            FLOAT4(smem_input[(load_flag ^ 1) * 128 * 8 + input_lds_addr + 32]);
        // GEMM
        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
            }
        }
    }

    // reuse smem
    float* smem_output = reinterpret_cast<float*>(smem); // 4096
    float* smem_bias   = reinterpret_cast<float*>(smem + 16 * 1024);

    // bias ldg/sts
    if(tx < 128)
    {
        smem_bias[tx] = param.bias[by * 128 + tx];
    }

    uint32_t output_sts_addr = warp_id * 512 + mma_tid_y * 4 * 8 * 4 + mma_tid_x * 4;
    uint32_t output_lds_addr = warp_id * 512 + lane_id;

    uint32_t m_idx = by * 128 + (warp_id / 2) * 32;
    uint32_t n_idx = bx * 128 + (warp_id % 2) * 64 + lane_id;

#pragma unroll
    for(int i = 0; i < 2; i++)
    {
#pragma unroll
        for(int j = 0; j < 2; j++)
        {
            __syncthreads();
#pragma unroll
            for(int subi = 0; subi < 4; subi++)
            {
#pragma unroll
                for(int subj = 0; subj < 4; subj++)
                {
                    smem_output[output_sts_addr + subi * 8 * 4 + subj] =
                        output_frag[i * 4 + subi][j * 4 + subj];
                }
            }
            __syncthreads();

#pragma unroll
            for(int k = 0; k < 16; k++)
            {
                int out_offset = z * param.k * param.Oh * param.Ow +
                                 (m_idx + i * 16 + k) * param.Oh * param.Ow + n_idx + j * 32;
                if((m_idx + i * 16 + k) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                {
                    param.output[out_offset] = smem_output[output_lds_addr + k * 32] +
                                               smem_bias[(warp_id / 2) * 32 + i * 16 + k];
                }
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

    int block_x  = ((out_h * out_w + 127) / 128);
    int block_y  = (k + 127) / 128;
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
        bias[i] = (rand() % 255) / 255.0;
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