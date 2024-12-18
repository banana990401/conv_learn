#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../../include/verify.h"
#include "../../include/conv_2d.h"

__global__ void implgemm(param_t param)
{
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *smemweight = reinterpret_cast<float *>(smem);
    float *smeminput = reinterpret_cast<float *>(smem + 16 * 1024);

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Warp tile
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int mma_tid_x = (lane_id / 2) % 8;
    const int mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    // lds addr
    int weight_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
    int input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

    int x = bx * 128 + input_lds_addr;
    int y = by * 128 + weight_lds_addr;
    int z = blockIdx.z;

    float weight_ldg_reg[4];
    float input_ldg_reg[4];
    // 当前线程处理的数据点在oh、ow上的坐标
    int posh_ori[4];
    int posw_ori[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / param.Ow) * param.u - param.p;
        posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % param.Ow) * param.v - param.q;
    }

    int inOffset = z * param.c * param.h * param.w;
    int weiOffset = (by * 128 + tx / 8 * 4) * param.c * param.r * param.s;
    int inChannelOffset = param.h * param.w;
    int weightChannelOffset = param.r * param.s;
    int weightKOffset = param.c * param.r * param.s;

    // sts addr
    int weight_sts_addr = (tx % 8) * 132 +
                          (tx / 8) * 4;
    int input_sts_addr = (tx / 32) * 128 + (tx % 32);

    int write_flag = 1;
    float weight_frag[2][8];
    float input_frag[2][8];
    float output_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            output_frag[i][j] = 0;
        }
    }
// ldg
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (tx % 8 < weightKOffset && by * 128 + tx / 8 * 4 + i < param.k)
        {
            weight_ldg_reg[i] = param.weight[weiOffset + tx % 8 + i * weightKOffset];
        }
        else
        {
            weight_ldg_reg[i] = 0.0;
        }
    }
    int curC = (tx / 32) / (param.r * param.s);             // channel offset
    int curR = ((tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
    int curS = ((tx / 32) % (param.r * param.s)) % param.s; // kernel s offset
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int curH = posh_ori[i] + curR; // input h
        int curW = posw_ori[i] + curS; // input w
        int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
        if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h)
        {
            input_ldg_reg[i] = param.input[inOffset + inOffsetTmp];
        }
        else
        {
            input_ldg_reg[i] = 0.0;
        }
    }
    // sts
    for (int i = 0; i < 4; ++i)
    {
        smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
    }

    __syncthreads();
    // lds
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        weight_frag[0][i] = smemweight[weight_lds_addr + i];
        weight_frag[0][i + 4] = smemweight[weight_lds_addr + i + 16];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        input_frag[0][i] = smeminput[input_lds_addr + i];
        input_frag[0][i + 4] = smeminput[input_lds_addr + i + 32];
    }
    for (int crs = 0; crs < param.r * param.s * param.c; crs += 8)
    {
        // ldg
        int weiOffsetTmp = crs + 8 + tx % 8;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (weiOffsetTmp < weightKOffset && by * 128 + tx / 8 * 4 + i < param.k)
            {
                weight_ldg_reg[i] = param.weight[weiOffset + weiOffsetTmp + i * weightKOffset];
            }
            else
            {
                weight_ldg_reg[i] = 0.0;
            }
        }
        curC = (crs + 8 + tx / 32) / (param.r * param.s);             // channel offset
        curR = ((crs + 8 + tx / 32) % (param.r * param.s)) / param.s; // kernel r offset
        curS = ((crs + 8 + tx / 32) % (param.r * param.s)) % param.s; // kernel s offset

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int curH = posh_ori[i] + curR; // input h
            int curW = posw_ori[i] + curS; // input w
            int inOffsetTmp = curC * inChannelOffset + curH * param.w + curW;
            if (curH >= 0 && curW >= 0 && curW < param.w && curH < param.h)
            {
                input_ldg_reg[i] = param.input[inOffset + inOffsetTmp];
            }
            else
            {
                input_ldg_reg[i] = 0.0;
            }
        }
        int load_flag = write_flag ^ 1;
#pragma unroll
        for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i];
                weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i + 16];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i];
                input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i + 32];
            }

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                }
            }
        }
        // sts
        for (int i = 0; i < 4; ++i)
        {
            smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        write_flag ^= 1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
            weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            input_frag[0][i] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i];
            input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i + 32];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
            }
        }
    }

    // reuse smem
    float *smemoutput = reinterpret_cast<float *>(smem);
    float *smembias = reinterpret_cast<float *>(smem + 16 * 1024);

    // bias ldg/sts
    if (tx < 128)
    {
        smembias[tx] = param.bias[by * 128 + tx];
    }

    uint32_t output_sts_addr = warp_id * 512 + mma_tid_y * 4 * 8 * 4 + mma_tid_x * 4;
    uint32_t output_lds_addr = warp_id * 512 + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            __syncthreads();

#pragma unroll
            for (int subi = 0; subi < 4; ++subi)
            {
#pragma unroll
                for (int subj = 0; subj < 4; ++subj)
                {
                    // output sts
                    smemoutput[output_sts_addr + subi * 8 * 4 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                }
            }
            __syncthreads();

#pragma unroll
            for (int subk = 0; subk < 16; ++subk)
            {
                int outOffset = z * param.k * param.Oh * param.Ow + (m_idx + i * 16 + subk) * param.Oh * param.Ow + n_idx + j * 32;
                if ((m_idx + i * 16 + subk) < param.k && (n_idx + j * 32) < param.Oh * param.Ow)
                    param.output[outOffset] = smemoutput[output_lds_addr + subk * 32] + smembias[m_idx + i * 16 + subk];
            }
        }
    }
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