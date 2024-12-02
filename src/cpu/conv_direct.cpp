#include <iostream>
#include <cstdlib>
#include <ctime>

void direct_conv2dcpu(float* input,
                      float* filter,
                      float* bias,
                      float* output,
                      int N,
                      int C,
                      int H,
                      int W,
                      int K,
                      int R,
                      int S,
                      int U,
                      int V,
                      int P,
                      int Q)
{
    // 计算输出的高度和宽度
    int Oh = (H + 2 * P - R) / U + 1;
    int Ow = (W + 2 * Q - S) / V + 1;

    // 对每个输入样本进行卷积操作
    for(int n = 0; n < N; n++)
    {
        // 对每个输出通道进行卷积操作
        for(int k = 0; k < K; k++)
        {
            // 对输出特征图的每个位置进行卷积操作
            for(int oh = 0; oh < Oh; oh++)
            {
                for(int ow = 0; ow < Ow; ow++)
                {
                    float sum = 0.0;
                    // 对每个输入通道进行卷积操作
                    for(int c = 0; c < C; c++)
                    {
                        // 对卷积核的每个位置进行卷积操作
                        for(int r = 0; r < R; r++)
                        {
                            for(int s = 0; s < S; s++)
                            {
                                // 计算当前卷积位置在输入特征图上的对应位置
                                int ih = oh * U - P + r;
                                int iw = ow * V - Q + s;
                                // 如果当前位置在输入特征图内，则进行卷积计算
                                if(iw >= 0 && ih >= 0 && iw < W && ih < H)
                                {
                                    // 计算卷积和
                                    sum += (input[n * C * H * W + c * (W * H) + ih * W + iw] *
                                            filter[k * R * S * C + c * R * S + r * S + s]);
                                }
                            }
                        }
                    }
                    // 将卷积结果加上偏置值，存储到输出特征图中
                    output[n * K * Oh * Ow + k * Oh * Ow + oh * Ow + ow] = sum + bias[k];
                }
            }
        }
    }
}


int main()
{
    unsigned int n = 1;
    unsigned int c = 1;
    unsigned int h = 4;
    unsigned int w = 4;
    unsigned int k = 1;
    unsigned int r = 3;
    unsigned int s = 3;
    unsigned int u = 1;
    unsigned int v = 1;
    unsigned int p = 0;
    unsigned int q = 0;

    int outh = (h + 2 * p - r) / u + 1;
    int outw = (w + 2 * q - s) / v + 1;

    double M            = k;
    double N            = n * outh * outw;
    double K            = c * r * s;
    double temp         = n * outh * outw * 1e-9f;
    double flopsPerConv = temp * M * K * 2.0;

    float* input  = (float*)malloc(n * c * h * w * sizeof(float));
    float* weight = (float*)malloc(k * c * r * s * sizeof(float));
    float* bias   = (float*)malloc(k * sizeof(float));
    float* output = (float*)malloc(n * k * outh * outw * sizeof(float));

    float fixed_input[] = {1, 2, 0, 1, 3, 1, 1, 0, 0, 1, 2, 3, 2, 0, 1, 1};

    float fixed_filter[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};

    float fixed_bias = 0.0;

    std::copy(fixed_input, fixed_input + n * c * h * w, input);
    std::copy(fixed_filter, fixed_filter + k * c * r * s, weight);
    bias[0] = fixed_bias;

    direct_conv2dcpu(input, weight, bias, output, n, c, h, w, k, r, s, u, v, p, q);

    std::cout << "Input matrix:" << std::endl;
    for(int i = 0; i < h; ++i)
    {
        for(int j = 0; j < w; ++j)
        {
            std::cout << input[i * w + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Filter matrix:" << std::endl;
    for(int i = 0; i < r; ++i)
    {
        for(int j = 0; j < s; ++j)
        {
            std::cout << weight[i * s + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Bias value: " << bias[0] << std::endl;
    std::cout << std::endl;

    std::cout << "Output matrix:" << std::endl;
    for(int i = 0; i < outh; ++i)
    {
        for(int j = 0; j < outw; ++j)
        {
            std::cout << output[i * outw + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    free(input);
    free(weight);
    free(bias);
    free(output);

    return 0;
}
