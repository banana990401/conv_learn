#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

float get_data(const float* input, int c, int w, int h, int row, int col, int p, int q)
{
    row = row - p;
    col = col - q;
    if(row < 0 || row >= h || col < 0 || col >= w)
    {
        return 0;
    }

    return input[c * w * h + row * w + col];
}

void im2col(const float* input,
            const int w,
            const int h,
            const int r,
            const int s,
            const int p,
            const int q,
            const int u,
            const int v,
            float* data_col,
            const int col_w,
            const int col_h)
{
    int win_w = (w + 2 * q - r + 1) / v;
    int win_h = (h + 2 * p - s + 1) / u;

    for(int i = 0; i < col_h; i++)
    {
        int x = i % win_w;
        int y = i / win_w;
        for(int j = 0; j < col_w; j++)
        {
            int c  = j / (r * s);
            int kj = j % r;
            int ki = j / r;

            int row = y * u + ki;
            int col = x * v + kj;

            data_col[i * col_w + j] = get_data(input, c, w, h, row, col, p, q);
        }
    }
}

int main()
{
    unsigned int n     = 1;
    unsigned int c     = 1;
    unsigned int h     = 4;
    unsigned int w     = 4;
    unsigned int k     = 1;
    unsigned int r     = 3;
    unsigned int s     = 3;
    unsigned int u     = 1;
    unsigned int v     = 1;
    unsigned int p     = 0;
    unsigned int q     = 0;
    unsigned int col_w = r * s;
    unsigned int col_h = (r - 1) * (s - 1);

    int outh = (h + 2 * p - r) / u + 1;
    int outw = (w + 2 * q - s) / v + 1;

    float* input    = (float*)malloc(n * c * h * w * sizeof(float));
    float* weight   = (float*)malloc(k * c * r * s * sizeof(float));
    float* output   = (float*)malloc(n * k * outh * outw * sizeof(float));
    float* data_col = (float*)malloc(col_w * col_h * sizeof(float));

    float fixed_input[] = {1, 2, 0, 1, 3, 1, 1, 0, 0, 1, 2, 3, 2, 0, 1, 1};

    float fixed_filter[] = {1, 0, -1, 1, 0, -1, 1, 0, -1};

    float fixed_col[col_w * col_h];
    for(int i = 0; i < col_w * col_h; i++)
    {
        fixed_col[i] = 0.0f;
    }

    std::copy(fixed_input, fixed_input + n * c * h * w, input);
    std::copy(fixed_filter, fixed_filter + k * c * r * s, weight);
    std::copy(fixed_col, fixed_col + col_w * col_h, data_col);

    im2col(input, w, h, r, s, p, q, u, v, data_col, col_w, col_h);

    for(int i = 0; i < outh * outw; i++)
    {
        for(int j = 0; j < n; j++)
        {
            float tmp = 0.0f;
            for(int k = 0; k < c * r * s; k++)
            {
                tmp += data_col[i * c * r * s + k] * weight[k * n + j];
            }
            output[i * n + j] = tmp;
        }
    }

    std::cout << "data col:" << std::endl;
    for(int i = 0; i < outh * outw; i++)
    {
        for(int k = 0; k < c * r * s; k++)
        {
            std::cout << data_col[i * c * r * s + k] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "filter" << std::endl;

    for(int k = 0; k < c * r * s; k++)
    {
        for(int j = 0; j < n; j++)
        {
            std::cout << weight[k * n + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "output:" << std::endl;
    for(int i = 0; i < col_h; i++)
    {
        for(int j = 0; j < n; j++)
        {
            std::cout << output[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    free(input);
    free(weight);
    free(output);
    free(data_col);
    return 0;
}
