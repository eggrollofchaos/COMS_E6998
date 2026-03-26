/// Wei Alexander Xin - wax1
///
/// c1.cu: Basic Convolution in CUDA (no tiling, no shared memory)
/// For COMS E6998 Spring 2026
///
/// Input:  I[C][H][W]         where C=3, H=1024, W=1024
/// Filter: F[K][C][FH][FW]    where K=64, C=3, FH=3, FW=3
/// Output: O[K][H][W]         (with padding P=1, stride 1)
/// All double precision.
///
/// Convolution (not cross-correlation):
///   O[k,x,y] = sum_c sum_j sum_i F[k,c,FW-1-i,FH-1-j] * I'[c,x+i,y+j]
///

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define C_IN  3
#define K_OUT 64
#define H     1024
#define W     1024
#define FH    3
#define FW    3
#define P     1   // padding

// One thread per output element, no tiling/ shared mem, just come correct
__global__ void conv2d_basic(
    const double* I,   // [C_IN][H+2P][W+2P] (padded input)
    const double* F,   // [K_OUT][C_IN][FH][FW]
    double* O,         // [K_OUT][H][W]
    int h, int w, int hp, int wp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K_OUT * h * w;
    if (idx >= total) return;

    int k = idx / (h * w);
    int rem = idx % (h * w);
    int x = rem / w;
    int y = rem % w;

    double val = 0.0;
    for (int c = 0; c < C_IN; c++) {
        for (int j = 0; j < FH; j++) {
            for (int i = 0; i < FW; i++) {
                double f_val = F[k * (C_IN * FH * FW) + c * (FH * FW) + (FH - 1 - j) * FW + (FW - 1 - i)];
                double i_val = I[c * (hp * wp) + (x + j) * wp + (y + i)];
                val += f_val * i_val;
            }
        }
    }
    O[k * (h * w) + x * w + y] = val;
}

int main() {
    int hp = H + 2 * P;  // padded height
    int wp = W + 2 * P;  // padded width

    size_t I_size = C_IN * hp * wp * sizeof(double);
    size_t F_size = K_OUT * C_IN * FH * FW * sizeof(double);
    size_t O_size = K_OUT * H * W * sizeof(double);

    // Allocate host memory
    double* h_I = (double*)calloc(C_IN * hp * wp, sizeof(double));  // zero-padded
    double* h_F = (double*)malloc(F_size);
    double* h_O = (double*)malloc(O_size);

    // Generate input: I[c,x,y] = c*(x+y), stored in padded array
    for (int c = 0; c < C_IN; c++)
        for (int x = 0; x < H; x++)
            for (int y = 0; y < W; y++)
                h_I[c * (hp * wp) + (x + P) * wp + (y + P)] = (double)(c * (x + y));

    // Generate filters: F[k,c,i,j] = (c+k)*(i+j)
    for (int k = 0; k < K_OUT; k++)
        for (int c = 0; c < C_IN; c++)
            for (int i = 0; i < FH; i++)
                for (int j = 0; j < FW; j++)
                    h_F[k * (C_IN * FH * FW) + c * (FH * FW) + i * FW + j] = (double)((c + k) * (i + j));

    // Allocate device memory
    double *d_I, *d_F, *d_O;
    cudaMalloc(&d_I, I_size);
    cudaMalloc(&d_F, F_size);
    cudaMalloc(&d_O, O_size);

    cudaMemcpy(d_I, h_I, I_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size, cudaMemcpyHostToDevice);

    int total = K_OUT * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Warm up
    conv2d_basic<<<blocks, threads>>>(d_I, d_F, d_O, H, W, hp, wp);
    cudaDeviceSynchronize();

    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2d_basic<<<blocks, threads>>>(d_I, d_F, d_O, H, W, hp, wp);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back and compute checksum, *yawn*
    cudaMemcpy(h_O, d_O, O_size, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (long i = 0; i < (long)K_OUT * H * W; i++)
        checksum += h_O[i];

    printf("%.6e,%.3f\n", checksum, ms);

    // Cleanup
    cudaFree(d_I); cudaFree(d_F); cudaFree(d_O);
    free(h_I); free(h_F); free(h_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
