/// Wei Alexander Xin - wax1
///
/// c2.cu: Tiled Convolution with Shared Memory
/// For COMS E6998 Spring 2026
///
/// Same convolution as c1.cu but uses shared memory tiling
/// to reduce global memory access.
///
/// Strategy: Each thread block loads a tile of the padded input
/// into shared memory (including halo for the 3x3 filter), then
/// each thread computes one output element from shared memory.
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
#define P     1

#define TILE_W 16
#define TILE_H 16
// Shared memory tile includes halo
#define SHARED_W (TILE_W + FW - 1)  // 18
#define SHARED_H (TILE_H + FH - 1)  // 18

// Tiled convolution kernel: grid over (output tiles in x, output tiles in y, K_OUT)
__global__ void conv2d_tiled(
    const double* I,                // [C_IN][H+2P][W+2P]
    const double* F,                // [K_OUT][C_IN][FH][FW]
    double* O,                      // [K_OUT][H][W]
    int h, int w, int hp, int wp)
{
    int tx = threadIdx.x;           // 0..TILE_W-1
    int ty = threadIdx.y;           // 0..TILE_H-1
    int k  = blockIdx.z;            // output channel

    // Output coordinates are?
    int out_x = blockIdx.y * TILE_H + ty;
    int out_y = blockIdx.x * TILE_W + tx;

    double val = 0.0;

    // 18x18 shared tile (16 output + 2 halo)
    // load once, reuse for every filter tap
    __shared__ double sI[SHARED_H][SHARED_W];

    for (int c = 0; c < C_IN; c++) {
        // Cooperatively load the tile, iterate
        int tid = ty * TILE_W + tx;
        int shared_size = SHARED_H * SHARED_W;
        for (int idx = tid; idx < shared_size; idx += TILE_H * TILE_W) {
            int si = idx / SHARED_W;
            int sj = idx % SHARED_W;
            // Map to padded input coordinates
            int gi = blockIdx.y * TILE_H + si;
            int gj = blockIdx.x * TILE_W + sj;
            if (gi < hp && gj < wp)
                sI[si][sj] = I[c * (hp * wp) + gi * wp + gj];
            else
                sI[si][sj] = 0.0;
        }
        __syncthreads();

        // Compute convolution!
        if (out_x < h && out_y < w) {
            for (int j = 0; j < FH; j++) {
                for (int i = 0; i < FW; i++) {
                    double f_val = F[k * (C_IN * FH * FW) + c * (FH * FW)
                                     + (FH - 1 - j) * FW + (FW - 1 - i)];
                    val += f_val * sI[ty + j][tx + i];
                }
            }
        }
        __syncthreads();
    }

    if (out_x < h && out_y < w)
        O[k * (h * w) + out_x * w + out_y] = val;
}

int main() {
    int hp = H + 2 * P;
    int wp = W + 2 * P;

    size_t I_size = C_IN * hp * wp * sizeof(double);
    size_t F_size = K_OUT * C_IN * FH * FW * sizeof(double);
    size_t O_size = K_OUT * H * W * sizeof(double);

    double* h_I = (double*)calloc(C_IN * hp * wp, sizeof(double));
    double* h_F = (double*)malloc(F_size);
    double* h_O = (double*)malloc(O_size);

    // Generate input: I[c,x,y] = c*(x+y)
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

    double *d_I, *d_F, *d_O;
    cudaMalloc(&d_I, I_size);
    cudaMalloc(&d_F, F_size);
    cudaMalloc(&d_O, O_size);

    cudaMemcpy(d_I, h_I, I_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size, cudaMemcpyHostToDevice);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H, K_OUT);

    // Warm up
    conv2d_tiled<<<grid, block>>>(d_I, d_F, d_O, H, W, hp, wp);
    cudaDeviceSynchronize();

    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2d_tiled<<<grid, block>>>(d_I, d_F, d_O, H, W, hp, wp);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_O, d_O, O_size, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (long i = 0; i < (long)K_OUT * H * W; i++)
        checksum += h_O[i];

    printf("%.6e,%.3f\n", checksum, ms);

    cudaFree(d_I); cudaFree(d_F); cudaFree(d_O);
    free(h_I); free(h_F); free(h_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
