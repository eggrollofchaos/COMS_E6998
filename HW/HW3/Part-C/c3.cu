/// Wei Alexander Xin - wax1
///
/// c3.cu: Convolution using cuDNN
/// For COMS E6998 Spring 2026
///
/// Uses cudnnFindConvolutionForwardAlgorithm() to pick the fastest algo.
/// (cudnnGetConvolutionForwardAlgorithm with PREFER_FASTEST was deprecated
///  in cuDNN 8+; FindAlgorithm is the recommended replacement per Ed #226)
///
/// Compile: nvcc -o c3 c3.cu -lcudnn
///

#include <cudnn.h>

#include <cstdio>
#include <cstdlib>

#define C_IN 3
#define K_OUT 64
#define H 1024
#define W 1024
#define FH 3
#define FW 3
#define P 1

#define CHECK_CUDNN(call)                                      \
  do {                                                         \
    cudnnStatus_t status = (call);                             \
    if (status != CUDNN_STATUS_SUCCESS) {                      \
      printf("cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
             cudnnGetErrorString(status));                     \
      exit(1);                                                 \
    }                                                          \
  } while (0)

int main() {
  // Data generation (same as c1/c2)
  int hp = H + 2 * P;
  int wp = W + 2 * P;

  double* h_I = (double*)calloc(C_IN * hp * wp, sizeof(double));
  double* h_F = (double*)malloc(K_OUT * C_IN * FH * FW * sizeof(double));
  double* h_O = (double*)malloc(K_OUT * H * W * sizeof(double));

  for (int c = 0; c < C_IN; c++)
    for (int x = 0; x < H; x++)
      for (int y = 0; y < W; y++)
        h_I[c * (hp * wp) + (x + P) * wp + (y + P)] = (double)(c * (x + y));

  for (int k = 0; k < K_OUT; k++)
    for (int c = 0; c < C_IN; c++)
      for (int i = 0; i < FH; i++)
        for (int j = 0; j < FW; j++)
          h_F[k * (C_IN * FH * FW) + c * (FH * FW) + i * FW + j] =
              (double)((c + k) * (i + j));

  // cuDNN handles padding itself, so pass unpadded input (N=1, C=3, H=1024, W=1024)
  size_t I_unpadded_size = 1 * C_IN * H * W * sizeof(double);
  double* h_I_unpadded = (double*)malloc(I_unpadded_size);
  for (int c = 0; c < C_IN; c++)
    for (int x = 0; x < H; x++)
      for (int y = 0; y < W; y++)
        h_I_unpadded[c * (H * W) + x * W + y] = (double)(c * (x + y));

  // Allocate device memory
  double *d_I, *d_F, *d_O;
  cudaMalloc(&d_I, I_unpadded_size);
  cudaMalloc(&d_F, K_OUT * C_IN * FH * FW * sizeof(double));
  cudaMalloc(&d_O, K_OUT * H * W * sizeof(double));

  cudaMemcpy(d_I, h_I_unpadded, I_unpadded_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, K_OUT * C_IN * FH * FW * sizeof(double),
             cudaMemcpyHostToDevice);

  // Initialize cuDNN
  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));

  // Input descriptor: N=1, C=3, H=1024, W=1024
  cudnnTensorDescriptor_t input_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_DOUBLE, 1, C_IN, H, W));

  // Filter descriptor: K=64, C=3, FH=3, FW=3
  cudnnFilterDescriptor_t filter_desc;
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(
      filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K_OUT, C_IN, FH, FW));

  // Convolution descriptor: pad=1, stride=1, dilation=1, CUDNN_CONVOLUTION mode
  cudnnConvolutionDescriptor_t conv_desc;
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

  // Output descriptor
  int out_n, out_c, out_h, out_w;
  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, input_desc, filter_desc, &out_n, &out_c, &out_h, &out_w));

  cudnnTensorDescriptor_t output_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_DOUBLE, out_n, out_c, out_h,
                                         out_w));

  // Let cuDNN pick the fastest algo w/ benchmarks internally, trust
  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perf;
  CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
      cudnn, input_desc, filter_desc, conv_desc, output_desc, 1,
      &returnedAlgoCount, &perf));

  // Allocate workspace
  size_t workspace_size = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, input_desc, filter_desc, conv_desc, output_desc, perf.algo,
      &workspace_size));
  void* d_workspace = nullptr;
  if (workspace_size > 0) cudaMalloc(&d_workspace, workspace_size);

  double alpha = 1.0, beta = 0.0;

  // Warm up
  CHECK_CUDNN(cudnnConvolutionForward(
      cudnn, &alpha, input_desc, d_I, filter_desc, d_F, conv_desc, perf.algo,
      d_workspace, workspace_size, &beta, output_desc, d_O));
  cudaDeviceSynchronize();

  // Time the kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  CHECK_CUDNN(cudnnConvolutionForward(
      cudnn, &alpha, input_desc, d_I, filter_desc, d_F, conv_desc, perf.algo,
      d_workspace, workspace_size, &beta, output_desc, d_O));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // Copy back and compute checksum
  cudaMemcpy(h_O, d_O, K_OUT * H * W * sizeof(double), cudaMemcpyDeviceToHost);

  double checksum = 0.0;
  for (long i = 0; i < (long)K_OUT * H * W; i++) checksum += h_O[i];

  printf("%.6e,%.3f\n", checksum, ms);

  // Cleanup
  if (d_workspace) cudaFree(d_workspace);
  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_O);
  free(h_I);
  free(h_I_unpadded);
  free(h_F);
  free(h_O);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(cudnn);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
