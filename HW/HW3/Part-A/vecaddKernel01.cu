/// Wei Alexander Xin - wax1
///
/// vecAddKernel01.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// using COALESCED memory access.
///
/// Key difference from Kernel00:
///   Kernel00: thread i processes elements [i*N .. i*N+N-1] (contiguous per thread)
///             => adjacent threads access addresses N apart => NOT coalesced
///   Kernel01: thread i processes elements [i, i+stride, i+2*stride, ...]
///             => adjacent threads access adjacent addresses => COALESCED
///

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // Coalesced reads, strided access so adjacent threads hit adjacent addresses -- nice
    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < totalThreads * N; i += totalThreads) {
        C[i] = A[i] + B[i];
    }
}
