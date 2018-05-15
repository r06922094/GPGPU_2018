#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__device__ __host__ int LowBit(int a) { return a&(-a); }

__global__ void Count_transform(const char *input, int *output, int text_size);
__global__ void Count_Sum(int *input, int *output, int text_size);

static void Launch_Count_transform_kernel(const char *input, int *output, int text_size, size_t grid_dim, size_t block_dim){
  dim3 grid = dim3(grid_dim);
  dim3 block= dim3(block_dim);

  Count_transform<<<grid, block>>>(input, output, text_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
}

static void Launch_Count_sum_kernel(int *input, int *output, int text_size, size_t grid_dim, size_t block_dim){
  dim3 grid = dim3(grid_dim);
  dim3 block= dim3(block_dim);

  Count_Sum<<<grid, block>>>(input, output, text_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
}

void CountPosition1(const char *text, int *pos, int text_size)
{
    char *spaces;
    cudaMalloc((void **)&spaces, sizeof(char) * text_size);
    cudaMemset(spaces, 10, text_size);

    thrust::device_ptr<const char> device_text = thrust::device_pointer_cast(text);
    thrust::device_ptr<int> device_pos = thrust::device_pointer_cast(pos);
    thrust::device_ptr<const char> device_spaces = thrust::device_pointer_cast(spaces);

    thrust::transform(device_text, device_text + text_size, device_spaces, device_pos, thrust::not_equal_to<const char>());
    thrust::inclusive_scan_by_key(device_pos, device_pos + text_size, device_pos, device_pos);

    cudaFree(spaces);


}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int block_dim = 128;

    int *tmp_pos;
    cudaMalloc((void **)&tmp_pos, sizeof(int) * text_size);
    cudaMemset(tmp_pos, 0, sizeof(int) * text_size);

    Launch_Count_transform_kernel(text, tmp_pos, text_size, CeilDiv(text_size, block_dim), block_dim);
    Launch_Count_sum_kernel(tmp_pos, pos, text_size, CeilDiv(text_size, block_dim), block_dim);

}



__global__ void Count_transform(const char *input, int *output, int text_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int result = 1;
    if (tid < text_size)
    {
        if (input[tid] == 10)
        {
            result = 0;
        }
        output[tid] = result;
    }
}

__global__ void Count_Sum(int *input, int *output, int text_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < text_size)
    {
        int index = tid;
        if (index == 0)
        {
            int sum = 0;
            while (input[index] != 0)
            {
                sum++;
                output[index] = sum;
                index++;
            }
        }
        else
        {
            if (input[index - 1] == 0 && input[index] == 1)
            {
                int sum = 0;
                while (input[index] != 0)
                {
                    sum++;
                    output[index] = sum;
                    index++;
                }
            }
        }
    }
}