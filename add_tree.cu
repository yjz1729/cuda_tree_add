#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define N_size 256
using namespace std;
#define THREAD_NUM 16
#define BLOCK_NUM 1

// __global__ 函数 (GPU上执行) 计算立方和
__global__ static void sumOfSquares(float *num, float* result,clock_t *time)
{

    //声明一块共享内存
    extern __shared__ int shared[];

    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    shared[tid] = 0;

    int i;
    //记录运算开始的时间
    clock_t start;
    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid] = clock();

    //thread需要同时通过tid和bid来确定，同时不要忘记保证内存连续性
    for (i = bid * THREAD_NUM + tid; i < N_size; i += THREAD_NUM*BLOCK_NUM) {

        shared[tid] += num[i] * num[i] * num[i];

    }

    //同步 保证每个 thread 都已经把结果写到 shared[tid] 里面
    __syncthreads();

    //树状加法
    int offset = 1, mask = 1;

    while (offset < 16)
    {
        if ((tid & mask) == 0)
        {
            shared[tid] += shared[tid + offset];
        }

        offset += offset;
        mask = offset + mask;
        __syncthreads();

    }

    //计算时间,记录结果，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    if (tid == 0)
    { 
        result[bid] = shared[0];
        time[bid + BLOCK_NUM] = clock(); 
    }

}

__host__ //这里是在 CPU 上运行的代码，用 __host__ 来指明
int main() {


	float h_A[N_size], h_B[N_size],h_C[N_size];
	clock_t time_used[BLOCK_NUM*2];

    for (int i = 0; i < N_size; i++) {
        h_A[i] = 1;
        h_B[i] = i;
    }

    float *d_A, *d_B, *d_C;
	clock_t *time;
    //在显卡上分配内存
    cudaMalloc((void**)&d_A, sizeof(float) * N_size);
    cudaMalloc((void**)&d_B, sizeof(float) * N_size);
    cudaMalloc((void**)&d_C, sizeof(float) * N_size);
	cudaMalloc((void**)&time,sizeof(clock_t)*BLOCK_NUM*2);
    //将内存中的数据拷贝到显卡上
    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);

    //这俩参数用来决定线程的组织方式，看下文
    dim3 DimGrid(BLOCK_NUM, 1, 1);
    dim3 DimBlock(THREAD_NUM,1,1);

    //在显卡上执行代码
    //sumOfSquares <<<DimGrid, DimBlock >>>(d_A,d_C);
	sumOfSquares <<<DimGrid,DimBlock,THREAD_NUM*sizeof(float)>>>(d_A,d_C,time);
	
    //将运算结果从显卡拷贝回来
    cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);
	cudaMemcpy(time_used,time,sizeof(time),cudaMemcpyDeviceToHost);
    for (int i = 0; i < N_size; i++) {
        cout <<h_C[i]<<"ok"<<'\n';
    }
	cout<<time_used[1]-time_used[0];


    //释放显存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


	
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
cout <<prop.name<<'\n'<<prop.sharedMemPerBlock<<'\n'<<prop.clockRate<<'\n';
//this device clockRate is 1084500

return 0;
}