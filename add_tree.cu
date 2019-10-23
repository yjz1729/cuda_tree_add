#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define N_size 256
using namespace std;
#define THREAD_NUM 16
#define BLOCK_NUM 1

// __global__ ���� (GPU��ִ��) ����������
__global__ static void sumOfSquares(float *num, float* result,clock_t *time)
{

    //����һ�鹲���ڴ�
    extern __shared__ int shared[];

    //��ʾĿǰ�� thread �ǵڼ��� thread���� 0 ��ʼ���㣩
    const int tid = threadIdx.x;

    //��ʾĿǰ�� thread ���ڵڼ��� block���� 0 ��ʼ���㣩
    const int bid = blockIdx.x;

    shared[tid] = 0;

    int i;
    //��¼���㿪ʼ��ʱ��
    clock_t start;
    //ֻ�� thread 0���� threadIdx.x = 0 ��ʱ�򣩽��м�¼��ÿ�� block �����¼��ʼʱ�估����ʱ��
    if (tid == 0) time[bid] = clock();

    //thread��Ҫͬʱͨ��tid��bid��ȷ����ͬʱ��Ҫ���Ǳ�֤�ڴ�������
    for (i = bid * THREAD_NUM + tid; i < N_size; i += THREAD_NUM*BLOCK_NUM) {

        shared[tid] += num[i] * num[i] * num[i];

    }

    //ͬ�� ��֤ÿ�� thread ���Ѿ��ѽ��д�� shared[tid] ����
    __syncthreads();

    //��״�ӷ�
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

    //����ʱ��,��¼�����ֻ�� thread 0���� threadIdx.x = 0 ��ʱ�򣩽��У�ÿ�� block �����¼��ʼʱ�估����ʱ��
    if (tid == 0)
    { 
        result[bid] = shared[0];
        time[bid + BLOCK_NUM] = clock(); 
    }

}

__host__ //�������� CPU �����еĴ��룬�� __host__ ��ָ��
int main() {


	float h_A[N_size], h_B[N_size],h_C[N_size];
	clock_t time_used[BLOCK_NUM*2];

    for (int i = 0; i < N_size; i++) {
        h_A[i] = 1;
        h_B[i] = i;
    }

    float *d_A, *d_B, *d_C;
	clock_t *time;
    //���Կ��Ϸ����ڴ�
    cudaMalloc((void**)&d_A, sizeof(float) * N_size);
    cudaMalloc((void**)&d_B, sizeof(float) * N_size);
    cudaMalloc((void**)&d_C, sizeof(float) * N_size);
	cudaMalloc((void**)&time,sizeof(clock_t)*BLOCK_NUM*2);
    //���ڴ��е����ݿ������Կ���
    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);

    //�����������������̵߳���֯��ʽ��������
    dim3 DimGrid(BLOCK_NUM, 1, 1);
    dim3 DimBlock(THREAD_NUM,1,1);

    //���Կ���ִ�д���
    //sumOfSquares <<<DimGrid, DimBlock >>>(d_A,d_C);
	sumOfSquares <<<DimGrid,DimBlock,THREAD_NUM*sizeof(float)>>>(d_A,d_C,time);
	
    //�����������Կ���������
    cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);
	cudaMemcpy(time_used,time,sizeof(time),cudaMemcpyDeviceToHost);
    for (int i = 0; i < N_size; i++) {
        cout <<h_C[i]<<"ok"<<'\n';
    }
	cout<<time_used[1]-time_used[0];


    //�ͷ��Դ�
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


	
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
cout <<prop.name<<'\n'<<prop.sharedMemPerBlock<<'\n'<<prop.clockRate<<'\n';
//this device clockRate is 1084500

return 0;
}