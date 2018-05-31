#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void InitMap(
	const float *background,
	const float *target,
	const float *mask,
	float *gradient,
	float *innerBoard,
	float *neighborCount,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	int xoffset[4] = {0,0,1,-1};
	int yoffset[4] = {1,-1,0,0};

	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb){
			// neighbor
			for(int i=0;i<4;i++){

				int yneighbor = yt+yoffset[i];
				int xneighbor = xt+xoffset[i];
				int neighbor = yneighbor*wt + xneighbor;
				int nbx = xt+ox+xoffset[i];
				int nby = yt+oy+yoffset[i];
				int nbb = wb*nby+nbx;

				if(0 <= yneighbor and yneighbor < ht and 0 <= xneighbor and xneighbor < wt){
					neighborCount[curt]++;
					if(mask[neighbor] > 127.0f){
						innerBoard[curt*4 + i] = 1;
					}

					gradient[curt*3+0] += (target[curt*3 + 0] - target[(neighbor)*3 + 0]);
					gradient[curt*3+1] += (target[curt*3 + 1] - target[(neighbor)*3 + 1]);
					gradient[curt*3+2] += (target[curt*3 + 2] - target[(neighbor)*3 + 2]);
				}

			}
		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	float *gradient,
	float *innerBoard,
	float *neighborCount,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	int xoffset[4] = {0,0,1,-1};
	int yoffset[4] = {1,-1,0,0};

	float sum_1 = 0;
	float sum_2 = 0;
	float sum_3 = 0;


	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;

		// fixed part
		sum_1 = sum_1 + gradient[curt*3+0];
		sum_2 = sum_2 + gradient[curt*3+1];
		sum_3 = sum_3 + gradient[curt*3+2];

		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			// neighbor
			for(int i=0;i<4;i++){
				int yneighbor = yt+yoffset[i];
				int xneighbor = xt+xoffset[i];
				int neighbor = yneighbor*wt + xneighbor;

				int nbx = xt+ox+xoffset[i];
				int nby = yt+oy+yoffset[i];
				int nbb = wb*nby+nbx;
				if(0 <= yneighbor and yneighbor < ht and 0 <= xneighbor and xneighbor < wt){
					// fixed part
					
					sum_1 = sum_1  + (1-innerBoard[curt*4 + i]) * background[nbb*3 + 0];
					sum_2 = sum_2  + (1-innerBoard[curt*4 + i]) * background[nbb*3 + 1];
					sum_3 = sum_3  + (1-innerBoard[curt*4 + i]) * background[nbb*3 + 2];

					// current value
					sum_1 = sum_1  + innerBoard[curt*4 + i] * target[(neighbor)*3 + 0];
					sum_2 = sum_2  + innerBoard[curt*4 + i] * target[(neighbor)*3 + 1];
					sum_3 = sum_3  + innerBoard[curt*4 + i] * target[(neighbor)*3 + 2];
				}
				
				else{
					//if(0 <= nby and nby < hb and 0 <= nbx and nbx < wb){
						sum_1 = sum_1  + background[nbb*3 + 0];
						sum_2 = sum_2  + background[nbb*3 + 1];
						sum_3 = sum_3  + background[nbb*3 + 2];
					//}
				}
				
			}
			sum_1 = sum_1 / 4;//neighborCount[curt];
			sum_2 = sum_2 / 4;//neighborCount[curt];
			sum_3 = sum_3 / 4;//neighborCount[curt];

			output[curt*3+0] = sum_1;
			output[curt*3+1] = sum_2;
			output[curt*3+2] = sum_3;
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	

	float *gradient, *innerBoard, *neighborCount, *buf1, *buf2;

	cudaMalloc((void **) &gradient, 3*wt*ht*sizeof(float));
	cudaMalloc((void **) &innerBoard, 4*wt*ht*sizeof(float));
	cudaMalloc((void **) &neighborCount, wt*ht*sizeof(float));
	cudaMalloc((void **) &buf1, 3*wt*ht*sizeof(float));
	cudaMalloc((void **) &buf2, 3*wt*ht*sizeof(float));
	
	cudaMemset((void*)gradient, 0, 3*wt*ht*sizeof(float));
	cudaMemset((void*)innerBoard, 0, 4*wt*ht*sizeof(float));
	cudaMemset((void*)neighborCount, 0, wt*ht*sizeof(float));

	cudaMemcpy(buf1, target, wt*ht*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	InitMap<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, gradient, innerBoard, neighborCount,
		wb, hb, wt, ht, oy, ox
	);
	
	for(int i=0; i< 10000; i++){
		PoissonImageCloningIteration<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
			background, buf1, mask, buf2, gradient, innerBoard, neighborCount,
			wb, hb, wt, ht, oy, ox
		);
		PoissonImageCloningIteration<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
			background, buf2, mask, buf1, gradient, innerBoard, neighborCount,
			wb, hb, wt, ht, oy, ox
		);

	}
	
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	
	cudaFree(gradient);
	cudaFree(innerBoard);
	cudaFree(neighborCount);
	cudaFree(buf1);
	cudaFree(buf2);

}
