#include "lab1.h"

#include <cuComplex.h>
static const unsigned W = 1024;
static const unsigned H = 1024;
static const unsigned NFRAME = 500;


__global__ void julia_Y(unsigned char *output, int nframe);
__global__ void julia_UV(unsigned char *input, unsigned char *output, int nframe, int color);


static void Launch_julia_Y_kernel(unsigned char *output, size_t grid_dim, size_t block_dim, int nframe){
  dim3 grid = dim3(grid_dim);
  dim3 block= dim3(block_dim);
  //printf("julia kernel <<<%d, %d>>> launching...\n", grid_dim, block_dim);

  julia_Y<<<grid, block>>>(output, nframe);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
}
static void Launch_julia_UV_kernel(unsigned char *input, unsigned char *output, size_t grid_dim, size_t block_dim, int nframe, int color){
  dim3 grid = dim3(grid_dim);
  dim3 block= dim3(block_dim);
  //printf("julia kernel <<<%d, %d>>> launching...\n", grid_dim, block_dim);

  julia_UV<<<grid, block>>>(input, output, nframe, color);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
}

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

static unsigned char * Init_channel(size_t channel_size){
	unsigned char *channel;
	cudaError_t err;
	err = cudaMalloc((void **)&channel, channel_size);
	if (err != cudaSuccess){
		printf("Init_channel Error: %s\n", cudaGetErrorString(err));
	}

	err = cudaMemset(channel, 0.0, channel_size);
	if (err != cudaSuccess){
		printf("Init_channel Error: %s\n", cudaGetErrorString(err));
	}

	return channel;
}

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	
	unsigned char *channelY = Init_channel(W*H);
	unsigned char *channelU = Init_channel(W*H/4);
	unsigned char *channelV = Init_channel(W*H/4);

	Launch_julia_Y_kernel(channelY, H, W, (impl->t));
	Launch_julia_UV_kernel(channelY, channelU, H/2, W/2, (impl->t), 250);
	Launch_julia_UV_kernel(channelY, channelV, H/2, W/2, (impl->t), 0);

	// cpy to screen
	cudaMemcpy(yuv, channelY, W*H, cudaMemcpyDeviceToDevice);
	cudaMemcpy(yuv+W*H, channelU, W*H/4, cudaMemcpyDeviceToDevice);
	cudaMemcpy(yuv+W*H+W*H/4, channelV, W*H/4, cudaMemcpyDeviceToDevice);
	++(impl->t);
}





__global__ void julia_Y(unsigned char *output, int nframe) 
{
	// constant
	float alpha = (1/(1+exp(-1*((float)nframe/100 + 0.1*cos((double)nframe*0.5)))));
	float zoom = 1.5*alpha + 3.5*(1-alpha);
	float xmin = -zoom;
	float xmax = zoom;
	float xwidth = xmax - xmin;
	float ymin = -zoom;
	float ymax = zoom;
	float yheight = ymax - ymin;


	double cr = -0.1*alpha   + 0.3*(1-alpha);
	double ci = 0.65*alpha   + 0.4*(1-alpha);
	cuDoubleComplex c = make_cuDoubleComplex(cr, ci);

	int pixelIdx = blockIdx.x*blockDim.x + threadIdx.x;
	float pixelX = threadIdx.x;
	float pixelY = blockIdx.x;
	int nit = 0;

	//creat pixel complex
	double zr = ((float)(pixelX / blockDim.x) * xwidth) + xmin;
	double zi = ((float)(pixelY / gridDim.x) * yheight) + ymin;
	double zr_rotate = zr * cos((double)nframe*0.01) + zi * sin((double)nframe*0.01);
	double zi_rotate = -zr * sin((double)nframe*0.01) + zi * cos((double)nframe*0.01);

	cuDoubleComplex z = make_cuDoubleComplex(zr_rotate, zi_rotate);


	while(cuCabs(z) <= 100000 && nit < nframe){
		z = cuCadd(cuCmul(z, z), c);
		nit++;
	}
	
	//printf("%f, %f: %d %d %d\n", zr,zi,nit, blockDim.x, gridDim.x);
	output[pixelIdx] = (nit/(nframe))*255;

}

__global__ void julia_UV(unsigned char *input, unsigned char *output, int nframe, int color) 
{
	int pixelIdx = blockIdx.x*blockDim.x + threadIdx.x;
	int pixelX = threadIdx.x;
	int pixelY = blockIdx.x;

	if (input[pixelY*4*blockDim.x + pixelX*2] == 255){
		output[pixelIdx] = color;
	}
	else{
		output[pixelIdx] = 128;
	}


}