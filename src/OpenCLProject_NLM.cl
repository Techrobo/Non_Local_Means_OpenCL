#ifndef __OPENCL_VERSION__
#include<math>
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

int getIndexGlobal(size_t countX, int i, int j) {
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(__global const float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || i >= countX || j < 0 || j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

bool isInBounds(int n, int x, int y) 
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

float computeWeight(float dist, float sigma) // compute weight without "/z(i)" division
{
    return exp(-dist / (sigma * sigma));
}


float computePatchDistance( __global float * image, 
							__global float * _weights, 
							int n, 
                            int patchSize, 
                            int p1_rowStart, 
                            int p1_colStart, 
                            int p2_rowStart, 
                            int p2_colStart ) 
{
    
    
    long ans = 0;
    long temp;
 
    //printf("\nkernelcheck4");
    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            //printf("kernelcheck5");
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)) {
                //printf("\nkernelcheck6");
                temp = image[(p1_rowStart + i) * n + p1_colStart + j] - image[(p2_rowStart + i) * n + p2_colStart + j];
               
                //printf("\nkernelcheck7");
                //printf("%f",_weights[i * patchSize + j]);
                ans +=  _weights[i * patchSize + j] * temp * temp;
                //ans +=  1* temp * temp;
               
                //printf("\nkernelcheck8");
            }
        }
    }
    //printf("\nkernelcheck9");

    return ans;

}

float filterPixel( __global float * image, 
                   __global float * _weights, 
                   int n, 
                   int patchSize, 
                   int pixelRow, 
                   int pixelCol, 
                   float sigma )
{
    float res = 0;
    float sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    float dist;
    float w;
    int patchRowStart = pixelRow - (patchSize / 2);
    int patchColStart = pixelCol - (patchSize / 2);

    //printf("kernelcheck2");
    
    barrier(CLK_LOCAL_MEM_FENCE);


    //printf("kernelcheck3");
    dist = computePatchDistance(image,  
                                _weights, 
                                n, 
                                patchSize, 
                                patchRowStart, 
                                patchColStart, 
                                pixelRow, 
                                pixelCol);
    //printf("kernelcheck10");                            
    w = computeWeight(dist, sigma);
    sumW += w;
    res += w * image[patchRowStart * n + patchColStart];
    //printf("kernelcheck11");
    //printf("kernelcheck12");
    res = res / sumW;
    return res;
}

__kernel void nlmKernel(__global float * image, 
                        __global float * _weights, 
                        int n, 
                        int patchSize, 
                        float sigma,
                        __global float *filteredImage)
{   
    int pixelRow = get_global_id(0);
    int pixelCol = get_global_id(1);
    
 	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
 
    //printf("kernelcheck1"); 
    filteredImage[getIndexGlobal(countX,pixelRow,pixelCol)] = filterPixel(image, _weights, n, patchSize, pixelCol,pixelRow, sigma);
    
}


