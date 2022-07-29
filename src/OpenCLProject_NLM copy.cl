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
    //return (-dist / (sigma * sigma));
}
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
// Read value from global array a, return 0 if outside image
float getValueImage(__read_only image2d_t a, int i, int j) {
	//if (i < 0 || i >= countX || j < 0 || j >= countY)
	//return 0;
	return read_imagef(a, sampler, (int2){i, j}).x;
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
    float ans = 0;
    float temp;
    printf("\nkernelcheck4");
    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            printf("kernelcheck5");
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)) {
                printf("\nkernelcheck6");
                temp = image[(p1_rowStart + i) * n + p1_colStart + j] - image[(p2_rowStart + i) * n + p2_colStart + j];
                printf("\nkernelcheck7");
                ans +=  _weights[i * patchSize + j] * temp * temp;
                printf("\nkernelcheck8");
            }
        }
    }
    printf("\nkernelcheck9");
    return ans;
}

float computePatchDistanceimg( __read_only image2d_t image, 
							__global float * _weights, 
							int n, 
                            int patchSize, 
                            int p1_rowStart, 
                            int p1_colStart, 
                            int p2_rowStart, 
                            int p2_colStart ) 
{
    float ans = 0;
    float temp;
     
    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            if (isInBounds(n, p1_rowStart + i, p1_colStart + j) && isInBounds(n, p2_rowStart + i, p2_colStart + j)) {
                
                temp= getValueImage(image, (p1_rowStart + i),(p1_colStart + j)) - getValueImage(image, (p2_rowStart + i), (p2_colStart + j));
                //temp = image[(p1_rowStart + i) * n + p1_colStart + j] - image[(p2_rowStart + i) * n + p2_colStart + j];
                ans +=  _weights[i * patchSize + j] * temp * temp;
            }
        }
    }

    return ans;
}


/*
__kernel void nlmKernel(__read_only image2d_t image, 
                        __global float * _weights, 
                        int n, 
                        int patchSize, 
                        float sigma,
                        __write_only image2d_t filteredImage)
{


    int pixelRow = get_global_id(0);
    int pixelCol = get_global_id(1);

 	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);   

    if (pixelRow>n && pixelCol>n){
    float res = 0;
    float sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    float dist;
    float w;
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist = computePatchDistanceimg(image,  
                                        _weights, 
                                        n, 
                                        patchSize, 
                                        patchRowStart, 
                                        patchColStart, 
                                        i - patchSize / 2, 
                                        j - patchSize / 2  );
            w = computeWeight(dist, sigma);
            sumW += w;
            res += w * getValueImage(image, j, i);
            //res += w * image[j * n + i];
        }
    }
    res = res / sumW ;
    write_imagef(filteredImage, (int2){pixelRow,pixelCol}, res);
    //filteredImage[getIndexGlobal(countX,pixelRow,pixelCol)] = 0.56;
    }
    
}
*/

/*
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


    //if (getIndexGlobal(countX,pixelRow,pixelCol) >= n * n) 
    //    return;

    //if (pixelRow>n && pixelCol>n){
    if (pixelRow>n*n){
    float res = 0;
    float sumW = 0;                    // sumW is the Z(i) of w(i, j) formula
    float dist;
    float w;
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist = computePatchDistance(image,  
                                        _weights, 
                                        n, 
                                        patchSize, 
                                        patchRowStart, 
                                        patchColStart, 
                                        i - patchSize / 2, 
                                        j - patchSize / 2  );
            w = computeWeight(dist, sigma);
            sumW += w;
            res += w * image[j * n + i];
        }
    }
    res = res / sumW;

    filteredImage[getIndexGlobal(countX,pixelCol,pixelRow)] = res;
    }
}
*/
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
    int patchRowStart = pixelRow - patchSize / 2;
    int patchColStart = pixelCol - patchSize / 2;
    printf("\nkernelcheck2");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("\nkernelcheck3");
            dist = computePatchDistance(image,  
                                        _weights, 
                                        n, 
                                        patchSize, 
                                        patchRowStart, 
                                        patchColStart, 
                                        i - patchSize / 2, 
                                        j - patchSize / 2  );
            printf("\nkernelcheck10");                            
            w = computeWeight(dist, sigma);
            sumW += w;
            res += w * image[i * n + j];
            printf("\nkernelcheck11");
        }
    }
    printf("\nkernelcheck12");
    res = res / sumW;

    return res;
}


//To check
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
    
    if (pixelRow<n && pixelCol<n){
    barrier(CLK_LOCAL_MEM_FENCE);
    //for (int i = 0; i < n; i++) {
        //for (int j = 0; j < n; j++) {
            printf("kernelcheck1");
            filteredImage[getIndexGlobal(countX,pixelCol,pixelRow)] = filterPixel(image, _weights, n, patchSize, pixelCol,pixelRow, sigma);
            printf("\nkernelcheck13");
       // }
   // }
    }

}



/*
//To check other method
__kernel void nlmKernel(__global float * image, 
                        __global float * _weights, 
                        int n, 
                        int patchSize, 
                        float sigma,
                        __global float *filteredImage)
{   float h_nlm = 10; //added
    float sumWeights= 0.0f; 
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int WIDTH = HEIGHT = n; //added
    int NLM_WINDOW=n/2; /added
    int NLM_BLOCK_RADIUS = patchSize;//added
    if(x<WIDTH && y < HEIGHT){
	    const int samples=(2*NLM_BLOCK_RADIUS+1)*(2*NLM_BLOCK_RADIUS+1);
		float weightIJ =0.0f;
		float sumWeights =0.0f;
		float clr =0.0f;
		float sigma11 =0.0f;
        
		#pragma unroll
		for(int j = -NLM_WINDOW; j <= NLM_WINDOW; j++){
			for(int i = -NLM_WINDOW; i <= NLM_WINDOW; i++){
			// Compute the Euclidean distance  beetween the two patches
			float weightIJ=0.0f;
			#pragma unroll
			for(int m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++){
				for(int n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++){
					int x1= (x+j + m)>=0?x+ j + m:0;
					int y1= (y+i + n)>=0? y+i+n:0;
					x1= x1<WIDTH? x1:WIDTH-1;
					y1= y1<HEIGHT? y1:HEIGHT-1;
					int x2=(x+ m)>=0? x+ m:0;
				    int y2=(y+ n)>=0 ? y+ n:0;
				    x2= x2<WIDTH? x2:WIDTH-1;
				    y2=y2<HEIGHT? y2:HEIGHT-1;
					int a=input[y1 *( WIDTH) +x1];
					int b=input[y2 *( WIDTH) +x2];
					weightIJ += vec_len_int(a,b);
				}
			}
			float factor=max((h_nlm*h_nlm),0.000001f);
			sigma11=1/factor;
			int x3= (x+j)>=0? j+x :0;
			int y3= (y+i)>=0? i +y:0;
			x3= x3<WIDTH? x3:WIDTH-1;
			y3= y3<HEIGHT? y3:HEIGHT-1;
			int clrIJ=input[mad24(y3,WIDTH ,x3)];
			weightIJ     = exp( -(	(	(weightIJ * sigma11 )    /samples	)));
			clr += clrIJ * weightIJ;
			sumWeights  += weightIJ;
			}
		}
	 sumWeights = 1.0f / sumWeights;
	 clr*= sumWeights;
	 output[mad24(y,WIDTH,x)]=(uchar)clr;
  	}
*/


/*
// TEST COLOR NLM CODE

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define NLM_WINDOW_RADIUS   3
#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )
#define NLM_WEIGHT_THRESHOLD    0.10f
#define NLM_LERP_THRESHOLD      0.10f
#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )

inline float vecLen(float3 a, float3 b)
{
	return (
		(b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z)
		);
}

inline float lerpf(float a, float b, float c)
{
	return a + (b - a) * c;
}

//__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void nlmKernel(read_only image2d_t src , 
	                     write_only image2d_t dst,
	                     __global int* width,
	                     __global int* height,
	                     __global float* lerpC,
	                     __global float* noise)
{
	__local float fweights[BLOCKDIM_X * BLOCKDIM_Y];

//	const int ix = get_local_size(0) * get_group_id(0) + get_local_id(0);
//	const int iy = get_local_size(1) * get_group_id(1) + get_local_id(1);
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	const float x = (float)ix + 0.5f;
	const float y = (float)iy + 0.5f;

	const int cx = get_local_size(0) * get_global_id(0) + NLM_WINDOW_RADIUS ;
	const int cy = get_local_size(1) * get_global_id(1) + NLM_WINDOW_RADIUS ;

	if(ix < *width && iy < *height){
		float weight = 0;

		for(float n = -NLM_WINDOW_RADIUS ; n <= NLM_WINDOW_RADIUS ; n++){
			for(float m = -NLM_WINDOW_RADIUS ; m <= -NLM_WINDOW_RADIUS ; m++){
				weight += vecLen(
					   (float3)read_imagef(src , sampler , (int2)(cx + m , cy + n)).xyz,
                       (float3)read_imagef(src , sampler , (int2)(ix + m , iy + n)).xyz

					);
			}
		}
	    float dist = (get_local_id(0) - NLM_WINDOW_RADIUS) * (get_local_id(0) - NLM_WINDOW_RADIUS) +
	                 (get_local_id(1) - NLM_WINDOW_RADIUS) * (get_local_id(1) - NLM_WINDOW_RADIUS);

	    weight = native_exp(-(weight * (*noise) + dist * INV_NLM_WINDOW_AREA));

	    fweights[get_local_id(1) * BLOCKDIM_X + get_local_id(0)] = weight;

	    barrier(CLK_LOCAL_MEM_FENCE);

	    float fCount = 0;

	    float sumWeights = 0;

	    float3 clr = {0,0,0};

	    int idx = 0;

		for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++){
			for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++){
	    		float weightij = fweights[idx++];

	    		float3 clrij = read_imagef(src, sampler, (int2)(ix + j , iy + i)).xyz;
	    		clr.x       += clrij.x * weightij;
	    		clr.y       += clrij.y * weightij;
	    		clr.z       += clrij.z * weightij;

				sumWeights += weightij;

				fCount += (weightij > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
	    	}
	    }

		sumWeights = 1.0f / sumWeights;
		clr.x *= sumWeights;
		clr.y *= sumWeights;
		clr.z *= sumWeights;

		float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? (*lerpC) : 1.0f - *lerpC;

        float3 clr00 = read_imagef(src, sampler, (int2)(ix , iy)).xyz;
        clr.x = lerpf(clr.x, clr00.x, lerpQ);
        clr.y = lerpf(clr.y, clr00.y, lerpQ);
        clr.z = lerpf(clr.z, clr00.z, lerpQ);
        write_imagef(dst, (int2)(ix, iy), (float4)(clr.x , clr.y, clr.z, 1.0f));
	}
}
*/