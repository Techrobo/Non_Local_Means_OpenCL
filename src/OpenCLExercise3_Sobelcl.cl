#ifndef __OPENCL_VERSION__
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
__kernel void sobelKernel1(__global float* d_input, __global float* d_output) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float Gx = getValueGlobal(d_input, countX, countY, i-1, j-1)+2*getValueGlobal(d_input, countX, countY, i-1, j)+getValueGlobal(d_input, countX, countY, i-1, j+1)
			-getValueGlobal(d_input, countX, countY, i+1, j-1)-2*getValueGlobal(d_input, countX, countY, i+1, j)-getValueGlobal(d_input, countX, countY, i+1, j+1);
	float Gy = getValueGlobal(d_input, countX, countY, i-1, j-1)+2*getValueGlobal(d_input, countX, countY, i, j-1)+getValueGlobal(d_input, countX, countY, i+1, j-1)
			-getValueGlobal(d_input, countX, countY, i-1, j+1)-2*getValueGlobal(d_input, countX, countY, i, j+1)-getValueGlobal(d_input, countX, countY, i+1, j+1);
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

__kernel void sobelKernel2(__global float* d_input, __global float* d_output) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float Gmm = getValueGlobal(d_input, countX, countY, i-1, j-1);
	float Gm0 = getValueGlobal(d_input, countX, countY, i-1, j);
	float Gmp = getValueGlobal(d_input, countX, countY, i-1, j+1);
	float Gpm = getValueGlobal(d_input, countX, countY, i+1, j-1);
	float Gp0 = getValueGlobal(d_input, countX, countY, i+1, j);
	float Gpp = getValueGlobal(d_input, countX, countY, i+1, j+1);
	float G0m = getValueGlobal(d_input, countX, countY, i, j-1);
	float G0p = getValueGlobal(d_input, countX, countY, i, j+1);

	float Gx = Gmm+2*Gm0+Gmp
			-Gpm-2*Gp0-Gpp;
	float Gy = Gmm+2*G0m+Gpm
			-Gmp-2*G0p-Gpp;
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

// Read value from global array a, return 0 if outside image
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
float getValueImage(__read_only image2d_t a, int i, int j) {
	//if (i < 0 || i >= countX || j < 0 || j >= countY)
	//return 0;
	return read_imagef(a, sampler, (int2){i, j}).x;
}
__kernel void sobelKernel3(__read_only image2d_t d_input, __global float* d_output) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float Gx = getValueImage(d_input, i-1, j-1)+2*getValueImage(d_input, i-1, j)+getValueImage(d_input, i-1, j+1)
			-getValueImage(d_input, i+1, j-1)-2*getValueImage(d_input, i+1, j)-getValueImage(d_input, i+1, j+1);
	float Gy = getValueImage(d_input, i-1, j-1)+2*getValueImage(d_input, i, j-1)+getValueImage(d_input, i+1, j-1)
			-getValueImage(d_input, i-1, j+1)-2*getValueImage(d_input, i, j+1)-getValueImage(d_input, i+1, j+1);
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}
__kernel void sobelKernel4(__read_only image2d_t d_input, __global float* d_output) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float Gmm = getValueImage(d_input, i-1, j-1);
	float Gm0 = getValueImage(d_input, i-1, j);
	float Gmp = getValueImage(d_input, i-1, j+1);
	float Gpm = getValueImage(d_input, i+1, j-1);
	float Gp0 = getValueImage(d_input, i+1, j);
	float Gpp = getValueImage(d_input, i+1, j+1);
	float G0m = getValueImage(d_input, i, j-1);
	float G0p = getValueImage(d_input, i, j+1);

	float Gx = Gmm+2*Gm0+Gmp
			-Gpm-2*Gp0-Gpp;
	float Gy = Gmm+2*G0m+Gpm
			-Gmp-2*G0p-Gpp;
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}

// Return the value at "current position + (x, y)"
float getValue(__global const float* a, __local const float* buffer, int x, int y) {
	int i = (int) get_global_id(0) + x;
	int j = (int) get_global_id(1) + y;

	int il = (int) get_local_id(0) + x;
	int jl = (int) get_local_id(1) + y;

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	size_t countXL = get_local_size(0);
	size_t countYL = get_local_size(1);

	if (1 || il < 0 || il >= countXL || jl < 0 || jl >= countYL)
		return getValueGlobal(a, countX, countY, i, j);
	else
		return buffer[il + jl * countXL];
}
__kernel void sobelKernel5(__global float* d_input, __global float* d_output, __local float* buffer) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	int il = get_local_id(0);
	int jl = get_local_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	size_t countXL = get_local_size(0);
	size_t countYL = get_local_size(1);

	buffer[il + countXL * jl] = d_input[i + countX * j];

	barrier(CLK_LOCAL_MEM_FENCE);

	float Gx = getValue(d_input, buffer, -1, -1)+2*getValue(d_input, buffer, -1, 0)+getValue(d_input, buffer, -1, +1)
			-getValue(d_input, buffer, +1, -1)-2*getValue(d_input, buffer, +1, 0)-getValue(d_input, buffer, +1, +1);
	float Gy = getValue(d_input, buffer, -1, -1)+2*getValue(d_input, buffer, 0, -1)+getValue(d_input, buffer, +1, -1)
			-getValue(d_input, buffer, -1, +1)-2*getValue(d_input, buffer, 0, +1)-getValue(d_input, buffer, +1, +1);
	d_output[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
}
