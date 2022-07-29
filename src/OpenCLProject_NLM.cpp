//////////////////////////////////////////////////////////////////////////////
// OpenCL Project Group 8 (Shubham Gupta): NON LOCAL MEANS
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
//#include <vector>

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

float * computeInsideWeights(int patchSize, float patchSigma)
{
    float * _weights = new float[patchSize * patchSize];
    int centralPixelRow = patchSize / 2;
    int centralPixelCol = centralPixelRow;
    float _dist;
    float _sumW = 0;

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            _dist = (centralPixelRow - i) * (centralPixelRow - i) +
                    (centralPixelCol - j) * (centralPixelCol - j);
            _weights[i * patchSize + j] = exp(-_dist / (2 * (patchSigma * patchSigma)));
            _sumW += _weights[i * patchSize + j];
        }
    }

    for (int i = 0; i < patchSize; i++) {
        for (int j = 0; j < patchSize; j++) {
            _weights[i * patchSize + j] = _weights[i * patchSize + j] / _sumW;
        }
    }

    return _weights;
}

bool isInBounds(int n, int x, int y) 
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

float computeWeight(float dist, float sigma) // compute weight without "/z(i)" division
{
    return expf(-dist / (sigma * sigma));
}

float computePatchDistance( float * image, 
							float * _weights, 
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
                temp = image[(p1_rowStart + i) * n + p1_colStart + j] - image[(p2_rowStart + i) * n + p2_colStart + j];
                ans +=  _weights[i * patchSize + j] * temp * temp;
            }
        }
    }

    return ans;
}
float filterPixel( float * image, 
                   float * _weights, 
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
            res += w * image[i * n + j];
        }
    }
    res = res / sumW;

    return res;
}

std::vector<float> nlmHost( float * image, 
                                int n, 
                                int patchSize,  
                                float patchSigma,
                                float filterSigma )
{
    std::vector<float> res(n * n);
    float * _weights = computeInsideWeights(patchSize, patchSigma);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = filterPixel(image, _weights, n, patchSize, i, j, filterSigma);
        }
    }

    return res;
}

// To read .txt image file
std::vector<float> read(std::string filePath, int n, int m, char delim) 
{
    std::vector<float > image(n * m);
    std::ifstream myfile(filePath);
    std::ifstream input(filePath);
    std::string s;

    for (int i = 0; i < n; i++) {
        std::getline(input, s);
        std::istringstream iss(s);
        std::string num;
        int j = 0;
        while (std::getline(iss, num, delim)) {
            image[i * m + j++] = std::stof(num);
        }
    }

    return image;
}
//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "/zhome/guptasm/gpulabproject/src/OpenCLProject_NLM.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel nlmkernel(program, "nlmKernel");

	// Declare some values
	float patchSize = 5 ;
    float filterSigma = 0.060000 ;
    float patchSigma = 1.2000000 ;
	//std::vector<float> res(n * n);
    float * _weights = computeInsideWeights(patchSize, patchSigma); //To calculate inside weights
	std::size_t wgSizeX = 16; //16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16; //16;
	std::size_t countX = wgSizeX * 4 ; //4; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 4 ;//4;
	//std::size_t countX = 64; // Overall number of work items in X direction = Number of elements in X direction
	//std::size_t countY = 64;
	std::size_t n = countX ; // Size of input image nxn
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size_image = count * sizeof (float); // Size of data in bytes
	std::size_t size_weights = patchSize * patchSize * sizeof(float ); //For the size of weight vector

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_outputGpu (count);

	// Allocate space for input and output data on the device
	cl::Buffer d_input (context, CL_MEM_READ_WRITE, size_image);
	cl::Buffer d_weights (context,CL_MEM_READ_WRITE,size_weights);
	cl::Buffer d_output (context, CL_MEM_READ_WRITE, size_image);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	//memset(h_input.data(), 255, size_image);
	//memset(h_outputCpu.data(), 255, size_image);
	memset(h_outputGpu.data(), 255, size_image);
	memset(_weights, 255, size_weights); //For weight vetor
	//TODO: GPU
	//queue.enqueueWriteBuffer(d_input, true, 0, size_image, h_input.data());
	//queue.enqueueWriteBuffer(d_weights, true, 0, size_weights, _weights);
	//queue.enqueueWriteBuffer(d_output, true, 0, size_image, h_outputGpu.data());

	//////// Load input data ////////////////////////////////
	// Use random input data
	/*
	for (int i = 0; i < count; i++)
		h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	*/
	/*
	// Use an image (Valve.pgm) as input data
	{
		std::vector<float> inputData;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("lenapgm.pgm", inputData, inputWidth, inputHeight);
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
	}
	// Use an image (Valve.pgm) as input data
	*/
	/*
	// Input image (noisy_house.txt)
	std::cout << "Image read txt" << std::endl ;
	h_input = read("/zhome/guptasm/gpulabproject/build/noisy_house.txt", n, n, ',');
    
	*/
	// Do calculation on the host side

	Core::TimeSpan cpuStart = Core::getCurrentTime();
	h_outputCpu= nlmHost(h_input.data(), n, patchSize, patchSigma, filterSigma);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();

	//for (size_t i = 0; i < countX*countX; i++)
	//std::cout<<h_outputCpu[i] ;

	//////// Store CPU output image ///////////////////////////////////
	//std::cout << std::endl<<"check1";
	Core::writeImagePGM("output_nlm_cpu.pgm", h_outputCpu, countX, countY);

	//std::cout << std::endl<<"check2";

    //Do Calculation on device side
	//Reinitialize output memory to 0xff
	memset(h_outputGpu.data(), 255, size_image);
	//TODO: GPU
	queue.enqueueWriteBuffer(d_output, true, 0, size_image, h_outputGpu.data());
	// Copy input data to device
	cl::Event copy1;
	queue.enqueueWriteBuffer(d_input, true, 0, size_image, h_input.data(),NULL, &copy1);
	//check if we should use cl::inputimage
	queue.enqueueWriteBuffer(d_weights, true, 0, size_weights, _weights);
	// Launch kernel on the device
	cl::Event execution;
    nlmkernel.setArg<cl::Buffer>(0, d_input);
	nlmkernel.setArg<cl::Buffer>(1, d_weights);
	nlmkernel.setArg<cl_int>(2, n);
	nlmkernel.setArg<cl_float>(3, patchSize);
	nlmkernel.setArg<cl_float>(4, filterSigma);
	nlmkernel.setArg<cl::Buffer>(5, d_output);
    queue.enqueueNDRangeKernel(nlmkernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);

/*
    //Device Calculation image2d
	// Reinitialize output memory to 0xff
	memset(h_outputGpu.data(), 255, size_image);
	//TODO: GPU
	queue.enqueueWriteBuffer(d_output, true, 0, size_image, h_outputGpu.data());


	// Copy input data to device
	cl::Event copy1;
	cl::Image2D image;
	image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	queue.enqueueWriteImage(image, true, origin, region, countX * sizeof (float), 0, h_input.data(), NULL, &copy1);
	
	cl::Image2D outputimage;
	outputimage = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	//cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	//cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	
	
	// Launch kernel on the device
	cl::Event execution;
    nlmkernel.setArg<cl::Image2D>(0, image);
	nlmkernel.setArg<cl::Buffer>(1, d_weights);
	nlmkernel.setArg<cl_int>(2, n);
	nlmkernel.setArg<cl_float>(3, patchSize);
	nlmkernel.setArg<cl_float>(4, filterSigma);
	nlmkernel.setArg<cl::Image2D>(5, outputimage);
	//nlmkernel.setArg<cl::Buffer>(5, d_output);
    queue.enqueueNDRangeKernel(nlmkernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);
    */
    /*
	// Copy input data to device for color image
	cl::Event copy1;
	cl::Image2D image;
	image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	queue.enqueueWriteImage(image, true, origin, region, countX * sizeof (float), 0, h_input.data(), NULL, &copy1);
	
	cl::Image2D outputimage;
	outputimage = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	//cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	//cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	
	
	// Launch kernel on the device
	cl::Event execution;
    nlmkernel.setArg<cl::Image2D>(0, image);
	nlmkernel.setArg<cl::Image2D>(1, outputimage);
	nlmkernel.setArg<cl_int>(2, n);
	nlmkernel.setArg<cl_int>(3, n);
	nlmkernel.setArg<cl_float>(4, 1.0f/(1.45f*1.45f));
	nlmkernel.setArg<cl_float>(5, 0.2f);
	//nlmkernel.setArg<cl::Buffer>(5, d_output);
    queue.enqueueNDRangeKernel(nlmkernel, cl::NullRange, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);
    */
   
    
	// Copy output data back to host
	cl::Event copy2;
	//queue.enqueueReadImage(outputimage, true, origin, region, countX * sizeof (float), 0, h_outputGpu.data(), NULL, &copy1);
	queue.enqueueReadBuffer(d_output, true, 0, size_image, h_outputGpu.data(), NULL, &copy2);
	std::cout<<std::endl;
	for (size_t i = 0; i < countX*countX; i++)
	std::cout<<h_outputGpu[i] ;
	std::cout<<std::endl;
	for (size_t i = 0; i < patchSize*patchSize; i++)
	std::cout<< _weights[i]<<',';
	std::cout<<std::endl;
	// Print performance data
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(execution);
	Core::TimeSpan copyTime = OpenCL::getElapsedTime(copy1) + OpenCL::getElapsedTime(copy2);
	Core::TimeSpan overallGpuTime = gpuTime + copyTime; 
	std::cout << "CPU Time: " << cpuTime.toString() << ", " << (count / cpuTime.getSeconds() / 1e6) << " MPixel/s" << std::endl;;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", " << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;

	//////// Store GPU output image ///////////////////////////////////
	Core::writeImagePGM("output_nlm_gpu.pgm", h_outputGpu, countX, countY);
    /*
		// Check whether results are correct
		std::size_t errorCount = 0;
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
				size_t index = i + j * countX;
				// Allow small differences between CPU and GPU results (due to different rounding behavior)
				if (!(std::abs (h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
					if (errorCount < 15)
						std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
			return 1;
		}

		std::cout << std::endl;
	}

	std::cout << "Success" << std::endl;
    */
	return 0;
}
