# Non_Local_Means_OpenCL
For any queries please contact me at : st173207@stud.uni-stuttgart.de
By: Shubham Gupta (3506475)
Master of Science Information Technology

Use the gpulabproject.zip(For github repo : Non_Local_Means_OpenCL-main.zip ) for linux and extract it in server as gpulabproject folder

In the terminal :
cd gpulabproject
mkdir build
cd build
cmake ..
make 


Note: after making build , make sure that all the input noisy images are in the build folder and output images will also be stored in build folder.
If make is successful then run in the terminal to run the NLM algorithm
./Opencl-project-group8

Dependencies (install yourself in your own linux)
sudo apt install cmake
sudo apt install libboost-all-dev
sudo apt install ocl-icd-opencl-dev
