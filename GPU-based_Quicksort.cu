// Improved GPU-based implementation of quicksort

#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>
#include <stdio.h>
#include <stdlib.h>


#define MaxDepth       16
#define Num_Elements  32


// when we exceed the capacity : depth gets more than the defined value or the number of elements less than wrap size
////////////////------------------selection sort algorithm which is serialized
__device__ void selection_sort(unsigned int* array, int left, int right)
{
    for (int i = left; i <= right; ++i)
    {
        unsigned val_i = array[i];
        int idx_i = i;
        // searching for the smaller value in the array then shift to keft side
        for (int j = i + 1; j <= right; ++j)
        {
            unsigned val_j = array[j];

            if (val_j < val_i)
            {
                idx_i = j;
                val_i = val_j;
            }
        }
        // changing the values
        if (i != idx_i)
        {
            array[idx_i] = array[i];
            array[i] = val_i;
        }
    }
}

////////////////------------------quick_sort algorithm recursive -------------------------------------


__global__ void quick_sort(unsigned int* array, int left, int right, int depth)
{
    //// checking for constraints
    if (depth >= MaxDepth || right - left <= Num_Elements)
    {
        selection_sort(array, left, right);
        return;
    }

    unsigned int* left_ptr = array + left;
    unsigned int* right_ptr = array + right;
    unsigned int  pivot = array[(left + right) / 2];

    // partitioning.
    while (left_ptr <= right_ptr)
    {
        unsigned int left_val = *left_ptr;
        unsigned int right_val = *right_ptr;

        // passing the left pointer till the value of pinter is smaller than the pivot.
        while (left_val < pivot)
        {
            left_ptr++;
            left_val = *left_ptr;
        }
        // passing the left pointer till the value of pinter is larger than the pivot.
        while (right_val > pivot)
        {
            right_ptr--;
            right_val = *right_ptr;
        }
        // otherwise we exchange the value
        if (left_ptr <= right_ptr)
        {
            *left_ptr++ = right_val;
            *right_ptr-- = left_val;
        }
    }

    // division of array for subarray
    int nright = right_ptr - array;
    int nleft = left_ptr - array;

    // recursive part : left subarray
    if (left < (right_ptr - array))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        quick_sort << < 1, 1, 0, s >> > (array, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }

    //  recursive part : right subarray
    if ((left_ptr - array) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        quick_sort << < 1, 1, 0, s1 >> > (array, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Calling and Launching the quick_sort kernel 

void Initializion_sort(unsigned int* array, unsigned int nums)
{
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MaxDepth));

    int left = 0;
    int right = nums - 1;
    std::cout << "Launching kernel " << std::endl;
    //Time kernel launch
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    float elapsedTime;
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    quick_sort << < 1, 1 >> > (array, left, right, 0);
    checkCudaErrors(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
    CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN(cudaGetLastError()); //Check if an error occurred in device code
    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(stop));
    std::cout << "done.\nElapsed Quick kernel time: " << elapsedTime << " ms\n" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// populating array randomly

void initialize_array(unsigned int* dataset, unsigned int nums)
{
    srand(2047);
    for (unsigned i = 0; i < nums; i++)
        dataset[i] = rand() % nums;
}

////////////////////////////////////////////////////////////////////////////////
// Validating  the results.

void check_results(int n, unsigned int* results_d)
{
    unsigned int* results_h = new unsigned[n];
    checkCudaErrors(cudaMemcpy(results_h, results_d, n * sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1; i < n; ++i)
        if (results_h[i - 1] > results_h[i])
        {
            std::cout << "Invalid item[" << i - 1 << "]: " << results_h[i - 1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "valid result from GPU" << std::endl;
    delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int array_lenght = 128;
    bool verbose = false;

    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "h"))
    {
        std::cerr << "Usage: " << argv[0] << " array_lenght=<array_lenght>\twhere array_lenght is the number of items to sort" << std::endl;
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "v"))
    {
        verbose = true;
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "array_lenght"))
    {
        array_lenght = getCmdLineArgumentInt(argc, (const char**)argv, "array_lenght");

        if (array_lenght < 1)
        {
            std::cerr << "ERROR: array_lenght has to be greater than 1" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // device properties  to check if it can support CUDA Dynamic Parallelism
    int device = -1;
    cudaDeviceProp deviceProp;
    device = findCudaDevice(argc, (const char**)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

    if (!(deviceProp.major > 3 || (deviceProp.major == 3 && deviceProp.minor >= 5)))
    {
        printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.", device, deviceProp.name);
        exit(EXIT_WAIVED);
    }

    // defining array
    unsigned int* h_array = 0;
    unsigned int* d_array = 0;

    // Allocating  memory and populating the array.
    std::cout << "Initializing array:" << std::endl;
    h_array = (unsigned int*)malloc(array_lenght * sizeof(unsigned int));
    initialize_array(h_array, array_lenght);

    if (verbose)
    {
        for (int i = 0; i < array_lenght; i++)
            std::cout << "array [" << i << "]: " << h_array[i] << std::endl;
    }

    // Allocating array on GPU memory.
    checkCudaErrors(cudaMalloc((void**)&d_array, array_lenght * sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_array, h_array, array_lenght * sizeof(unsigned int), cudaMemcpyHostToDevice));


    std::cout << "quick_sort with " << array_lenght << " elements" << std::endl;
    Initializion_sort(d_array, array_lenght);

    // Check Validating
    std::cout << "Validating results: ";
    check_results(array_lenght, d_array);

    free(h_array);
    checkCudaErrors(cudaFree(d_array));

    exit(EXIT_SUCCESS);
}

