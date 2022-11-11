#include "Common/Drivers.h"

// std::string Drivers::CPUDriver = "CPUExecutionProvider";
// std::string Drivers::GPUDriver = "CUDAExecutionProvider";
namespace Drivers
{
    OrtCUDAProviderOptions GPU_CUDA::GPU0 = []()
    {
        OrtCUDAProviderOptions opt;
        opt.device_id = 0;
        return opt;
    }();

    OrtCUDAProviderOptions GPU_CUDA::GPU1 = []()
    {
        OrtCUDAProviderOptions opt;
        opt.device_id = 1;
        return opt;
    }();

}