#include "Common/PathManager.h"

#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <string>

// OnnxPathManager
std::filesystem::path OnnxPathManager::onnxRootFold = "Onnxs";

std::filesystem::path OnnxPathManager::GetOnnxRootFold()
{
    return RootPathManager::GetRunRootFold() / OnnxPathManager::onnxRootFold;
}

std::filesystem::path OnnxPathManager::GetModelSavePath(std::string model_name)
{
    std::filesystem::create_directories(OnnxPathManager::GetOnnxRootFold() / model_name);
    return OnnxPathManager::GetOnnxRootFold() / model_name / (model_name + ".onnx");
}

std::filesystem::path OnnxPathManager::GetModelParamsSavePath(std::string model_name)
{
    std::filesystem::create_directories(OnnxPathManager::GetOnnxRootFold() / model_name);
    return OnnxPathManager::GetOnnxRootFold() / model_name / (model_name + ".json");
}

std::filesystem::path OnnxPathManager::GetChildModelSavePath(std::string model_name, int idx)
{
    if (idx < 0)
    {
        return OnnxPathManager::GetModelSavePath(model_name);
    }
    std::filesystem::create_directories(OnnxPathManager::GetOnnxRootFold() / model_name / "childs" / std::to_string(idx));
    return OnnxPathManager::GetOnnxRootFold() / model_name / "childs" / std::to_string(idx) / (model_name + "-" + std::to_string(idx) + ".onnx");
}

std::filesystem::path OnnxPathManager::GetChildModelParamsSavePath(std::string model_name, int idx)
{
    if (idx < 0)
    {
        return OnnxPathManager::GetModelParamsSavePath(model_name);
    }
    std::filesystem::create_directories(OnnxPathManager::GetOnnxRootFold() / model_name / "childs" / std::to_string(idx));
    return OnnxPathManager::GetOnnxRootFold() / model_name / "childs" / std::to_string(idx) / (model_name + "-" + std::to_string(idx) + "-params.json");
}

std::filesystem::path OnnxPathManager::GetChildModelSumParamsSavePath(std::string model_name)
{
    std::filesystem::create_directories(OnnxPathManager::GetOnnxRootFold() / model_name / "childs");
    return OnnxPathManager::GetOnnxRootFold() / model_name / "childs" / (model_name + "-params.json");
}

std::filesystem::path OnnxPathManager::GetChildModelSumCacheSavePath(std::string model_name)
{
    std::filesystem::create_directories(OnnxPathManager::GetOnnxRootFold() / model_name);
    return OnnxPathManager::GetOnnxRootFold() / model_name / "childs/cache.json";
}

// BenchmarkPathManager
std::filesystem::path BenchmarkPathManager::benchmarkRootFold = "Benchmark";

// RootPathManager
std::filesystem::path RootPathManager::projectRootFold = []() -> std::filesystem::path
{
    std::filesystem::path p;

    char execPath[1024];
    memset(execPath, 0, sizeof(execPath));
    if (readlink("/proc/self/exe", execPath, sizeof(execPath) - 1) >= 1023) // read exec path
    {
        std::cout << "warning: process exec path is too long(>=1023), we try to use work path instead, may meet some error." << std::endl;
        char *workPath = getcwd(NULL, 0);
        if (workPath == NULL)
        {
            p = ".";
        }
        else
        {
            p = workPath;
        }
    }
    else
    {
        p = execPath;
        p = p.parent_path();
    }

    if (p.parent_path().filename() == "bin" && (p.filename() == "release" || p.filename() == "debug"))
    {
        p = p.parent_path().parent_path();
    }
    return p.make_preferred();
}();

std::filesystem::path RootPathManager::GetRunRootFold()
{
    return RootPathManager::projectRootFold;
}