#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <time.h>
#include <onnxruntime_cxx_api.h>
#include "Common/Drivers.h"
#include "Tensor/ValueInfo.h"
#include "Tensor/TensorValue.hpp"
#include "Tensor/ModelTensorsInfo.h"
#include "Common/PathManager.h"
using namespace std;

// 41.6 28.3ms 24.3ms 18.5ms
int main(int argc, char *argv[])
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // log id: "test"
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);

    std::filesystem::path model_path;
    model_path = OnnxPathManager::GetChildModelSavePath("resnet50");
    Ort::Session session(env, model_path.c_str(), session_options);

    std::filesystem::path model_path_0;
    model_path_0 = OnnxPathManager::GetChildModelSavePath("resnet50", 0);
    Ort::Session session_0(env, model_path_0.c_str(), session_options);

    std::filesystem::path model_path_1;
    model_path_1 = OnnxPathManager::GetChildModelSavePath("resnet50", 1);
    Ort::Session session_1(env, model_path_1.c_str(), session_options);

    ModelInfo modelInfo_0(session_0);
    ModelInfo modelInfo_1(session_1);

    cout << "input:" << endl;
    vector<TensorValue<float>> input_tensors;
    vector<const char *> input_labels;
    for (const ValueInfo &info : modelInfo_0.GetInput().GetAllTensors())
    {
        input_tensors.push_back(TensorValue(info, true));
        input_labels.push_back(info.GetName().c_str());
    }

    vector<Ort::Value> input_values;
    for (auto &tensor : input_tensors)
    {
        input_values.push_back(tensor);
    }

    vector<TensorValue<float>> middle_tensors;
    vector<const char *> middle_labels;
    for (const ValueInfo &info : modelInfo_0.GetOutput().GetAllTensors())
    {
        middle_tensors.push_back(TensorValue(info, false));
        middle_labels.push_back(info.GetName().c_str());
    }

    vector<TensorValue<float>> output_tensors;
    vector<const char *> output_labels;
    for (const ValueInfo &info : modelInfo_1.GetOutput().GetAllTensors())
    {
        output_tensors.push_back(TensorValue(info, false));
        output_labels.push_back(info.GetName().c_str());
    }

    cout << "total:" << endl;
    vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
    for (int i = 0; i < output_values.size(); i++)
    {
        output_tensors[i].RecordOrtValue(output_values[i]);
    }

    for (auto &tensor : output_tensors)
    {
        tensor.Print();
    }

    cout << "childs:" << endl;
    vector<Ort::Value> middle_values;
    // test in short-scope
    {
        vector<Ort::Value> middle_values_ = session_0.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), middle_labels.data(), middle_labels.size());
        clock_t start = clock();
        middle_values = std::move(middle_values_);
        cout << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)." << endl;
    }

    // clock_t start_1 = clock();
    // middle_values=std::move(middle_values_);
    // cout<<(clock() - start_1) * 1000.0 / CLOCKS_PER_SEC << "ms)." << endl;

    vector<Ort::Value> output_values_child = session_1.Run(Ort::RunOptions{nullptr}, middle_labels.data(), middle_values.data(), middle_labels.size(), output_labels.data(), output_labels.size());

    clock_t start = clock();
    for (int i = 0; i < output_values_child.size(); i++)
    {
        output_tensors[i].RecordOrtValue(output_values_child[i]);
    }
    cout << (clock() - start) * 1.0 / CLOCKS_PER_SEC * 1000.0 << "ms)." << endl;

    for (auto &tensor : output_tensors)
    {
        tensor.Print();
    }

    return 0;
}
