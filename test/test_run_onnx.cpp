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
    std::filesystem::path model_path;
    if (argc >= 3)
    {
        model_path = OnnxPathManager::GetChildModelSavePath(argv[1], atoi(argv[2]));
    }
    else
    {
        model_path = OnnxPathManager::GetChildModelSavePath("resnet50");
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // log id: "test"

    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);
    Ort::Session session(env, model_path.c_str(), session_options);

    ModelInfo modelInfo(session);
    // cout << modelInfo << endl;
    cout << modelInfo << endl;

    cout << "input:" << endl;
    vector<TensorValue<float>> input_tensors;
    vector<const char *> input_labels;

    for (const ValueInfo &info : modelInfo.GetInput().GetAllTensors())
    {
        input_tensors.push_back(TensorValue(info, true));
        input_labels.push_back(info.GetName().c_str());
    }

    vector<Ort::Value> input_values;
    for (auto &tensor : input_tensors)
    {
        input_values.push_back(tensor);
    }

    vector<TensorValue<float>> output_tensors;
    vector<const char *> output_labels;
    for (const ValueInfo &info : modelInfo.GetOutput().GetAllTensors())
    {
        output_tensors.push_back(TensorValue(info, false));
        output_labels.push_back(info.GetName().c_str());
    }

    cout << "input values:" << endl;
    for (auto &tensor : input_tensors)
    {
        tensor.Print();
    }

    clock_t start = clock();
    vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
    cout << endl
         << "run-0(" << setiosflags(ios::fixed) << setprecision(2) << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)." << endl;
    // print with TensorValue
    for (int i = 0; i < output_values.size(); i++)
    {
        output_tensors[i].RecordOrtValue(output_values[i]);
    }

    for (auto &tensor : output_tensors)
    {
        tensor.Print();
    }
 
    // start to test run time
    for (int i = 0; i < 1000; i++)
    {
        // for (auto &tensor : input_tensors)
        // {
        //     tensor.Random();
        // }
        clock_t start = clock();
        vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
        cout << "run-" << i << "(" << setiosflags(ios::fixed) << setprecision(2) << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)."
             << "=> [" << setprecision(6) << *output_values[0].GetTensorMutableData<float>() << " ...]" << endl;

        // release memory
        for (auto &value : output_values)
        {
            Ort::OrtRelease(value.release());
        }
    }

    // print with Ort::Value
    // for(auto& value: output_values)
    // {
    //     std::cout << "--" << *value.GetTensorMutableData<float>() << endl;
    // }

    return 0;
}
