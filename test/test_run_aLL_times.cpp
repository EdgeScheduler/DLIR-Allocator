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

#define TYPE float

// ./exec $onnx_path_name
// 41.6 28.3ms 24.3ms 18.5ms
int main(int argc, char *argv[])
{
    std::filesystem::path model_path = RootPathManager::GetRunRootFold() / std::string(argv[1]);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); // log id: "test"
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);
    Ort::Session session(env, model_path.c_str(), session_options);

    ModelInfo modelInfo(session);
    cout << modelInfo << endl;

    cout<<modelInfo.GetInput()<<endl;
    cout<<modelInfo.GetOutput()<<endl;

    cout << modelInfo.ToJson()<<endl;
    cout << modelInfo << endl;

    cout << "input:" << endl;
    vector<std::shared_ptr<TensorValueObject>> input_tensors;
    vector<const char *> input_labels;

    for (const ValueInfo &info : modelInfo.GetInput().GetAllTensors())
    {
        switch (info.GetType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            input_tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            input_tensors.push_back(std::make_shared<TensorValue<int16_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            input_tensors.push_back(std::make_shared<TensorValue<int32_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            input_tensors.push_back(std::make_shared<TensorValue<int64_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
            break;
        default:
            input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
        }

        input_labels.push_back(info.GetName().c_str());
    }

    vector<Ort::Value> input_values;
    for (auto &tensor : input_tensors)
    {
        input_values.push_back(*tensor);
    }

    vector<shared_ptr<TensorValueObject>> output_tensors;
    vector<const char *> output_labels;
    for (const ValueInfo &info : modelInfo.GetOutput().GetAllTensors())
    {
        switch (info.GetType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            output_tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            output_tensors.push_back(std::make_shared<TensorValue<int16_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            output_tensors.push_back(std::make_shared<TensorValue<int32_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            output_tensors.push_back(std::make_shared<TensorValue<int64_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            output_tensors.push_back(std::make_shared<TensorValue<float>>(info, false));
            break;
        default:
            output_tensors.push_back(std::make_shared<TensorValue<float>>(info, false));
        }

        output_labels.push_back(info.GetName().c_str());
    }

    // cout << "input values:" << endl;
    // for (auto &tensor : input_tensors)
    // {
    //     tensor.Print();
    // }
    std::cout << argv[1] << "(ms):" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        clock_t start = clock();
        vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
        std::cout << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << endl;
    }

    for (int i = 0; i < 30; i++)
    {
        clock_t start = clock();
        vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
        std::cout << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << endl;
    }

    // release memory
    // for (auto &value : output_values)
    // {
    //     Ort::OrtRelease(value.release());
    // }

    // cout << endl << "run-0(" << setiosflags(ios::fixed) << setprecision(2) << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)." << endl;
    // // print with TensorValue
    // for (int i = 0; i < output_values.size(); i++)
    // {
    //     output_tensors[i].RecordOrtValue(output_values[i]);
    // }

    // for (auto &tensor : output_tensors)
    // {
    //     tensor.Print();
    // }

    // start to test run time
    // for (int i = 0; i < 1000; i++)
    // {
    //     // for (auto &tensor : input_tensors)
    //     // {
    //     //     tensor.Random();
    //     // }
    //     clock_t start = clock();
    //     vector<Ort::Value> output_values = session.Run(Ort::RunOptions{nullptr}, input_labels.data(), input_values.data(), input_labels.size(), output_labels.data(), output_labels.size());
    //     cout << "run-" << i << "(" << setiosflags(ios::fixed) << setprecision(2) << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms)."
    //          << "=> [" << setprecision(6) << *output_values[0].GetTensorMutableData<float>() << " ...]" << endl;

    //     // release memory
    //     for (auto &value : output_values)
    //     {
    //         Ort::OrtRelease(value.release());
    //     }
    // }

    // print with Ort::Value
    // for(auto& value: output_values)
    // {
    //     std::cout << "--" << *value.GetTensorMutableData<float>() << endl;
    // }

    return 0;
}
