#include "GPUAllocator/ModelExecutor.h"
#include "Common/JsonSerializer.h"
#include "Common/PathManager.h"
#include "Tensor/ModelInputCreator.h"
#include "Common/DLIException.h"
#include <vector>
#include <iostream>
#include <ctime>

#define ErrorRange 1.05

ModelExecutor::ModelExecutor(std::string model_name, Ort::SessionOptions *session_opt, Ort::Env *env, int token_id, TokenManager *token_manager, std::mutex *gpu_mutex, std::condition_variable *deal_task) : modelName(model_name), sessionOption(session_opt), onnxruntimeEnv(env), todo(0), modelCount(0), tokenID(token_id), tokenManager(token_manager), gpuMutex(gpu_mutex), dealTask(deal_task), rawSession(nullptr), closeExecutor(false)
{
    std::filesystem::path rawModelPath = OnnxPathManager::GetModelSavePath(modelName);
    this->executeTime = std::make_shared<std::vector<float>>();
    this->rawSession = std::make_shared<Ort::Session>(*onnxruntimeEnv, rawModelPath.c_str(), *sessionOption);
    this->rawModelInfo = std::make_shared<ModelInfo>(*rawSession, rawModelPath);

    std::filesystem::path modelSumParamsPath = OnnxPathManager::GetChildModelSumParamsSavePath(modelName);
    nlohmann::json json = JsonSerializer::LoadJson(modelSumParamsPath);
    int start = 0;
    while (json.contains(std::to_string(start)))
    {
        std::filesystem::path model_path = OnnxPathManager::GetChildModelSavePath(modelName, start);
        Ort::Session session(*onnxruntimeEnv, model_path.c_str(), *sessionOption);

        this->modelInfos.push_back(ModelInfo(session, model_path));
        this->sessions.push_back(std::move(session));
        this->modelCount += 1;
        start++;
    }

    for (auto &modelInfo : modelInfos)
    {
        std::vector<const char *> inputs;
        for (const ValueInfo &info : modelInfo.GetInput().GetAllTensors())
        {
            inputs.push_back(info.GetName().c_str());
        }
        this->inputLabels.push_back(inputs);

        std::vector<const char *> outputs;
        for (const ValueInfo &info : modelInfo.GetOutput().GetAllTensors())
        {
            outputs.push_back(info.GetName().c_str());
        }
        this->outputLabels.push_back(outputs);
    }

    // run test to skip cold-run and get run-time

    // test child-models
    {
        std::cout << "start to run " << modelName << " test." << std::endl;
        clock_t sum_raw = 0.0;
        for (int i = 0; i < modelCount; i++)
        {
            // child-model
            std::vector<std::shared_ptr<TensorValueObject>> input_tensors;
            std::vector<Ort::Value> input_values;
            for (auto &info : modelInfos[i].GetInput().GetAllTensors())
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

                // input_Tensors.push_back(TensorValue(info, true));
            }

            for (auto &tensor : input_tensors)
            {
                input_values.push_back(*tensor);
            }

            // raw-model
            std::vector<std::shared_ptr<TensorValueObject>> raw_input_tensors;
            std::vector<Ort::Value> raw_input_values;
            for (auto &info : modelInfos[0].GetInput().GetAllTensors())
            {
                switch (info.GetType())
                {
                case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                    raw_input_tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, true));
                    break;
                case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                    raw_input_tensors.push_back(std::make_shared<TensorValue<int16_t>>(info, true));
                    break;
                case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    raw_input_tensors.push_back(std::make_shared<TensorValue<int32_t>>(info, true));
                    break;
                case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    raw_input_tensors.push_back(std::make_shared<TensorValue<int64_t>>(info, true));
                    break;
                case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    raw_input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
                    break;
                default:
                    raw_input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
                }
            }

            for (auto &tensor : raw_input_tensors)
            {
                raw_input_values.push_back(*tensor);
            }

            // skip child cold-run
            for (int k = 0; k < 3; k++)
            {
                this->sessions[i].Run(Ort::RunOptions{nullptr}, inputLabels[i].data(), input_values.data(), inputLabels[i].size(), outputLabels[i].data(), outputLabels[i].size());
            }

            // skip raw cold-run
            for (int k = 0; k < 3; k++)
            {
                this->rawSession->Run(Ort::RunOptions{nullptr}, inputLabels[0].data(), raw_input_values.data(), inputLabels[0].size(), outputLabels[modelCount - 1].data(), outputLabels[modelCount - 1].size());
            }

            // evaluate the time-cost
            clock_t sum_time = 0.0;
            for (int k = 0; k < 3; k++)
            {
                for (auto &tensor : input_tensors)
                {
                    tensor->Random();
                }

                for (auto &tensor : raw_input_tensors)
                {
                    tensor->Random();
                }

                clock_t start = clock();
                auto result = this->sessions[i].Run(Ort::RunOptions{nullptr}, inputLabels[i].data(), input_values.data(), inputLabels[i].size(), outputLabels[i].data(), outputLabels[i].size());
                sum_time += clock() - start;

                clock_t raw_start = clock();
                auto raw_result = this->rawSession->Run(Ort::RunOptions{nullptr}, inputLabels[0].data(), raw_input_values.data(), inputLabels[0].size(), outputLabels[modelCount - 1].data(), outputLabels[modelCount - 1].size());
                sum_raw += clock() - raw_start;
            }
            this->executeTime->push_back(sum_time / 3.0 / CLOCKS_PER_SEC * 1000.0);
            // test child-models end.
        }
        this->modelExecuteTime = sum_raw / (3.0 * modelCount) / CLOCKS_PER_SEC * 1000.0;

        // test raw-model
        {
            // // run to skip cold-run
            // std::vector<std::shared_ptr<TensorValueObject>> raw_input_tensors;
            // std::vector<Ort::Value> raw_input_values;
            // for (auto &info : modelInfos[0].GetInput().GetAllTensors())
            // {
            //     switch (info.GetType())
            //     {
            //     case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            //         raw_input_tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, true));
            //         break;
            //     case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            //         raw_input_tensors.push_back(std::make_shared<TensorValue<int16_t>>(info, true));
            //         break;
            //     case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            //         raw_input_tensors.push_back(std::make_shared<TensorValue<int32_t>>(info, true));
            //         break;
            //     case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            //         raw_input_tensors.push_back(std::make_shared<TensorValue<int64_t>>(info, true));
            //         break;
            //     case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            //         raw_input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
            //         break;
            //     default:
            //         raw_input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
            //     }
            // }

            // for (auto &tensor : raw_input_tensors)
            // {
            //     raw_input_values.push_back(*tensor);
            // }

            // // skip cold-run
            // for (int k = 0; k < 3; k++)
            // {
            //     this->rawSession->Run(Ort::RunOptions{nullptr}, inputLabels[0].data(), raw_input_values.data(), inputLabels[0].size(), outputLabels[modelCount - 1].data(), outputLabels[modelCount - 1].size());
            // }

            // evaluate the time-cost
            // clock_t sumt_raw = 0.0;
            // for (int k = 0; k < 1; k++)
            // {
            //     for(auto &tensor: raw_input_tensors)
            //     {
            //         tensor->Random();
            //     }
            //     clock_t start = clock();
            //     auto result=this->rawSession->Run(Ort::RunOptions{nullptr}, inputLabels[0].data(), raw_input_values.data(), inputLabels[0].size(), outputLabels[modelCount - 1].data(), outputLabels[modelCount - 1].size());
            //     sumt_raw+=clock() - start;
            // }
            // this->modelExecuteTime = sumt_raw / 1.0 / CLOCKS_PER_SEC * 1000.0;

            // test raw-model end.
        }

        // evaluate end
    }

    // test end

    if (modelCount > 0)
    {
        // replace model-0 by raw-input
        std::vector<const char *> inputs;
        for (const ValueInfo &info : rawModelInfo->GetInput().GetAllTensors())
        {
            inputs.push_back(info.GetName().c_str());
        }
        this->inputLabels[0] = inputs;

        std::vector<const char *> outputs;
        for (const ValueInfo &info : rawModelInfo->GetOutput().GetAllTensors())
        {
            outputs.push_back(info.GetName().c_str());
        }
        this->outputLabels[modelCount - 1] = outputs;
    }

    // evaluate end
    std::cout << "run " << modelName << " test to end." << std::endl;
}

void ModelExecutor::UpdateExecutorTime()
{
    std::cout << "start to re-run " << modelName << " test." << std::endl;
    this->executeTime->clear();

    for (int i = 0; i < modelCount; i++)
    {
        // child-model
        std::vector<std::shared_ptr<TensorValueObject>> input_tensors;
        std::vector<Ort::Value> input_values;
        for (auto &info : modelInfos[i].GetInput().GetAllTensors())
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

            // input_Tensors.push_back(TensorValue(info, true));
        }

        for (auto &tensor : input_tensors)
        {
            input_values.push_back(*tensor);
        }

        // evaluate the time-cost

        clock_t start = clock();
        auto result = this->sessions[i].Run(Ort::RunOptions{nullptr}, inputLabels[i].data(), input_values.data(), inputLabels[i].size(), outputLabels[i].data(), outputLabels[i].size());

        this->executeTime->push_back((clock() - start) / CLOCKS_PER_SEC * 1000.0);

                // test child-models end.
    }

    // raw-model
    std::vector<std::shared_ptr<TensorValueObject>> raw_input_tensors;
    std::vector<Ort::Value> raw_input_values;
    for (auto &info : modelInfos[0].GetInput().GetAllTensors())
    {
        switch (info.GetType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            raw_input_tensors.push_back(std::make_shared<TensorValue<int8_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            raw_input_tensors.push_back(std::make_shared<TensorValue<int16_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            raw_input_tensors.push_back(std::make_shared<TensorValue<int32_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            raw_input_tensors.push_back(std::make_shared<TensorValue<int64_t>>(info, true));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            raw_input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
            break;
        default:
            raw_input_tensors.push_back(std::make_shared<TensorValue<float>>(info, true));
        }
    }

    for (auto &tensor : raw_input_tensors)
    {
        raw_input_values.push_back(*tensor);
    }

    clock_t raw_start = clock();
    auto raw_result = this->rawSession->Run(Ort::RunOptions{nullptr}, inputLabels[0].data(), raw_input_values.data(), inputLabels[0].size(), outputLabels[modelCount - 1].data(), outputLabels[modelCount - 1].size());

    //std::cout<<clock()<<"--"<<raw_start<<std::endl;
    this->modelExecuteTime = (clock() - raw_start)* 1000.0 / CLOCKS_PER_SEC ;
    //std::cout<<"=>"<<this->modelExecuteTime<<std::endl;
}

ModelExecutor::~ModelExecutor()
{
    Ort::OrtRelease(this->rawSession->release());
    for (auto &session : this->sessions)
    {
        Ort::OrtRelease(session.release());
    }
}

void ModelExecutor::ToNext(bool toEnd)
{
    try
    {
        if (toEnd)
        {
            this->todo = 0;
        }
        else
        {
            this->todo = (this->todo + 1) % this->modelCount;
        }
        if (this->todo == 0)
        {
            this->current_task->SetOutputs(this->current_task->_input_datas);
            this->finish_queue.Emplace(std::move(this->task_queue.Pop()));
            this->current_task = nullptr;
        }
    }
    catch (DLIException ex)
    {
        if (ex == DLIException::SYSTEM_CLOSE)
        {
            throw ex;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

void ModelExecutor::LoadTask()
{
    try
    {
        if (this->todo == 0)
        {
            // block while empty
            this->current_task = this->task_queue.front();
        }

        current_task->_session = &this->sessions[this->todo];
        current_task->_input_labels = &this->inputLabels[this->todo];
        current_task->_output_labels = &this->outputLabels[this->todo];
    }
    catch (DLIException ex)
    {
        if (ex == DLIException::SYSTEM_CLOSE)
        {
            throw ex;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

bool ModelExecutor::RunOnce()
{
    this->LoadTask();
    if (this->current_task == nullptr)
    {
        std::cout << "warning: meet no input." << std::endl;
        return true;
    }

#ifndef PARALLER_MODE
    std::unique_lock<std::mutex> lock(*gpuMutex);
    dealTask->wait(lock, [this]() -> bool
                   { return (this->tokenManager->GetFlag() >> 1) == tokenID || closeExecutor; });
    bool rawRun = ((this->tokenManager->GetFlag() & 0x1) == 1);

    this->tokenManager->Expire();
    lock.unlock(); // release lock here to satisfy notify_all broadcast. Token can still ensure gpu-mutex

    if (closeExecutor)
    {
        if ((this->tokenManager->GetFlag() >> 1) == tokenID)
        {
            this->tokenManager->Release();
        }
        throw DLIException::SYSTEM_CLOSE;
    }

    // std::cout<<"recv token: "<<this->tokenManager->GetFlag()<<std::endl;

    Ort::Session *runSession = nullptr;
    if (rawRun)
    {
        runSession = this->rawSession.get();
        current_task->_output_labels = &this->outputLabels[this->modelCount - 1];
    }
    else
    {
        runSession = current_task->_session;
    }
    // use token already
    // this->tokenManager->Release();
#else
    Ort::Session *runSession = this->rawSession.get();
    current_task->_output_labels = &this->outputLabels[this->modelCount - 1];
#endif // !PARALLER_MODE

    clock_t start = clock();
    // here may need to consider release old current_task->_input_datas if this->todo>0
    current_task->_input_datas = runSession->Run(Ort::RunOptions{nullptr}, current_task->_input_labels->data(), current_task->_input_datas.data(), current_task->_input_labels->size(), current_task->_output_labels->data(), current_task->_output_labels->size());
    current_task->RecordTimeCosts(start, clock());

#ifndef PARALLER_MODE
    this->tokenManager->Release();
    if (rawRun)
    {
        this->ToNext(true);
    }
    else
    {
        this->ToNext(false);
    }
#else
    this->ToNext(true);
#endif
    return false;
}

void ModelExecutor::AddTask(std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>> datas, std::string tag)
{
    try
    {
        std::shared_ptr<Task> new_task = std::make_shared<Task>(this->modelName, this->modelExecuteTime, this->rawModelInfo, tag);
        new_task->SetInputs(datas);
        this->task_queue.Push(new_task);
    }
    catch (DLIException ex)
    {
        if (ex == DLIException::SYSTEM_CLOSE)
        {
            return;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

void ModelExecutor::RunCycle()
{
    try
    {
        if (gpuMutex == nullptr || tokenManager == nullptr)
        {
            std::cout << "you give no device-mutex and token-manager info, system exit. This mode is only only allow while compiler with \"-DPARALLER_MODE\"" << std::endl;
            return;
        }
        while (true)
        {
            if (this->RunOnce())
            {
                break;
            }
        }
    }
    catch (DLIException ex)
    {
        if (ex == DLIException::SYSTEM_CLOSE)
        {
            std::cout << this->modelName << " executor end sync." << std::endl;
            return;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

SafeQueue<std::shared_ptr<Task>> &ModelExecutor::GetResultQueue()
{
    return this->finish_queue;
}

SafeQueue<std::shared_ptr<Task>> &ModelExecutor::GetTaskQueue()
{
    return this->task_queue;
}

std::shared_ptr<std::vector<float>> ModelExecutor::GetExecuteTime()
{
    return this->executeTime;
}

int ModelExecutor::GetTokenID()
{
    return this->tokenID;
}

float &ModelExecutor::GetModelExecuteTime()
{
    return this->modelExecuteTime;
}

int ModelExecutor::GetChildModelCount()
{
    return this->modelCount;
}

void ModelExecutor::CloseExecutor()
{
    this->closeExecutor = true;
    this->dealTask->notify_all();
    this->task_queue.Close();
    this->finish_queue.Close();
}
