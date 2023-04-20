#include <cassert>

#include "GPUAllocator/Task.h"

Task::Task(std::string modelName, float limitCost, std::shared_ptr<ModelInfo> modelInfo, std::string tag) : limitCost(limitCost), modelName(modelName), tag(tag), modelInfo(modelInfo) {}

void Task::SetModelInfo(std::shared_ptr<ModelInfo> modelInfo)
{
    this->modelInfo = modelInfo;
}

float Task::TimeCost()
{
    return double(endTime - startTime) / CLOCKS_PER_SEC * 1000.0;
}

void Task::SetInputs(std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>> datas)
{
    for (const ValueInfo &info : this->modelInfo->GetInput().GetAllTensors())
    {
        this->Inputs.push_back(datas->at(info.GetName())); //
    }

    for (const ValueInfo &info : this->modelInfo->GetOutput().GetAllTensors())
    {
        switch (info.GetType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            this->Outputs.push_back(std::make_shared<TensorValue<int8_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            this->Outputs.push_back(std::make_shared<TensorValue<int16_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            this->Outputs.push_back(std::make_shared<TensorValue<int32_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            this->Outputs.push_back(std::make_shared<TensorValue<int64_t>>(info, false));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            this->Outputs.push_back(std::make_shared<TensorValue<float>>(info, false));
            break;
        default:
            this->Outputs.push_back(std::make_shared<TensorValue<float>>(info, false));
        }
    }

    for (auto &value : this->Inputs)
    {
        this->_input_datas.push_back(*value);
    }

    this->startTime = clock();
}

void Task::SetOutputs(std::vector<Ort::Value> &tensors)
{
    this->endTime = clock();
    for (int i = 0; i < tensors.size(); i++)
    {
        this->Outputs[i]->RecordOrtValue(tensors[i]);
    }
}

const std::vector<std::shared_ptr<TensorValueObject>> &Task::GetInputs()
{
    return this->Inputs;
}

const std::vector<std::shared_ptr<TensorValueObject>> &Task::GetOutputs()
{
    return this->Outputs;
}

void Task::RecordTimeCosts(clock_t start_time, clock_t end_time)
{
    this->timeCosts.push_back(std::pair<clock_t, clock_t>(start_time, end_time));
}

std::vector<std::pair<clock_t, clock_t>> &Task::GetTimeCosts()
{
    return this->timeCosts;
}

std::vector<float> Task::GetTimeCostsByMs()
{
    std::vector<float> result(timeCosts.size());
    for (int i = 0; i < timeCosts.size(); i++)
    {
        auto &value = timeCosts[i];
        result[i] = double(value.second - value.first) / CLOCKS_PER_SEC * 1000.0;
    }

    return result;
}

std::string &Task::GetTag()
{
    return this->tag;
}

std::string &Task::GetModelName()
{
    return this->modelName;
}

clock_t Task::GetStartTime()
{
    return this->startTime;
}

clock_t Task::GetEndTime()
{
    return this->endTime;
}

nlohmann::json Task::GetDescribe()
{
    nlohmann::json obj;
    obj["tag"] = tag;
    obj["model_name"] = modelName;
    obj["recv_time"] = startTime;
    obj["finish_time"] = endTime;
    obj["total_cost_by_ms"] = this->TimeCost();
    obj["child_model_execute_cost_by_ms"] = this->GetTimeCostsByMs();
    obj["child_model_run_time"] = this->GetTimeCosts();
    float execute_cost = [this]() -> float
    {
        float cost = 0.0;
        for (auto &value : this->GetTimeCostsByMs())
        {
            cost += value;
        }
        return cost;
    }();

    obj["execute_cost"] = execute_cost;

    obj["wait_cost"] = this->TimeCost() - execute_cost;
    obj["limit_cost_by_ms"] = this->limitCost;

    return obj;
}

void Task::PrintOutputs()
{
    for(auto output: this->GetOutputs())
    {
        std::cout<<output->GetValueInfo().GetName()<<std::endl;
        output->Print(30,false);
    }
}