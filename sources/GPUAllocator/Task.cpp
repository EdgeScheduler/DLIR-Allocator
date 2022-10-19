#include <cassert>

#include "../../include/GPUAllocator/Task.h"

Task::Task(std::string modelName, std::shared_ptr<ModelInfo> modelInfo, std::string tag) : modelName(modelName), tag(tag), modelInfo(modelInfo) {}

void Task::SetModelInfo(std::shared_ptr<ModelInfo> modelInfo)
{
    this->modelInfo = modelInfo;
}

float Task::TimeCost()
{
    return double(endTime - startTime) / CLOCKS_PER_SEC * 1000.0;
}

void Task::SetInputs(std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> datas)
{
    for (const ValueInfo &info : this->modelInfo->GetInput().GetAllTensors())
    {
        this->Inputs.push_back(datas->at(info.GetName())); //
    }

    for (const ValueInfo &info : this->modelInfo->GetOutput().GetAllTensors())
    {
        this->Outputs.push_back(std::make_shared<TensorValue<float>>(info, false));
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

const std::vector<std::shared_ptr<TensorValue<float>>> &Task::GetInputs()
{
    return this->Inputs;
}

const std::vector<std::shared_ptr<TensorValue<float>>> &Task::GetOutputs()
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
    obj["finish_time"] = startTime;
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

    return obj;
}
