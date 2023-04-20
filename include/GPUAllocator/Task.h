#ifndef __TASK_H__
#define __TASK_H__

#include <ctime>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_cxx_api.h>
#include "Tensor/ModelTensorsInfo.h"
#include "Tensor/TensorValue.hpp"
#include "Tensor/ValueInfo.h"

class Task
{
public:
    Task(std::string modelName, float limitCost, std::shared_ptr<ModelInfo> modelInfo = nullptr, std::string tag = "");
    /// @brief set model-infos
    /// @param modelInfo
    void SetModelInfo(std::shared_ptr<ModelInfo> modelInfo);

    /// @brief get how much time cost by task.(ms)
    /// @return
    float TimeCost();

    /// @brief Set model-input-value (data)
    /// @param tensors
    void SetInputs(std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>> datas);

    /// @brief Get model-input reference (data)
    /// @return
    const std::vector<std::shared_ptr<TensorValueObject>> &GetInputs();

    /// @brief record model inference result (model-output)
    /// @param tensors
    void SetOutputs(std::vector<Ort::Value> &tensors);

    /// @brief Get model-output reference
    /// @return
    const std::vector<std::shared_ptr<TensorValueObject>> &GetOutputs();

    /// @brief record (start_time, end_time) for each child-module. (clock_t, us)
    /// @param cost
    void RecordTimeCosts(clock_t start_time, clock_t end_time);

    /// @brief get tag
    /// @return
    std::string &GetTag();

    /// @brief get this task is make for which model
    /// @return
    std::string &GetModelName();

    /// @brief get (start_time, end_time) for each child-model.(clock_t, us)
    /// @return
    std::vector<std::pair<clock_t, clock_t>> &GetTimeCosts();

    /// @brief get how much time cost by run for each child-model.(ms)
    /// @return
    std::vector<float> GetTimeCostsByMs();

    /// @brief get when the task received
    /// @return
    clock_t GetStartTime();

    /// @brief get when the task get result.
    /// @return
    clock_t GetEndTime();

    nlohmann::json GetDescribe();

    void PrintOutputs();

public:
    std::vector<std::shared_ptr<TensorValueObject>> Inputs;
    std::vector<std::shared_ptr<TensorValueObject>> Outputs;
    std::vector<std::pair<clock_t, clock_t>> timeCosts;

    // runtime args
public:
    Ort::Session *_session;
    std::vector<const char *> *_input_labels;
    std::vector<const char *> *_output_labels;
    std::vector<Ort::Value> _input_datas;

private:
    clock_t startTime;
    clock_t endTime;
    std::string tag;
    std::string modelName;
    std::shared_ptr<ModelInfo> modelInfo;
    float limitCost;
};

#endif // __TASK_H__