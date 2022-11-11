#include "Tensor/ModelTensorsInfo.h"
#include <iostream>
#include <string>
// TensorsInfo
TensorsInfo::TensorsInfo(const std::vector<std::string> &labels, const std::vector<std::vector<int64_t>> &shapes, const std::vector<ONNXTensorElementDataType> &types)
{
    this->SetTensorsInfo(labels, shapes, types);
}

TensorsInfo::TensorsInfo(const nlohmann::json &json)
{
    this->LoadFromJson(json);
}

int TensorsInfo::SetTensorsInfo(const std::vector<std::string> &labels, const std::vector<std::vector<int64_t>> &shapes, const std::vector<ONNXTensorElementDataType> &types)
{
    this->tensors.clear();
    int length = labels.size() < shapes.size() ? labels.size() : shapes.size();

    for (int i = 0; i < length; i++)
    {
        this->AppendTensorInfo(labels[i], shapes[i], types[i]);
    }

    return length;
}

void TensorsInfo::LoadFromJson(const nlohmann::json &json)
{
    std::vector<nlohmann::json> obj = json.get<std::vector<nlohmann::json>>();
    for (auto iter = obj.begin(); iter < obj.end(); iter++)
    {
        this->tensors.push_back(ValueInfo(*iter));
    }
}

nlohmann::json TensorsInfo::ToJson() const
{
    if (this->tensors.size() < 1)
    {
        return nullptr;
    }

    std::vector<nlohmann::json> obj;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        obj.push_back(iter->ToJson());
    }

    return obj;
}

void TensorsInfo::AppendTensorInfo(const char *label, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type)
{
    this->tensors.push_back(ValueInfo(label, shape, type));
}

void TensorsInfo::AppendTensorInfo(const std::string &label, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type)
{
    this->tensors.push_back(ValueInfo(label, shape, type));
}

std::vector<std::string> TensorsInfo::GetLabels() const
{
    std::vector<std::string> labels;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        labels.push_back(iter->GetName());
    }
    return labels;
}

std::vector<std::vector<int64_t>> TensorsInfo::GetShapes() const
{
    std::vector<std::vector<int64_t>> shapes;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        shapes.push_back(iter->GetShape());
    }
    return shapes;
}

std::vector<ONNXTensorElementDataType> TensorsInfo::GetTypes() const
{
    std::vector<ONNXTensorElementDataType> types;
    for (auto iter = this->tensors.begin(); iter < this->tensors.end(); iter++)
    {
        types.push_back(iter->GetType());
    }
    return types;
}

int TensorsInfo::GetTensorsCount() const
{
    return this->tensors.size();
}

const std::vector<ValueInfo> &TensorsInfo::GetAllTensors() const
{
    return this->tensors;
}

std::ostream &operator<<(std::ostream &out, const TensorsInfo &value)
{
    for (ValueInfo tensor : value.tensors)
    {
        out << tensor << std::endl;
    }

    return out;
}

// ModelInfo

ModelInfo::ModelInfo(const nlohmann::json &json)
{
    this->LoadFromJson(json);
}

ModelInfo::ModelInfo(const Ort::Session &session, std::filesystem::path model_path)
{
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < session.GetInputCount(); i++)
    {
        // need to get shape
        // const Ort::TensorTypeAndShapeInfo& info=session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        this->input.AppendTensorInfo(session.GetInputNameAllocated(i, allocator).get(), session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape(), session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
    }

    for (int i = 0; i < session.GetOutputCount(); i++)
    {
        // need to get shape
        // const Ort::TensorTypeAndShapeInfo& info=session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        this->output.AppendTensorInfo(session.GetOutputNameAllocated(i, allocator).get(), session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape(), session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
    }

    this->modelPath = model_path;
}

void ModelInfo::SetModelPath(std::filesystem::path model_path)
{
    this->modelPath = model_path;
}

std::filesystem::path ModelInfo::GetModelPath()
{
    return this->modelPath;
}

nlohmann::json ModelInfo::ToJson() const
{
    nlohmann::json obj;
    nlohmann::json input;
    nlohmann::json output;

    input["data"] = this->input.ToJson();
    output["data"] = this->output.ToJson();

    obj["input"] = input;
    obj["output"] = output;
    obj["model_path"] = std::string(this->modelPath);
    return obj;
}

void ModelInfo::LoadFromJson(const nlohmann::json &json)
{
    if (json.contains("input"))
    {
        this->input = TensorsInfo(json["input"]["data"]);
    }

    if (json.contains("output"))
    {
        this->output = TensorsInfo(json["output"]["data"]);
    }

    if (json.contains("model_path"))
    {
        this->modelPath = json["model_path"].get<std::string>();
    }
    else
    {
        this->modelPath = "";
    }
}

const TensorsInfo &ModelInfo::GetInput() const
{
    return this->input;
}

const TensorsInfo &ModelInfo::GetOutput() const
{
    return this->output;
}

std::ostream &operator<<(std::ostream &out, const ModelInfo &value)
{
    out << "input:" << std::endl;
    out << value.input << std::endl;
    out << "output:" << std::endl;
    ;
    out << value.output;

    return out;
}
