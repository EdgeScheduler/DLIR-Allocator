#include "Tensor/ValueInfo.h"
#include <iostream>
namespace OnnxValueType
{
    std::string OnnxTypeToString(const ONNXTensorElementDataType type)
    {
        switch (type)
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64_t";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "double";

        default:
            return "unknown";
        }
    }

    ONNXTensorElementDataType StringToOnnxType(const std::string str)
    {
        if (str == "float32")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        else if (str == "uint8_t")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        }
        else if (str == "int8_t")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        }
        else if (str == "uint16_t")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        }
        else if (str == "int16_t")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        }
        else if (str == "int32_t")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
        else if (str == "int64_t")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
        else if (str == "string")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
        }
        else if (str == "double")
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        }
        else
        {
            std::cout << "warning: unknown type,  use float32 instead." << std::endl;
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
    }
}

nlohmann::json ValueInfo::ToJson() const
{
    nlohmann::json obj;
    if (this->name == "" && this->shape.size() < 1)
    {
        return std::move(nullptr);
    }

    obj["type"] = OnnxValueType::OnnxTypeToString(this->type);
    obj["name"] = this->name;
    obj["shape"] = this->shape;

    return obj;
}

void ValueInfo::LoadFromJson(const nlohmann::json &json)
{
    if (json == nullptr)
    {
        this->name = "";
        this->shape.clear();
        this->type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    else
    {
        this->name = json["name"].get<std::string>();
        this->type = OnnxValueType::StringToOnnxType(json["type"].get<std::string>());
        this->shape = json["shape"].get<std::vector<int64_t>>();
    }

    this->UpdateCount();
}

ValueInfo::ValueInfo(const nlohmann::json &json)
{
    this->LoadFromJson(json);
}

ValueInfo::ValueInfo(const std::string &name, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type)
{
    this->name = name;
    this->shape = shape;
    this->type = type;
    this->UpdateCount();
}

ValueInfo::ValueInfo(const char *name, const std::vector<int64_t> &shape, const ONNXTensorElementDataType &type)
{
    this->name = std::string(name);
    this->shape = shape;
    this->type = type;
    this->UpdateCount();
}

void ValueInfo::UpdateCount()
{
    if (this->shape.size() < 1)
    {
        this->dataCount = 0;
        return;
    }
    else
    {
        this->dataCount = 1;
        for (auto value : this->shape)
        {
            this->dataCount *= value;
        }
    }
}

const std::string &ValueInfo::GetName() const
{
    return this->name;
}

void ValueInfo::SetName(std::string name)
{
    this->name = name;
}

const std::vector<int64_t> &ValueInfo::GetShape() const
{
    return this->shape;
}

const ONNXTensorElementDataType &ValueInfo::GetType() const
{
    return this->type;
}

const std::int64_t &ValueInfo::GetDataCount() const
{
    return this->dataCount;
}

std::string ValueInfo::GetTypeString() const
{
    return OnnxValueType::OnnxTypeToString(this->type);
}

std::size_t ValueInfo::GetDimSize() const
{
    return this->shape.size();
}

std::ostream &operator<<(std::ostream &out, const ValueInfo &value)
{
    out << value.name << ": [";
    for (auto iter = value.shape.begin(); iter < value.shape.end(); iter++)
    {
        if (iter == value.shape.begin())
        {
            out << *iter;
        }
        else
        {
            out << ", " << *iter;
        }
    }
    out << "], type=" << value.GetTypeString();

    return out;
}