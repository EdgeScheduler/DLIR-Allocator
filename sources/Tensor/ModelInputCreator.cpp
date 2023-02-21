#include "Tensor/ModelInputCreator.h"

ModelInputCreator::ModelInputCreator(const TensorsInfo &tensorsInfo) : tensorsInfo(tensorsInfo)
{
    for (const ValueInfo &info : tensorsInfo.GetAllTensors())
    {
        std::shared_ptr<TensorValueObject> ptr;
        switch (info.GetType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            ptr = std::make_shared<TensorValue<int8_t>>(info, true);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            ptr =std::make_shared<TensorValue<int16_t>>(info, true);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            ptr =std::make_shared<TensorValue<int32_t>>(info, true);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            ptr =std::make_shared<TensorValue<int64_t>>(info, true);
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            ptr =std::make_shared<TensorValue<float>>(info, true);
            break;
        default:
            ptr =std::make_shared<TensorValue<float>>(info, true);
        }

        dataTemplate.insert(std::pair<std::string, std::shared_ptr<TensorValueObject>>(info.GetName(), ptr));
    }
}

std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>> ModelInputCreator::CreateInput()
{
    std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>> datas = std::make_shared<std::map<std::string, std::shared_ptr<TensorValueObject>>>();
    for (auto iter = dataTemplate.begin(); iter != dataTemplate.end(); iter++)
    {
        iter->second->Random();

        std::shared_ptr<TensorValueObject> ptr;
        switch (iter->second->GetDataElementType())
        {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            ptr = std::make_shared<TensorValue<int8_t>>(*(std::dynamic_pointer_cast<TensorValue<int8_t>>(iter->second)));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            ptr =std::make_shared<TensorValue<int16_t>>(*(std::dynamic_pointer_cast<TensorValue<int16_t>>(iter->second)));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            ptr =std::make_shared<TensorValue<int32_t>>(*(std::dynamic_pointer_cast<TensorValue<int32_t>>(iter->second)));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            ptr =std::make_shared<TensorValue<int64_t>>(*(std::dynamic_pointer_cast<TensorValue<int64_t>>(iter->second)));
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            ptr =std::make_shared<TensorValue<float>>(*(std::dynamic_pointer_cast<TensorValue<float>>(iter->second)));
            break;
        default:
            ptr =std::make_shared<TensorValue<float>>(*(std::dynamic_pointer_cast<TensorValue<float>>(iter->second)));
        }

        datas->insert(std::pair<std::string, std::shared_ptr<TensorValueObject>>(iter->first, ptr));
    }
    return datas;
}
