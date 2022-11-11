#include "Tensor/ModelInputCreator.h"

ModelInputCreator::ModelInputCreator(const TensorsInfo &tensorsInfo) : tensorsInfo(tensorsInfo)
{
    for (const ValueInfo &info : tensorsInfo.GetAllTensors())
    {
        dataTemplate.insert(std::pair<std::string, TensorValue<float>>(info.GetName(), TensorValue(info, false)));
    }
}

std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> ModelInputCreator::CreateInput()
{
    std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> datas = std::make_shared<std::map<std::string, std::shared_ptr<TensorValue<float>>>>();
    for (auto iter = dataTemplate.begin(); iter != dataTemplate.end(); iter++)
    {
        iter->second.Random();
        datas->insert(std::pair<std::string, std::shared_ptr<TensorValue<float>>>(iter->first, std::make_shared<TensorValue<float>>(iter->second)));
    }
    return datas;
}
