#include <iostream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "../include/GPUAllocator/ExecutorManager.h"
#include "../include/Tensor/TensorValue.hpp"
#include "../include/Common/JsonSerializer.h"
#include "../include/Tensor/ModelInputCreator.h"
#include "../include/Common/PathManager.h"
using namespace std;
using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>>;

// g++ -DPARALLER_MODE for parallel
int main(int argc, char *argv[])
{
    int tokens[] = {3, 3, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1};
    if (argc >= 13)
    {
        for (int i = 0; i < 12; i++)
        {
            tokens[i] = atoi(argv[i + 1]);
        }
    }

    ExecutorManager executorManager;

    nlohmann::json vgg19 = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("vgg19"));
    ModelInfo vgg19Info(vgg19);
    ModelInputCreator vgg19Creator(vgg19Info.GetInput());
    executorManager.RunExecutor("vgg19"); // 1

    nlohmann::json resnet50 = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("resnet50"));
    ModelInfo resnet50Info(resnet50);
    ModelInputCreator resnet50Creator(resnet50Info.GetInput());
    executorManager.RunExecutor("resnet50"); // 2

    nlohmann::json googlenet = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath("googlenet"));
    ModelInfo googlenetInfo(googlenet);
    ModelInputCreator googlenetCreator(googlenetInfo.GetInput());
    executorManager.RunExecutor("googlenet"); // 3

    std::vector<DatasType> vgg19Datas;
    std::vector<DatasType> resnet50Datas;
    std::vector<DatasType> googlenetDatas;
    for (int i = 0; i < 2; i++)
    {
        vgg19Datas.push_back(vgg19Creator.CreateInput());
    }
    for (int i = 0; i < 2; i++)
    {
        resnet50Datas.push_back(resnet50Creator.CreateInput());
    }
    for (int i = 0; i < 2; i++)
    {
        googlenetDatas.push_back(googlenetCreator.CreateInput());
    }

    for (int i = 0; i < 2; i++)
    {
        executorManager.AddTask("vgg19", vgg19Datas[i], std::to_string(i));
    }
    for (int i = 0; i < 2; i++)
    {
        executorManager.AddTask("resnet50", resnet50Datas[i], std::to_string(i));
    }
    for (int i = 0; i < 2; i++)
    {
        executorManager.AddTask("googlenet", googlenetDatas[i], std::to_string(i));
    }

#ifndef PARALLER_MODE
    // cout << "use token:";
    // for (int token : tokens)
    // {
    //     cout << token << " ";
    //     executorManager.Grant(token);
    // }
    // cout << endl;
#else
    cout << "tokens deprecated, you have allow GPU parallel." << endl;
#endif // !1

    auto executorInfo = executorManager.GetExecutorInformation();
    for (auto iter = executorInfo.begin(); iter != executorInfo.end(); iter++)
    {
        auto &results = iter->second->executor->GetResultQueue();
        for (int i = 0; i < 2; i++)
        {
            auto result = results.Pop();
            // cout << result->GetModelName() << "-" << i << ": " << result->TimeCost() << "ms start=" << result->GetStartTime() << "," << result->GetEndTime() << endl;
            // for (auto t : result->GetTimeCostsByMs())
            // {
            //     cout << t << " ";
            // }

            // cout << setw(4) << result->GetDescribe();
            nlohmann::json describe = result->GetDescribe();
            cout << describe["model_name"].get<std::string>() << "-" << describe["tag"].get<std::string>() << ":" << endl;
            cout << "total(ms): " << describe["total_cost_by_ms"].get<float>() << " wait (ms): " << describe["wait_cost"].get<float>() << " execute (ms): " << describe["execute_cost"].get<float>() << endl
                 << endl;
        }
    }

    executorManager.Join();
    return 0;
}

// 3 3   2 2   2 2   1 1 1   1 1 1
/*

googlenet-0: 83.763ms start=6166155,6249918
83.731

googlenet-1: 119.054ms start=6173647,6292701
42.745

resnet50-0: 95.105ms start=6151218,6246323
55.3 39.781

resnet50-1: 159.077ms start=6158750,6317827
43.933 27.516

vgg19-0: 106.114ms start=6140261,6246375
36.04 31.432 38.623

vgg19-1: 193.785ms start=6143842,6337627
46.164 20.74 24.315

*/