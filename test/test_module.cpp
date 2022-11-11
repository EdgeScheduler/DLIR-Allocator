#include <iostream>
#include "Common/Drivers.h"
#include "Common/PathManager.h"
#include "Common/JsonSerializer.h"
#include "Tensor/TensorValue.hpp"

using namespace std;

/// @brief test partial-function
/// @return
int main()
{
    // std::string model_name = "googlenet";
    // std::cout<<Drivers::CPUDriver<<std::endl;

    // test PathManager
    //  cout << RootPathManager::GetRunRootFold() << endl;
    //  cout << OnnxPathManager::GetOnnxRootFold() << endl;
    //  cout << OnnxPathManager::GetModelSavePath(model_name) << endl;
    //  cout << OnnxPathManager::GetModelParamsSavePath(model_name) << endl;
    //  cout << OnnxPathManager::GetChildModelSavePath(model_name,0) << endl;
    //  cout << OnnxPathManager::GetChildModelParamsSavePath(model_name,0) << endl;
    //  cout << OnnxPathManager::GetChildModelSumParamsSavePath(model_name) << endl;
    //  cout << OnnxPathManager::GetChildModelSumCacheSavePath(model_name) << endl;

    // auto a=JsonSerializer::LoadJson("test.json");
    // JsonSerializer::StoreJson(a,"aa/test_1.json",true);

    // ValueInfo info("label",{1,2,2});
    // TensorValue value(info,true);
    // value.Print(1,false);
    // Ort::Value o_value=value;        // copy

    // TensorValue v1(info,false);
    // v1.RecordOrtValue(o_value);      // deep-copy

    // value.Random();

    // TensorValue v2=v1;               // deep-copy

    // std::cout<<"--"<<*o_value.GetTensorMutableData<float>()<<endl;
    // v1.Print(1,false);

    // v2.Random();
    // v1.Print(1,false);
    // v2.Print(1,false);

    // test override
    ValueInfo info_new("label-new", {2, 2, 2});
    ValueInfo info_old("label-old", {1, 2, 2});
    TensorValue value_new(info_new, true);
    Ort::Value v = value_new;

    TensorValue v2(info_old, false);
    v2.RecordOrtValue(v);
    v2.Print();

    v2.RecordOrtValueIgnoreShape(v);
    v2.Print();

    return 0;
}