#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
// grpc 头文件
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include "RPCResponse/GRPCInterface.h"
#include "RPCResponse/RPCServer.h"

using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>>;

void RunRPCServer(ExecutorManager *executorManager,std::map<std::string, ModelInfo>* modelInfos,int port)
{
    std::string server_addr(std::string("0.0.0.0:")+std::to_string(port));
    GRPCInterface grpcservice(executorManager,modelInfos,port);

    grpc::ServerBuilder builder;

    builder.SetMaxSendMessageSize(1024*1024*1024);
    builder.SetMaxReceiveMessageSize(1024*1024*1024);
    builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 5000);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    builder.SetSyncServerOption(grpc::ServerBuilder::MIN_POLLERS, 4);
    builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, 20);
    builder.RegisterService(&grpcservice);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout<<"gRPC server listen on: "<<server_addr<<std::endl;
    
    // 
    server->Wait();
    return;
}

// void AddRequestInThread(std::mutex *mutex, std::condition_variable *condition, int index, ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, float lambda, int *current_count)
// {
//     static std::atomic<int> flag=0;
//     PossionRandom possionRandom(index * 100+100);
//     UniformRandom uniformRandom(index * 10+10);
//     std::pair<std::string, ModelInputCreator> *creator = &(*inputCreators)[index];
//     std::vector<std::pair<std::string, DatasType>> datas(20);
//     for (int i = 0; i < 20; i++)
//     {
//         datas[i] = std::pair<std::string, DatasType>(creator->first, creator->second.CreateInput());
//     }

//     flag++;

//     while(flag<inputCreators->size())
//     {
//         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }
//     std::cout<<"start to send request for "<<creator->first<<"."<<std::endl;

//     int i = 0;
//     int tmp=0;
//     while (true)
//     {
//         std::this_thread::sleep_for(std::chrono::milliseconds((int)possionRandom.Random(lambda)));
//         std::unique_lock<std::mutex> lock(*mutex);

//         if (*current_count >= count)
//         {
//             // to be end.
//             break;
//         }

//         tmp = *current_count;
//         (*current_count)++;
//         lock.unlock();
//         condition->notify_all();
//             executorManager->AddTask(datas[i%20].first, datas[i%20].second, std::to_string(tmp+1));
//         i++;
//     }
//     std::cout<<"end send "<<creator->first<<", total: "<<i<<std::endl;
// }

// void ReqestGenerate(ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, std::vector<int> lambdas)
// {
//     std::cout << "wait system to init..." << std::endl;
//     std::this_thread::sleep_for(std::chrono::seconds(3));

//     std::cout << "run request generate with possion, and start to prepare random input." << std::endl;
    
//     std::condition_variable condition;
//     std::vector<std::shared_ptr<std::thread>> threads;
//     int current_count=0;
//     std::mutex mutex;
//     for(int i=0;i<inputCreators->size();i++)
//     {
//         if(lambdas[i]<=0)
//         {
//             continue;
//         }
//         threads.push_back(std::make_shared<std::thread>(AddRequestInThread,&mutex,&condition,i,executorManager,inputCreators,count,lambdas[i],&current_count));
//     }

//     for(auto thread: threads)
//     {
//         thread->join();
//     }

//     std::cout << "end send request." << std::endl;
// }

// std::filesystem::path SavePath(int count, std::vector<int> lambdas, std::vector<std::string> model_names)
// {
//     std::string fold="";
//     if(lambdas.size()>0)
//     {
//         fold=SaveHashFold(count,lambdas,model_names);
//     }

// #ifdef PARALLER_MODE
//     std::filesystem::path saveFold = RootPathManager::GetRunRootFold() / "data" / "PARALLEL";
// #elif OYST_MODE
//     std::filesystem::path saveFold = RootPathManager::GetRunRootFold() / "data" / "OYST";
// #elif BNST_MODE
//     std::filesystem::path saveFold = RootPathManager::GetRunRootFold() / "data" / "BNST";
// #elif FIFO_MODE
//     std::filesystem::path saveFold = RootPathManager::GetRunRootFold() / "data" / "FIFO";
// #else
//     std::filesystem::path saveFold = RootPathManager::GetRunRootFold() / "data" / "DLIR";
// #endif

//     if(fold.size()>0)
//     {
//         saveFold/=fold;
//     }

//     return saveFold;
// }

// void ReplyGather(ExecutorManager *executorManager, int count, std::vector<int> lambdas, std::vector<std::string> model_names)
// {
//     std::cout << "run reply gather." << std::endl;
    
//     nlohmann::json catalogue;
//     if(std::filesystem::exists(RootPathManager::GetRunRootFold() / "data"/ "catalogue.json"))
//     {
//         catalogue=JsonSerializer::LoadJson(RootPathManager::GetRunRootFold() / "data"/ "catalogue.json");
//     }

//     std::string fold="";
//     if(lambdas.size()>0)
//     {
//         fold=SaveHashFold(count,lambdas,model_names);
//     }
//     std::filesystem::path saveFold=SavePath(count,lambdas,model_names);

//     nlohmann::json record;
//     record["count"]=count;
//     for(int i=0;i<lambdas.size();i++)
//     {
//         record[model_names[i]]=lambdas[i];
//     }
//     catalogue[fold]=record;

//     JsonSerializer::StoreJson(catalogue,RootPathManager::GetRunRootFold() / "data"/ "catalogue.json",true);
    
//     std::filesystem::remove_all(saveFold);
//     std::filesystem::create_directories(saveFold);
//     auto &applyQueue = executorManager->GetApplyQueue();
//     for (int i = 0; i < count; i++)
//     {
//         auto task = applyQueue.Pop();
//         JsonSerializer::StoreJson(task->GetDescribe(), saveFold / (std::to_string(i) + ".json"));
//     }
//     std::cout << "all apply for " << count << " task received." << std::endl;
// }

// std::string SaveHashFold(int count, std::vector<int> lambdas, std::vector<std::string> model_names)
// {
//     std::string key=std::string("count=")+std::to_string(count);
//     for(int i=0;i<lambdas.size();i++)
//     {
//         key+=",";
//         key+=model_names[i];
//         key+="=";
//         key+=std::to_string(lambdas[i]);
//     }

//     std::hash<std::string> szHash;
// 	size_t hashVal = szHash(key);
    
//     return std::to_string(hashVal);
// }

// bool CheckReady(int count, std::vector<int> lambdas, std::vector<std::string> model_names)
// {
//     if(!std::filesystem::exists(RootPathManager::GetRunRootFold() / "data"/ "catalogue.json"))
//     {
//         return false;
//     }

//     nlohmann::json catalogue=JsonSerializer::LoadJson(RootPathManager::GetRunRootFold() / "data"/ "catalogue.json");
//     std::string foldName=SaveHashFold(count, lambdas,model_names);
//     if(!catalogue.contains(foldName))
//     {
//         return false;
//     }

//     if(!std::filesystem::exists(SavePath(count,lambdas,model_names)/(std::to_string(count-1)+".json")))
//     {
//         return false;
//     }

//     return true;
// }
