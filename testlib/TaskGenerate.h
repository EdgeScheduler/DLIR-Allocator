#ifndef __TASKGENERATE_H__
#define __TASKGENERATE_H__

#include <iostream>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <string>
#include <condition_variable>
#include <atomic>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "GPUAllocator/ExecutorManager.h"
#include "Tensor/TensorValue.hpp"
#include "Common/JsonSerializer.h"
#include "Tensor/ModelInputCreator.h"
#include "Common/PathManager.h"
#include "Random/UniformRandom.h"
#include "Random/PossionRandom.h"

void ReqestGenerate(ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, std::vector<int> lambdas);
void ReplyGather(ExecutorManager *executorManager, int count, std::vector<int> lambdas, std::vector<std::string> model_names);
std::string SaveHashFold(int count, std::vector<int> lambdas, std::vector<std::string> model_names);
std::filesystem::path SavePath(int count, std::vector<int> lambdas, std::vector<std::string> model_names);
bool CheckReady(int count, std::vector<int> lambdas, std::vector<std::string> model_names);

#endif // __TASKGENERATE_H__