#ifndef __TASKGENERATE_H__
#define __TASKGENERATE_H__

#include <iostream>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../include/GPUAllocator/ExecutorManager.h"
#include "../../include/Tensor/TensorValue.hpp"
#include "../../include/Common/JsonSerializer.h"
#include "../../include/Tensor/ModelInputCreator.h"
#include "../../include/Common/PathManager.h"
#include "../../include/Random/UniformRandom.h"
#include "../../include/Random/PossionRandom.h"

void ReqestGenerate(ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, float lambda = 30);
void ReplyGather(ExecutorManager *executorManager, int count);

#endif // __TASKGENERATE_H__