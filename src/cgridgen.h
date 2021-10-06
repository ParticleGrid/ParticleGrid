#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "generate.h"

class PyConsumeInfo {
public:
    int n_molecules;
    int* molecule_sizes;
    float** molecules;
    OutputSpec* outs;
    float* tensor_out;
};

class PyDataQueue {
public:
    bool done = 0;
  int amt_done = 0;
    int n_molecules;
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<PyConsumeInfo> consume_queue;
};

void datagen_consumer_cuda_py(PyDataQueue* data_ptr);
