#ifndef LIB_CPPDL_VARIABLE_H
#define LIB_CPPDL_VARIABLE_H

#include <list>
#include <random>
#include <boost/intrusive_ptr.hpp>
#include <iostream>
#include <chrono>
#include <memory>

#include "../Function/Function.h"
#include "../cudaMat/cudaMat.h"
#include "../cudaMat/cudaMatSparse.h"

using namespace std;
class Function;

class Variable {
private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version) {
        ar &id;
        ar &data;
        ar &grad;
        ar &seed;
        ar &isGetGrad;
    }

public:
    int id = 0;
    int opt = 0;
    int *last_opt = NULL;
    bool *is_last_backward = NULL;
    int forward_count = 0;
    Function *creator = NULL;           //Function creating this Variable instance itself
    string name;

    cudaMat data;                       //Result data saved at Forward
    cudaMatSparse data_sparse;
    cudaMat grad;                       //Gradient value for back propagation
    cudaMat seed;

    int grad_num = -999;
    bool isGetGrad = true;
    bool is_sparse = false;

public:
    Variable();
    Variable(const Variable &a);
    Variable(int rows, int cols);
    Variable(int rows, int cols, bool is_get_grad);
    Variable(Function *f, int rows, int cols);
    Variable(cudaMat &input);
    Variable(Function *f, cudaMat &input);
    Variable(vector<float> &ids, int nums);
    ~Variable();

    void creatorSet(Function *f);
    Variable &operator=(const Variable &a);
    Variable sin();
    Variable log();
    void backward();
    void backward(Variable *v);
    void zero_grads();
    void zero_grads(Variable *v);
    void ones();
    void zeros();
    void unchain();
    void zero_grad();
    void randoms(float m, float a);
    void binominal_randoms(float ratio);
    float val();

};

using PVariable = shared_ptr<Variable>;

Variable *variable_construct(int rows, int cols);
void variable_destroy(Variable *ptr);

#endif //LIB_CPPDL_VARIABLE_H
