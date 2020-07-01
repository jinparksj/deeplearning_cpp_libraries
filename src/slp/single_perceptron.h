#ifndef LIB_CPPDL_SINGLE_PERCEPTRON_H
#define LIB_CPPDL_SINGLE_PERCEPTRON_H

#include <iostream>
using namespace std;

class SLP {
public:
    SLP(float learning_rate, float **input_array, float *target, float *weight, const int data_size, const int input_size, int epoch);
    float DotArray(float input_array[], float weight_array[], int array_size);
    int StepFxn(float u);
    int Forward(float input_array[], float weight_array[], int array_size);
    void Train(float input_array[], float weight_array[], float target_value, float learning_rate, int array_size);
    void SLPProcess();
    void TerminateProcess();

private:
    float **_input = NULL;
    float *_target = NULL;
    float *_weight = NULL;
    float _learning_rate;
    int _epoch;
    int _input_size;
    int _data_size;
};

#endif //LIB_CPPDL_SINGLE_PERCEPTRON_H
