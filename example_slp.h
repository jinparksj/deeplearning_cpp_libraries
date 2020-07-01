#ifndef LIB_CPPDL_EXAMPLE_SLP_H
#define LIB_CPPDL_EXAMPLE_SLP_H

#include "src/slp/single_perceptron.h"

void ExampleSLP() {
    const int data_nums = 4;
    const int input_nums = 3;

    float x[data_nums][input_nums] = {{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    float t[data_nums] = {0, 0, 0, 1};
    float w[input_nums] = {-0.5, 0, 0};
    float e = 0.1;
    int epoch = 10;

    float **input_x;
    input_x = new float*[data_nums];
    for (int i = 0; i < data_nums; i++) {
        input_x[i] = new float[input_nums];
        for (int j = 0; j < input_nums; j++) {
            input_x[i][j] = x[i][j];
        }
    }

    SLP slp(e, input_x, t, w, data_nums, input_nums, epoch);
    slp.SLPProcess();
    slp.TerminateProcess();

    for (int i = 0; i < data_nums; i++) {
        delete[] input_x[i];
    }
    delete [] input_x;
}

#endif //LIB_CPPDL_EXAMPLE_SLP_H
