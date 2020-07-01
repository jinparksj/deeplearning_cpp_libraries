#include "single_perceptron.h"

SLP::SLP(float learning_rate, float **input_array, float target_array[], float weight_array[], const int data_size, const int input_size, int epoch) {
    _data_size = data_size;
    _input_size = input_size;
    _learning_rate = learning_rate;
    _epoch = epoch;
    _input = new float*[_data_size];
    _target = new float[_data_size];
    _weight = new float[_input_size];

    for (int i = 0; i < _data_size; i++) {
        _input[i] = new float[_input_size];
        _target[i] = target_array[i];
        for (int j = 0; j < _input_size; j++) {
            _input[i][j] = input_array[i][j];
        }
    }

    for (int i = 0; i < _input_size; i++) {
        _weight[i] = weight_array[i];
    }
}

float SLP::DotArray(float input_array[], float weight_array[], int array_size) {
    float u_net = 0;
    for (int i = 0; i < array_size; i++) {
        u_net += input_array[i] * weight_array[i];
    }
    return u_net;
}

int SLP::StepFxn(float u) {
    return u > 0 ? 1 : 0;
}

int SLP::Forward(float input_array[], float weight_array[], int array_size) {
    float u_net = DotArray(input_array, weight_array, array_size);
    return StepFxn(u_net);
}

void SLP::Train(float input_array[], float weight_array[], float target_value, float learning_rate, int array_size) {
    float output_z = (float) Forward(input_array, weight_array, array_size);
    cout << "output : " << output_z;
    for (int i = 0; i < array_size; i++) {
        weight_array[i] += (target_value - output_z) * input_array[i] * learning_rate;
    }
}

void SLP::SLPProcess() {
    for (int i = 0; i < _epoch; i++) {
        cout << "epoch " << i + 1 << " : ";
        for (int j = 0; j < _data_size; j++) {
            Train(_input[j], _weight, _target[j], _learning_rate, _input_size);
        }
        for (int j = 0; j < _input_size; j++) {
            cout << "weight - " << j + 1 << " : " << _weight[j] << " / ";
        }
        cout << endl;
    }

    for (int i = 0; i < _data_size; i++) {
        cout << Forward(_input[i], _weight, _input_size) << " ";
    }
    cout << endl;
}

void SLP::TerminateProcess() {
    for (int i = 0; i < _data_size; i++) {
        delete[] _input[i];
    }
    delete [] _input;
    delete [] _weight;
    delete [] _target;
}
