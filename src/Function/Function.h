#ifndef LIB_CPPDL_FUNCTION_H
#define LIB_CPPDL_FUNCTION_H

#include <list>
#include <random>
#include <vector>
#include <map>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include "../Variable/Variable.h"
#include "../cudaMat/cudaMat.h"

using namespace std;

extern map<Variable *, bool> obj_pool2;
extern int count_function;
extern int count_variable;

class Function {
public:
    vector<PVariable> inputs;
    vector<PVariable> outputs;

    int id = -1;
    string name;
    string custom_name;
    int inner_count = 0;

public:
    Function();
    virtual ~Function();

    virtual PVariable forward(PVariable input);
    virtual PVariable forward(PVariable x, PVariable t);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3, PVariable input4);
    virtual PVariable forward(PVariable input1, PVariable input2, PVariable input3, PVariable input4,
                              PVariable input5, PVariable input6, PVariable input7, PVariable input8,
                              PVariable input9, PVariable input10, PVariable input11, PVariable input12
    );
    virtual void backward(cudaMat &p_grad);
    virtual PVariable forward(vector<PVariable> &inputs, vector<PVariable> &outputs);
    virtual void backward(cudaMat &p_grad, vector<PVariable> &inputs, vector<PVariable> &outputs);

    void init();
    void clip_grad(Variable *v);
    virtual void reset_state();

private:
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive &ar, const unsigned int version) {}

};




#endif //LIB_CPPDL_FUNCTION_H
