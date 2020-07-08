#include "Variable.h"

int count_function = 0;
int count_variable = 0;

map<Variable *, bool> variable_pool;

Variable *variable_construct(int rows, int cols) {
    count_variable++;

    for(auto itr = variable_pool.begin(); itr != variable_pool.end(); ++itr) {
        if (!itr -> second) {
            Variable *v = (Variable *)itr -> first;
            if (v -> data.rows == rows && v -> data.cols == cols) {
                v -> zeros();
                v -> creator = NULL;
                variable_pool[v] = true;

                return v;
            }
        }
    }
    Variable *r = new Variable(rows, cols);
    variable_pool[r] = true;

    return r;
}

void variable_destroy(Variable *ptr){
    count_variable--;

    variable_pool[ptr] = false;
    if (variable_pool.size() > 4000) {
        variable_pool.erase(ptr);
        delete ptr;
    }
}

int global_Variable_ID = 0;

Variable::Variable() {
    id = global_Variable_ID;
    global_Variable_ID;
}

Variable::Variable(const Variable &a) {
    id = global_Variable_ID;
    global_Variable_ID++;

    data = a.data;
    grad = a.grad;
    data_sparse = a.data_sparse;
    seed = a.seed;
    creator = a.creator;
    this -> isGetGrad = a.isGetGrad;
    this -> is_sparse = a.is_sparse;

}

Variable::Variable(int rows, int cols) {
    id = global_Variable_ID;
    global_Variable_ID++;

    data = cudaMat(rows, cols);
    grad = cudaMat(rows, cols);
    seed = cudaMat(grad.rows, grad.cols);
    seed.ones();
    creator = NULL;
}

Variable::Variable(int rows, int cols, bool is_get_grad) {
    this -> isGetGrad = is_get_grad;

    id = global_Variable_ID;
    global_Variable_ID++;

    data = cudaMat(rows, cols);
    grad = cudaMat(rows, cols);
    seed = cudaMat(grad.rows, grad.cols);
    seed.ones();
    creator = NULL;
}

Variable::Variable(cudaMat &input) {
    id = global_Variable_ID;
    global_Variable_ID++;
    data = input;
    grad = cudaMat(input.rows, input.cols);
    seed = cudaMat(grad.rows, grad.cols);
    seed.ones();
    creator = NULL;
}

Variable::Variable(Function *f, int rows, int cols) {
    id = global_Variable_ID;
    global_Variable_ID++;

    data = cudaMat(rows, cols);
    grad = cudaMat(rows, cols);
    seed = cudaMat(grad.rows, grad.cols);
    seed.ones();
    creator = f;
}

Variable::Variable(Function *f, cudaMat &input) {
    id = global_Variable_ID;
    global_Variable_ID++;

    data = input;
    grad = cudaMat(input.rows, input.cols);
    seed = cudaMat(grad.rows, grad.cols);
    seed.ones();
    creator = f;
}

Variable::Variable(vector<float> &ids, int nums) {
    id = global_Variable_ID;
    global_Variable_ID++;

    data_sparse = cudaMatSparse(ids, nums);
    grad = cudaMat(data_sparse.rows, data_sparse.cols);
    seed = cudaMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;

    this -> isGetGrad = false;
    this -> is_sparse = true;
}

Variable::~Variable() {}

Variable &Variable::operator=(const Variable &a) {
    id = global_Variable_ID;
    global_Variable_ID++;

    data = a.data;
    grad = a.grad;
    seed = a.seed;
    creator = a.creator;

    this -> isGetGrad = a.isGetGrad;
    this -> is_sparse = a.is_sparse;

    return *this;
}

void Variable::zeros() {
    data.mul(0, data);
    grad.mul(0, grad);
    forward_count = 0;
    last_opt = NULL;
    is_last_backward = NULL;
    this -> creator = NULL;
}

void Variable::creatorSet(Function *f) {
    this -> creator = f;
}

void Variable::backward() {
    this -> grad = seed;
    this -> backward(this);
}

void Variable::backward(Variable *v) {
    if (v == NULL) return;

    if (v -> creator != NULL) {
        if (v -> last_opt != NULL && v -> opt == *v -> last_opt) {
            *v -> is_last_backward = true;
        }

        if (v -> forward_count > 0) v -> forward_count--;

        if (v -> is_last_backward != NULL && *v -> is_last_backward == false) return;

        if (v -> forward_count != 0) return;

        v -> creator -> backward(v -> grad);

        for (int i = 0; i < v -> creator -> inputs.size(); i++) {
            PVariable nv = v -> creator -> inputs[i];

            if (nv -> isGetGrad) this -> backward(nv.get());
        }

    }
}

void Variable::zero_grads() {
    this -> zero_grads(this);
}

void Variable::zero_grads(Variable *v) {
    if (v == NULL) return;

    v -> grad.mul(0, v-> grad);
    v -> forward_count = 0;

    if (v -> creator != NULL) {
        for (int i = 0; i < v -> creator -> inputs.size(); i++) {
            PVariable nv = v -> creator -> inputs[i];
            this -> zero_grads(nv.get());
        }
    }
}


