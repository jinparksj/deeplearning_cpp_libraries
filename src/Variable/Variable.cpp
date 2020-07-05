#include "Variable.h"

int count_function = 0;
int count_variable = 0;

map<Variable *, bool> variable_pool;

Variable *variable_construct(int rows, int cols) {
    count_variable++;

    for(auto itr = variable_pool.begin(); itr != variable_pool.end(); ++itr) {
        if (!itr -> second) {
            Variable *v = (Variable *)itr -> first;

        }
    }
}