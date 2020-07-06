#ifndef LIB_CPPDL_CUDAMATSPARSE_H
#define LIB_CPPDL_CUDAMATSPARSE_H

#include <iostream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>

#include "cudaMat.h"

class cudaMatSparse {
public:
    int rows = 0;
    int cols = 0;

    cusparseHandle_t cuHandle;
    cusparseMatDescr_t descr;

    float *csrVal = NULL;
    int *csrRowPtr = NULL;
    int *csrColInd = NULL;

    float *csrValDevice = NULL;
    int *csrRowPtrDevice = NULL;
    int *csrColIndDevice = NULL;

    int numVals = 0;

    cudaMat rt, bt;

public:
    cudaMatSparse() {
        cusparseCreate(&cuHandle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }

    cudaMatSparse(vector<float> &ids, int col_nums) : cudaMatSparse() {
        embed(ids, col_nums);
    }

    //column-major as FORTRAN and cuBLAS usages
    void embed(vector<float> &ids, int col_nums) {
        rows = ids.size();
        cols = col_nums;

        int num_vals = rows;
        numVals = num_vals;

        csrVal = (float *) malloc(num_vals * sizeof(*csrVal));
        csrRowPtr = (int *) malloc((rows + 1) * sizeof(*csrRowPtr)); // 1-based indexing
        csrColInd = (int *) malloc(num_vals * sizeof(*csrColInd));

        cudaError_t error = cudaMalloc((void**) &csrValDevice, num_vals * sizeof(*csrValDevice));
        error = cudaMalloc((void**) &csrRowPtrDevice)
    }

};

#endif //LIB_CPPDL_CUDAMATSPARSE_H
