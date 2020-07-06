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
        error = cudaMalloc((void**) &csrRowPtrDevice, (rows + 1) * sizeof(*csrRowPtrDevice));
        error = cudaMalloc((void**) &csrColIndDevice, num_vals * sizeof(*csrColIndDevice));

        memset(csrRowPtr, 0x00, (rows + 1) * sizeof(*csrRowPtr));
        csrRowPtr[0] = 0;
        for (int i = 0; i < rows; i++) {
            csrVal[i] = 1.;
            csrColInd[i] = ids[i];
            csrRowPtr[i + 1] = csrRowPtr[i] + 1;
        }

//        //to check values
//        cout << "csrVal : ";
//        for (int i = 0; i < num_vals; i++) {
//            cout << csrVal[i] << " ";
//        }
//        cout << endl;
//
//        cout << "csrRowPtr : ";
//        for (int i = 0; i < rows + 1; i++) {
//            cout << csrRowPtr[i] << " ";
//        }
//        cout << endl;
//
//        cout << "csrColInd : ";
//        for (int i = 0; i < num_vals; i++) {
//            cout << csrColInd[i] << " ";
//        }
//        cout << endl;

        memSetHost(csrVal, csrRowPtr, csrColInd);
    }

};

#endif //LIB_CPPDL_CUDAMATSPARSE_H
