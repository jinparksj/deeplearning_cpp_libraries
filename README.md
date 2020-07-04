# Deep Learning C++ Libraries

- **07/01/2020 - Update Single Layer Perceptron**


- **CUDA Environement Comments**
    
    - Command environments
    
    -ldlib -L/usr/local/cuda-9.0/lib64 -lcudnn -lpthread -lcuda -lcudart -lcublas -lcurand -lcusolver -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_dnn -lcudart -lgomp -lm -lstdc++ 

    - CMakeLists environments
    
    -#include_directories(/usr/local/cuda-9.0/targets/x86_64-linux/include)
    -#include_directories(/usr/local/cuda/targets/x86_64-linux/include)
    -#include_directories(/usr/local/cuda-9.0/lib64)
    -#include_directories(/usr/local/cuda/extras/CUPTI/lib64)
    -#include_directories(/usr/local/cuda-9.0)
    -#find_package(CUDA REQUIRED)
    -#include_directories("${CUDA_INCLUDE_DIRS}")
    
    
- **Reference**
https://github.com/takezo5096
