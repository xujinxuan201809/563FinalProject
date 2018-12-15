#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <math.h>
#include <time.h>
#include <unistd.h>

using namespace std;

__global__ void kMartixByMatrixElementwise(const int nThreads, const float *m1, const float *m2, float *output) {
    /*  Computes the product of two arrays (elementwise multiplication).
     Inputs:
     m1: array
     m2: array
     output: array,the results of the multiplication are to be stored here
     */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        output[i] = m1[i] * m2[i];
    }
}

__device__ float* dMartixByMatrixElementwise(const float *m1, const float *m2, float *output, const int width, const int height) {
    
    kMartixByMatrixElementwise << < width, height >> > (width * height, m1, m2, output);
    cudaDeviceSynchronize();
    return output;
}

__global__ void kMartixSubstractMatrix(const int nThreads, const float *m1, const float *m2, float *output) {
    /*  Computes the (elementwise) difference between two arrays
     Inputs:
     m1: array
     m2: array
     output: array,the results of the computation are to be stored here
     */
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        output[i] = m1[i] - m2[i];
    }
}

__device__ float* dMartixSubstractMatrix(const float *m1, const float *m2, float *output, const int width, const int height) {
    
    kMartixSubstractMatrix << < width, height >> > (width * height, m1, m2, output);
    cudaDeviceSynchronize();
    return output;
}


__global__ void kSoftMaxCrossEntropy(const int nThreads, float *output, int oX, float* y) {  //oY is index, oX is column
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nThreads) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < 1e-10) {
            sum = 1e-10;
        }

        // Softmax = exp(value) / sum(exp(allValues))
        // Subtract truth (which is one hot)
        for (int i = 0; i < oX; i++) {
            y[row*oX + i] = (exp(output[row*oX + i]) / sum);
        }
    }
}


__device__ float* dSoftMaxCrossEntropy(const int height, float *output, int oX, float* y){
    kSoftMaxCrossEntropy << < height, oX >> > (oX*height, output, oX, y);
    cudaDeviceSynchronize();
    return output;
}




__global__ void kSigmoid(const int nThreads, float const *input, float *output) {
    /*  Computes the value of the sigmoid function f(x) = 1/(1 + e^-x).
     Inputs:
     input: array
     output: array, the results of the computation are to be stored here
     */
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        output[i] = 1.0 / (1.0 + std::exp(-input[i]));
    }
}

__device__ void dSigmoid(float const *input, float *output, const int height, const int width) {
    
    kSigmoid << < height, width >> > (height * width, input, output);
    cudaDeviceSynchronize();
}

__global__ void kSigmoid_d(const int nThreads, float const *input, float *output) {
    /*  Computes the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
     where f(x) is sigmoid function.
     Inputs:
     input: array
     output: array, the results of the computation are to be stored here:
     x(1 - x) for every element of the input matrix m1.
     */
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        output[i] = input[i] * (1 - input[i]);
    }
}

__device__ float* dSigmoid_d(float const *input, float *output, const int rows, const int columns) {
    kSigmoid_d << < rows, columns >> > (rows*columns, input, output);
    cudaDeviceSynchronize();
    return output;
}

__global__ void kDot(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows, const int m1_columns, const int m2_columns) {
    /*  Computes the product of two matrices: m1 x m2.
     Inputs:
     m1: array, left matrix of size m1_rows x m1_columns
     m2: array, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
     must be equal to the number of the columns in the left one)
     output: array, the results of the computation are to be stored here:
     m1 * m2, product of two arrays m1 and m2, a matrix of size m1_rows x m2_columns
     m1_rows: int, number of rows in the left matrix m1
     m1_columns: int, number of columns in the left matrix m1
     m2_columns: int, number of columns in the right matrix m2
     */
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        int r = (int)i / m2_columns;
        int c = i % m2_columns;
        float t_output = 0.f;
        
        for (int k = 0; k < m1_columns; ++k) {
            t_output += m1[r * m1_columns + k] * m2[k * m2_columns + c];
        }
        
        output[i] = t_output;
    }
}

__device__ float* dDot(const float *m1, const float *m2, float *output, const int m1_rows, const int m1_columns, const int m2_columns) {
    
    kDot << < m1_rows, m2_columns >> > (m1_rows * m2_columns, m1, m2, output, m1_rows, m1_columns, m2_columns);
    cudaDeviceSynchronize();
    return output;
}

__global__ void kDot_m1_m2T(const int nThreads, const float *m1, const float *m2, float *output, const int m1_columns, const int m2_rows) {
    /*  Updates the output matrix with the product of two matrices: m1 and m2 transposed.
     Inputs:
     m1: array, left matrix of size m1_rows x m1_columns
     m2: array, right matrix of size m2_rows x m1_columns (m2 transposed will be of size m1_columns x m2_rows)
     output: array, the results of the computation are to be stored here:
     m1 * m2, product of two arrays m1 and m2, a matrix of size m1_rows x m2_rows
     m1_columns: int, number of columns in the left matrix m1
     m2_rows: int, number of rows in the left matrix m2
     */
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        int r = (int)i / m2_rows;
        int c = i % m2_rows;
        float t_output = 0.0;
        int id_T;
        
        for (int k = 0; k < m1_columns; ++k) {
            id_T = c * m1_columns + k;
            t_output += m1[r * m1_columns + k] * m2[id_T];
        }
        
        output[i] = t_output;
    }
}

__device__ float* dDot_m1_m2T(const float *m1, const float *m2, float *output, const int m1_rows, const int m1_columns, const int m2_rows)
{
    kDot_m1_m2T << < m1_rows, m2_rows >> > (m1_rows * m2_rows, m1, m2, output, m1_columns, m2_rows);
    cudaDeviceSynchronize();
    return output;
}

__global__ void kDot_m1T_m2(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows,
                            const int m1_columns, const int m2_columns) {
    /*  Increments the output matrix with the product of two matrices: m1 transposed and m2.
     Inputs:
     m1: array, left matrix of size m1_rows x m1_columns (m1 transposed will be of size m1_columns x m1_rows)
     m2: array, right matrix of size m1_rows x m2_columns
     output: array, the results of the computation are to be stored here:
     m1 * m2, product of two arrays m1 and m2, a matrix of size m1_columns x m2_columns
     m1_rows: int, number of rows in the left matrix m1
     m1_columns: int, number of columns in the left matrix m1
     m2_rows: int, number of rows in the left matrix m2
     */
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nThreads;
         i += blockDim.x * gridDim.x)
    {
        int r = (int)i / m2_columns;
        int c = i % m2_columns;
        int id_T;
        float t_output = 0.0;
        
        for (int k = 0; k < m1_rows; ++k) {
            id_T = k * m1_columns + r;
            t_output += m1[id_T] * m2[k * m2_columns + c];
        }
        
        output[i] += t_output*0.02;
    }
}

__device__ void dDot_m1T_m2(const float *m1, const float *m2, float *output, const int m1_height, const int m1_width, const int m2_width)
{
    kDot_m1T_m2 << < m1_width, m2_width >> > (m1_width * m2_width, m1, m2, output, m1_height, m1_width, m2_width);
    cudaDeviceSynchronize();
}

__device__ void kPrintMatrix(const float* M, int h, int w) {
    /*  Prints out the input array as h x w matrix.
     Inputs:
     m: vector, matrix of size n_rows x n_columns
     h: int, number of rows in the matrix M
     w: int, number of columns in the matrix M
     */
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f  ", M[i*w + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void kFit(const float* X, const int X_w, const int X_h,
                     const float* y, const int y_w,
                     float* l1, const int l1_w, float* l_1_d,
                     
                     float* l2, const int l2_w, float* l_2_d,
                     
                     float* pred, float* pred_d,
                     float* W0,
                     float* W1,
                     
                     float* W2,
                     
                     float* buffer,
                     float* buffer_1,
                     float* buffer_2
                     )
{
    
        dSigmoid(dDot(X, W0, l1, X_h, X_w, l1_w), l1, X_h, l1_w);
        
        dSigmoid(dDot(l1, W2, l2, X_h, l1_w, l2_w), l2, X_h, l2_w);
        
        dSigmoid(dDot(l2, W1, pred, X_h, l2_w, y_w), pred, X_h, y_w);
  
        // dSoftMaxCrossEntropy(X_h, pred, y_w , pred);
        

        // dMartixByMatrixElementwise(dMartixSubstractMatrix(y, pred, pred_d, X_h, y_w), dSigmoid_d(pred, buffer, X_h, y_w), pred_d, y_w, X_h);
        
        // dMartixByMatrixElementwise(dDot_m1_m2T(pred_d, W1, l_2_d, X_h, y_w, l2_w), dSigmoid_d(l2, buffer_2, X_h, l2_w), l_2_d, l2_w, X_h);
        
        // dMartixByMatrixElementwise(dDot_m1_m2T(l_2_d, W2, l_1_d, X_h, l2_w, l1_w), dSigmoid_d(l1, buffer_1, X_h, l1_w), l_1_d, l1_w, X_h);

        // dDot_m1T_m2(l2, pred_d, W1, X_h, l2_w, y_w);
        
        // dDot_m1T_m2(l1, l_2_d,  W2, X_h, l1_w, l2_w);
        
        // dDot_m1T_m2(X, l_1_d, W0, X_h, X_w, l1_w);


}


vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}



int main(void) {
//    cout << "hello" << endl;
    float forward_time[165];
    float batch_time[165];
    float f_time,b_time;
    int batch_count=0;

    
    const int TRAINING_SIZE = 420;
    const int TRAINING_DIM = 784;
    const int L1_SIZE = 500;
    const int L2_SIZE = 500;
    
    int BATCH_SIZE = 256;
    
    float X_train[TRAINING_SIZE * TRAINING_DIM];
    float Y_train[TRAINING_SIZE * 10];
    

    // float h_X[BATCH_SIZE * TRAINING_DIM];
    // float* h_X = (float*)malloc(X_size_out_malloc); 
    // float* h_layer_2 = (float*)malloc(L2_size);
    // float h_y[BATCH_SIZE * 10];
    // float* h_y = (float*)malloc(y_size_out_malloc);
    const signed int y_size_out = BATCH_SIZE*10;
    const signed int X_size_out = BATCH_SIZE*TRAINING_DIM;

    string line;
    vector<string> line_v;
    
    cout << "Loading data ...\n";
    ifstream myfile("/home/ruixi/Downloads/train1.txt", ios::in);
    
    ofstream file;
    file.open("/home/ruixi/Downloads/output_forward_time_3layers_500.txt");
    
    if (myfile.is_open()) {
        int data_num = 0;
        while (getline(myfile, line)) {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(), 0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit)
                {
                    //                    y_train.push_back(1.);
                    Y_train[data_num * 10 + i] = 1;
                }
                //                else y_train.push_back(0.);
                else Y_train[data_num * 10 + i] = 0;
                
            }
            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                //                X_train.push_back(strtof((line_v[i]).c_str(),0));
                X_train[data_num * 784 + i - 1] = strtof((line_v[i]).c_str(), 0);
            }
            data_num++;
        }
        for(int j = 0;j< TRAINING_SIZE*TRAINING_DIM;j++){
        // X_train[j] = (X_train[j]-128)/128;
        X_train[j] = X_train[j]/255;
    }
        myfile.close();
    }
    else cout<<"Unable to open file"<<endl;
    
//    const signed int X_size = sizeof(X_train);
//    const signed int y_size = sizeof(Y_train);
    
    cout<<"Training the model"<<endl;
    
    
//    for(int i_batch = 0; i_batch<101; i_batch++){
        // Building batches of input variables (X) and labels (y)
    
    ////////////////////////////allocate the memory for Weights//////////////////////////////
    ////////////////////////////WEIGHTS_0
    //initialize the W0, the size of W0 = 784 * 128

    const long signed int X_size_out_malloc = X_size_out * sizeof(float);
    float* h_X = (float*)malloc(X_size_out_malloc); 
    for (int i=0; i < BATCH_SIZE*TRAINING_DIM; i++) {
        h_X[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
    }

    float *d_X;
    cudaMalloc(&d_X, X_size_out_malloc);
    // cudaMemcpy(d_X, h_X, X_size_out, cudaMemcpyHostToDevice);

    const long signed int W0_size = L1_SIZE * TRAINING_DIM * sizeof(float);
    float *h_W0 = (float*)malloc(W0_size);
    for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++) {
        h_W0[i] = 0.1 * (2.0*rand() / RAND_MAX - 1.0);
    }
    //W0 in cuda
    float *d_W0;
    int test_cudamalloc;
    // test_cudamalloc = cudaMalloc(&d_W0, W0_size);
    cudaMalloc(&d_W0, W0_size);
    cudaMemcpy(d_W0, h_W0, W0_size, cudaMemcpyHostToDevice);

/////////////////////test cudamalloc/////////////////////////////
    
    // cout<<test_cudamalloc;


    
    //LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
    const long signed int L1_size = L1_SIZE * BATCH_SIZE * sizeof(float);//128 * 256 layer 1 output size
    
    float* h_layer_1 = (float*)malloc(L1_size);
    float* h_layer_1_delta = (float*)malloc(L1_size);
    float* h_buffer_1 = (float*)malloc(L1_size);
    
    for (int i = 0; i < L1_SIZE*BATCH_SIZE; i++) {//initialize
        h_layer_1[i] = 0.0;
        h_buffer_1[i] = 0.0;
        h_layer_1_delta[i] = 0.0;
    }
    //copy to device
    float *d_layer_1;
    cudaMalloc(&d_layer_1, L1_size);
    cudaMemcpy(d_layer_1, h_layer_1, L1_size, cudaMemcpyHostToDevice);
    float *d_buffer_1;
    cudaMalloc(&d_buffer_1, L1_size);
    cudaMemcpy(d_buffer_1, h_buffer_1, L1_size, cudaMemcpyHostToDevice);
    float *d_layer_1_delta;
    cudaMalloc(&d_layer_1_delta, L1_size);
    cudaMemcpy(d_layer_1_delta, h_layer_1_delta, L1_size, cudaMemcpyHostToDevice);
    
    
    //WEIGHTS_2///////////////////////////
    //INITIALIZE THE W2, THE SIZE OF W2 = 128 * 64
    const long signed int W2_size = L2_SIZE * L1_SIZE * sizeof(float);//L1_SIZE = 128, W2_size = 64 * 128
    float *h_W2 = (float*)malloc(W2_size);
    for (int i = 0; i < L2_SIZE * L1_SIZE; i++) {
        h_W2[i] = 0.1 * (2.0*rand() / RAND_MAX - 1.0);
    }
    
    float *d_W2;
    cudaMalloc(&d_W2, W2_size);
    cudaMemcpy(d_W2, h_W2, W2_size, cudaMemcpyHostToDevice);
    //LAYER_2
    const long signed int L2_size = L2_SIZE * BATCH_SIZE * sizeof(float);//64 * 420 layer 2 output size
    
    float* h_layer_2 = (float*)malloc(L2_size);
    float* h_layer_2_delta = (float*)malloc(L2_size);
    float* h_buffer_2 = (float*)malloc(L2_size);
    
    for (int i = 0; i < L2_SIZE*BATCH_SIZE; i++) {//initialize
        h_layer_2[i] = 0.0;
        h_buffer_2[i] = 0.0;
        h_layer_2_delta[i] = 0.0;
    }
    float *d_layer_2;
    cudaMalloc(&d_layer_2, L2_size);
    cudaMemcpy(d_layer_2, h_layer_2, L2_size, cudaMemcpyHostToDevice);
    float *d_buffer_2;
    cudaMalloc(&d_buffer_2, L2_size);
    cudaMemcpy(d_buffer_2, h_buffer_2, L2_size, cudaMemcpyHostToDevice);
    float *d_layer_2_delta;
    cudaMalloc(&d_layer_2_delta, L2_size);
    cudaMemcpy(d_layer_2_delta, h_layer_2_delta, L2_size, cudaMemcpyHostToDevice);
    /////////////////////////////////////////////////
    
    /////////////////////////////////////////////////WEIGHTS_1
    const long signed int W1_size = L2_SIZE * 10 * sizeof(float);
    float *h_W1 = (float*)malloc(W1_size);
    for (int i = 0; i < L2_SIZE*10; i++) {
        h_W1[i] = 0.1* (2.0*rand() / RAND_MAX - 1.0);
    }
    
    float *d_W1;
    cudaMalloc(&d_W1, W1_size);
    cudaMemcpy(d_W1, h_W1, W1_size, cudaMemcpyHostToDevice);
    

    const long signed int y_size_out_malloc = y_size_out*sizeof(float);
    float* h_y = (float*)malloc(y_size_out_malloc);
    for (int i = 0; i < BATCH_SIZE*10;i++) {
        h_y[i] = 0.1* (2.0*rand() / RAND_MAX - 1.0);
    }


    float *d_y;
    cudaMalloc(&d_y, y_size_out_malloc);
    // cudaMemcpy(d_y, h_y, y_size_out, cudaMemcpyHostToDevice);
    
    float* h_pred = (float*)malloc(y_size_out_malloc);
    float* h_pred_delta = (float*)malloc(y_size_out_malloc);
    float* h_buffer = (float*)malloc(y_size_out_malloc); 

    for (int i = 0; i < BATCH_SIZE*10; i++) {
        h_pred[i] = 0.0;
        h_buffer[i] = 0.0;
        h_pred_delta[i] = 0.0;
    }
    float *d_buffer;
    cudaMalloc(&d_buffer, y_size_out_malloc);
    cudaMemcpy(d_buffer, h_buffer, y_size_out_malloc, cudaMemcpyHostToDevice);
    
    float *d_pred;
    cudaMalloc(&d_pred, y_size_out_malloc);
    cudaMemcpy(d_pred, h_pred, y_size_out_malloc, cudaMemcpyHostToDevice);
    
    float *d_pred_delta;
    cudaMalloc(&d_pred_delta, y_size_out_malloc);
    cudaMemcpy(d_pred_delta, h_pred_delta, y_size_out_malloc, cudaMemcpyHostToDevice);
    
    
    
    /////////////////////////////////////////////////BATCH LOOP///////////////////////////////////////////////////
    for(int i_batch = 0; i_batch<165; i_batch++){
        //timing the forward time and batch time
        // clock_t batch_start_forward;
        // clock_t batch_end_forward;
        // clock_t batch_start;
        // clock_t batch_end;
        
        // batch_start_forward = clock();
        // batch_start = clock();
        clock_t batch_start_forward;
        clock_t batch_end_forward;
        clock_t batch_start;
        clock_t batch_end;
        
        batch_start_forward = clock();
        batch_start = clock();

        int randindx = rand() % (420-BATCH_SIZE);//block index
    //    vector<float> b_X;
    //    vector<float> b_y;
        int h_X_index = 0;
        int h_y_index = 0;
        for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
    //        b_X.push_back(X_train[j]);
            h_X[h_X_index] = X_train[j];
            h_X_index++;
        }

        cudaMemcpy(d_X, h_X, X_size_out_malloc, cudaMemcpyHostToDevice);

        // for(int i=0;i<BATCH_SIZE*TRAINING_DIM;i++){
        //     cout<<h_X[i]<<" ";
        //     if(i%28==0){
        //         cout<<endl;
        //     }
        // }

        for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
    //        b_y.push_back(y_train[k]);
            h_y[h_y_index] = Y_train[k];
            h_y_index++;
        }

        cudaMemcpy(d_y, h_y, y_size_out_malloc, cudaMemcpyHostToDevice);


    const signed int X_size = BATCH_SIZE*TRAINING_DIM;
    const signed int y_size = BATCH_SIZE*10;


// clock_t batch_start_forward;
// clock_t batch_end_forward;
// clock_t batch_start;
// clock_t batch_end;

// batch_start_forward = clock();
// batch_start = clock();

        kFit << < 1, 1 >> > (d_X, TRAINING_DIM, BATCH_SIZE,
                             d_y, 10,
                             d_layer_1, L1_SIZE, d_layer_1_delta,
                             
                             d_layer_2, L2_SIZE, d_layer_2_delta,
                             
                             d_pred,
                             d_pred_delta,
                             d_W0,
                             d_W1,
                             
                             d_W2,
                             
                             d_buffer,
                             d_buffer_1,
                             d_buffer_2

                            );
        // batch_end_forward = clock();
        // batch_end = clock();

        // cout<<"Feed forward running time is: "<<f_time<<" ms"<<endl;
        // // cout<<"batch running time is :"<<b_time<<" ms"<<endl;

        // f_time = (double)(batch_end_forward-batch_start_forward)*1000.0/CLOCKS_PER_SEC;
        // // b_time = (double)(batch_end-batch_start)*1000.0/CLOCKS_PER_SEC;


        // if(i_batch<165){
        //     // forward_time[batch_count] = f_time;
        //     // batch_time[batch_count] = b_time;
        //     batch_count++;
        //     file<<f_time<<endl;
        //     // file<<b_time<<endl;
        // }

        cudaMemcpy(h_pred, d_pred, y_size_out_malloc, cudaMemcpyDeviceToHost);

        batch_end_forward = clock();
        batch_end = clock();

        cout<<"Feed forward running time is: "<<f_time<<" ms"<<endl;
        // cout<<"batch running time is :"<<b_time<<" ms"<<endl;

        f_time = (double)(batch_end_forward-batch_start_forward)*1000.0/CLOCKS_PER_SEC;
        // b_time = (double)(batch_end-batch_start)*1000.0/CLOCKS_PER_SEC;


        if(i_batch<165){
            // forward_time[batch_count] = f_time;
            // batch_time[batch_count] = b_time;
            batch_count++;
            file<<f_time<<endl;
            // file<<b_time<<endl;
        }
        

        if ((i_batch+1) % 100 == 0){
            cout << "-----------------------------------------------Epoch " << i_batch+1 << "--------------------------------------------------" <<"\n";
            cout << "Predictions:" << "\n";
//            print ( yhat, 10, 10 );
            for (int pre = 0; pre<100; pre++) {
                if (pre%10==0) {
                    cout<<endl;
                }
                cout<<h_pred[pre]<<" ";
            }
            cout <<endl<< "Ground truth:" << "\n";
//            print ( b_y, 10, 10 );
            for (int ground = 0; ground<100; ground++) {
                if (ground%10==0) {
                    cout<<endl;
                }
                cout<<h_y[ground]<<" ";
            }

            cout << "--------------------------------------------End of Epoch :(------------------------------------------------" <<"\n";
        };

        // cudaFree(d_X);

       

    }
    cudaFree(d_pred);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_layer_1_delta);
    cudaFree(d_pred_delta);
    cudaFree(d_W0);
    cudaFree(d_W1);
    cudaFree(d_buffer_1);
    cudaFree(d_buffer_2);
    cudaFree(d_buffer);
    
    cudaFree(d_layer_2_delta);
    cudaFree(d_W2);
    
    
    free(h_layer_1_delta);
    free(h_pred_delta);
    free(h_W0);
    free(h_W1);
    free(h_buffer_1);
    free(h_buffer_2);
    free(h_buffer);
    
    free(h_layer_2_delta);
    free(h_W2);
    
    free(h_pred);
    cout<<"HAPPY ENDING"<<endl;
}
