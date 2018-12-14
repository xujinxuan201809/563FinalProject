/*
 563FinalProject
  Created by Jinxuan Xu on 11/20/18.

  3 Layer Neural Network on CPU
  configuration with 500, 500 units
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <stdlib.h>
#include <time.h>

using namespace std;

void print ( const vector <float>& m, int n_rows, int n_columns ) {
    
    /*  "Couts" the input vector as n_rows x n_columns matrix.
     Inputs:
     m: vector, matrix of size n_rows x n_columns
     n_rows: int, number of rows in the left matrix m1
     n_columns: int, number of columns in the left matrix m1
     */
    
    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            cout << m[ i * n_columns + j ] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

int argmax ( const vector <float>& m ) {
    
    return distance(m.begin(), max_element(m.begin(), m.end()));
}

vector <float> relu(const vector <float>& z){
    int size = z.size();
    vector <float> output;
    for( int i = 0; i < size; ++i ) {
        if (z[i] < 0){
            output.push_back(0.0);
        }
        else output.push_back(z[i]);
    }
    return output;
}

vector <float> reluPrime (const vector <float>& z) {
    int size = z.size();
    vector <float> output;
    for( int i = 0; i < size; ++i ) {
        if (z[i] <= 0){
            output.push_back(0.0);
        }
        else output.push_back(1.0);
    }
    return output;
}

static vector<float> random_vector(const int size)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distribution(0.0, 0.05);
    static default_random_engine generator;
    
    vector<float> data(size);
    generate(data.begin(), data.end(), [&]() { return distribution(generator); });
    return data;
}

vector <float> softmax (const vector <float>& z, const int dim) {
    
    const int zsize = static_cast<int>(z.size());
    vector <float> out;
    
    for (unsigned i = 0; i != zsize; i += dim) {
        vector <float> foo;
        for (unsigned j = 0; j != dim; ++j) {
            foo.push_back(z[i + j]);
        }
        
        float max_foo = *max_element(foo.begin(), foo.end());
        
        for (unsigned j = 0; j != dim; ++j) {
            foo[j] = exp(foo[j] - max_foo);
        }
        
        float sum_of_elems = 0.0;
        for (unsigned j = 0; j != dim; ++j) {
            sum_of_elems = sum_of_elems + foo[j];
        }
        
        for (unsigned j = 0; j != dim; ++j) {
            out.push_back(foo[j]/sum_of_elems);
        }
    }
    return out;
}

vector <float> sigmoid_d (const vector <float>& m1) {
    
    /*  Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
     where f(x) is sigmoid function.
     Input: m1, a vector.
     Output: x(1 - x) for every element of the input matrix m1.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = m1[ i ] * (1 - m1[ i ]);
    }
    
    return output;
}

vector <float> sigmoid (const vector <float>& m1) {
    
    /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
     Input: m1, a vector.
     Output: 1/(1 + e^-x) for every element of the input matrix m1.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = 1 / (1 + exp(-m1[ i ]));
    }
    
    return output;
}

vector <float> operator+(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the elementwise sum of two vectors.
     Inputs:
     m1: a vector
     m2: a vector
     Output: a vector, sum of the vectors m1 and m2.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

vector <float> operator-(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the difference between two vectors.
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 - m2, difference between two vectors m1 and m2.
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

vector <float> operator*(const vector <float>& m1, const vector <float>& m2){
    
    /*  Returns the product of two vectors (elementwise multiplication).
     Inputs:
     m1: vector
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

vector <float> operator*(const float m1, const vector <float>& m2){
    
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1 * m2[i];
    };
    
    return product;
}

vector <float> operator/(const vector <float>& m2, const float m1){
    
    /*  Returns the product of a float and a vectors (elementwise multiplication).
     Inputs:
     m1: float
     m2: vector
     Output: vector, m1 * m2, product of two vectors m1 and m2
     */
    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m2[i] / m1;
    };
    
    return product;
}

vector <float> transpose (float *m, const int C, const int R) {
    
    /*  Returns a transpose matrix of input matrix.
     Inputs:
     m: vector, input matrix
     C: int, number of columns in the input matrix
     R: int, number of rows in the input matrix
     Output: vector, transpose matrix mT of input matrix m
     */
    
    vector <float> mT (C*R);
    
    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}

vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns) {
    
    vector <float> output (m1_rows*m2_columns);
    
    for( int row = 0; row != m1_rows; ++row ) {
        for( int col = 0; col != m2_columns; ++col ) {
            output[ row * m2_columns + col ] = 0.f;
            for( int k = 0; k != m1_columns; ++k ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }
    
    return output;
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

int main(int argc, const char * argv[]) {
    clock_t epoch_start, epoch_end;
    epoch_start = clock();

    double forward_time[165];
    double batch_time[165];
    
    string line;
    vector<string> line_v;
    
    cout << "Loading data ...\n";
    vector<float> X_train;
    vector<float> y_train;
    ifstream myfile ("/Users/jinxuanxu/Projects/563/FinalProject/563FinalProject/563FinalProject/train.txt", ios::in);
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(),0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit)
                {
                    y_train.push_back(1.);
                }
                else y_train.push_back(0.);
            }
            
            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                X_train.push_back(strtof((line_v[i]).c_str(),0));
            }
            
        }
        X_train = X_train/255.0;
        myfile.close();
    }
    
    else cout << "Unable to open file" << '\n';
    
    int xsize = static_cast<int>(X_train.size());
    int ysize = static_cast<int>(y_train.size());
    int batch_count = 0;
    ofstream file;
    file.open("/Users/jinxuanxu/Projects/563/FinalProject/563FinalProject/563FinalProject/time_CPU3Layers_500.txt");
    
    // Some hyperparameters for the NN
    int BATCH_SIZE = 256;
    float lr = .01/BATCH_SIZE;
    
    // Random initialization of the weights
    vector <float> W1 = random_vector(784*500);
    vector <float> W3 = random_vector(500*500);
    vector <float> W4 = random_vector(500*10);
    
    cout << "Training the model ...\n";
    
    for (unsigned i = 0; i < 165; ++i) {
        //start timeing the batch training
        clock_t batch_start_forward, batch_end_forward;
        batch_start_forward = clock();
        clock_t batch_start, batch_end;
        batch_start = clock();

        // Building batches of input variables (X) and labels (y)
        int randindx = rand() % (42000-BATCH_SIZE);//block index
        vector<float> b_X;
        vector<float> b_y;
        for (unsigned j = randindx*784; j < (randindx+BATCH_SIZE)*784; ++j){
            b_X.push_back(X_train[j]);
        }
        for (unsigned k = randindx*10; k < (randindx+BATCH_SIZE)*10; ++k){
            b_y.push_back(y_train[k]);
        }
        
        // Feed forward//a2 new, a2->a3
        vector<float> a1 = sigmoid(dot( b_X, W1, BATCH_SIZE, 784, 500 ));
        vector<float> a3 = sigmoid(dot( a1, W3, BATCH_SIZE, 500, 500 ));
        vector<float> yhat = softmax(dot( a3, W4, BATCH_SIZE, 500, 10 ), 10);
        
        batch_end_forward = clock();
        
        // Back propagation
        vector<float> dyhat = (yhat - b_y);
        // dW4 = a3.T * dyhat
        vector<float> dW4 = dot(transpose( &a3[0], BATCH_SIZE, 500 ), dyhat, 500, BATCH_SIZE, 10);
        // dz3 = dyhat * W4.T * relu'(a3)
        vector<float> dz3 = dot(dyhat, transpose( &W4[0], 500, 10 ), BATCH_SIZE, 10, 500) * sigmoid_d(a3);
        // dW3 = a1.T * dz3
        vector<float> dW3 = dot(transpose( &a1[0], BATCH_SIZE, 500 ), dz3, 500, BATCH_SIZE, 500);//a1->a2
        // dz2 = dz3 * W3.T * relu'(a1)
        vector<float> dz2 = dot(dz3, transpose( &W3[0], 500, 500 ), BATCH_SIZE, 500, 500) * sigmoid_d(a1);
        // dW1 = X.T * dz1
        vector<float> dW1 = dot(transpose( &b_X[0], BATCH_SIZE, 784 ), dz2, 784, BATCH_SIZE, 256);
        
        // Updating the parameters
        W4 = W4 - lr * dW4;
        W3 = W3 - lr * dW3;
        W1 = W1 - lr * dW1;
        
        
        if ((i+1) % 100 == 0){
            cout << "-----------------------------------------------Epoch " << i+1 << "--------------------------------------------------" <<"\n";
            cout << "Predictions:" << "\n";
            print ( yhat, 10, 10 );
            cout << "Ground truth:" << "\n";
            print ( b_y, 10, 10 );
            vector<float> loss_m = yhat - b_y;
            float loss = 0.0;
            for (unsigned k = 0; k < BATCH_SIZE*10; ++k){
                loss += loss_m[k]*loss_m[k];
            }

            cout << "                                            Loss " << loss/BATCH_SIZE <<"\n";
            cout << "--------------------------------------------End of Epoch :(------------------------------------------------" <<"\n";
        };
        batch_end = clock();
        
        double f_time = (double)(batch_end_forward-batch_start_forward)*1000.0/CLOCKS_PER_SEC;
        double b_time = (double)(batch_end-batch_start)*1000.0/CLOCKS_PER_SEC;
        
        
        cout<<"Feed forward running time is: "<<f_time<<" ms"<<endl;
        cout<<"batch running time is :"<<b_time<<" ms"<<endl;
        
        if(i<165){
            forward_time[batch_count] = f_time;
            batch_time[batch_count] = b_time;
            batch_count++;
            file<<f_time<<" , "<<b_time<<endl;
        }
    };
    epoch_end = clock();
    cout <<"Epoch Time is :"<<(double)(epoch_end-epoch_start)*1000.0/CLOCKS_PER_SEC<<endl;
    return 0;
}
