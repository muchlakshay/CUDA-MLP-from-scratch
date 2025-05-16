#pragma once
#include <cmath>

namespace derivatives{
    __device__ float sigmoidDerivative(float sigmoidAct){
        return sigmoidAct * (1.0f - sigmoidAct);
    }
    __device__ float reluDerivative(float reluAct){
        return (reluAct > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void MatMul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // i
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // j

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void ComputeWeightsGradients(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K, float clip) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            float a = A[i * N + row]; 
            float b = B[i * K + col]; 
            sum += a * b;
        }
        C[ row * K + col ] = sum/static_cast<float>(M);
        C[ row * K + col ] = fminf(clip, fmaxf(C[ row * K + col ], -clip));
    }
}

__global__ void ComputeBiasGradients( float* errors, float* bias_gradients, int batch_size, int output_size,
    float clip, float* loss, int label_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= output_size) return;

    float sum = 0.0f;

    if (col==0) {
        *loss = (*loss) / (batch_size);
    }

    for (int row {}; row < batch_size; ++row) {
        sum += errors[row * output_size + col];

    }

    bias_gradients[col] = sum / static_cast<float>(batch_size);
    bias_gradients[col] = fminf(clip, fmaxf(bias_gradients[col], -clip));
}

__global__ void addBias(float* preActivations, float* bias, int output_size, int batch_size){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * output_size + col;

    if (col<output_size && row<batch_size){
        preActivations[ idx ] = preActivations[ idx ] + bias[ col ];
    }
}

__global__ void maxLogit( float* weighted_sum, float* maxArr, float* sum, int rows, int cols){
    int row = blockIdx.x;
    if (row<rows){
        float max {-INFINITY};
        float local_sum {};

        for (int i {}; i<cols; ++i){
            if (weighted_sum[ row * cols + i ]>max) max = weighted_sum[ row * cols + i ];
        }
        for (int i {}; i<cols; ++i){
            local_sum+=expf(weighted_sum[ row * cols + i ]-max);
        }
       sum[ row ] = local_sum;
       maxArr[ row ] = max;
    }
}

__global__ void addBiasAndActivation(
    float* preActivations,
    float* bias,
    float* activations,
    int batch_size,
    int output_size,
    int activation_id,
    float* maxLogit,
    float* sum){

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * output_size + col;

    if (col<output_size && row<batch_size){
        float eps = 1e-7f;
        if (activation_id != 3) preActivations[ idx ] = preActivations[ idx ] + bias[ col ];

        if (activation_id==0) activations[ idx ] = 1.0f / (1.0f + expf( -preActivations[ idx ] ));
        else if (activation_id==1) activations[ idx ] = fmaxf(0.0f, preActivations[ idx ]);
        else if (activation_id==2) activations[ idx ] = preActivations[ idx ];
        else if (activation_id==3) activations[ idx ] = expf( preActivations[ idx ] - maxLogit[ row ] ) / (sum[ row ]+eps);
    }
}

__global__ void ComputeOutputError(float* activations, float* Y_train,float* errors,
    float* preActivations, int activation_id, int batch_size, int output_size, float* loss, int labels_len, int loss_id){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * output_size + col;

    if (col<output_size && row<batch_size){

        if (loss_id==0){
            errors[ idx ] = 2*(activations[ idx ] - Y_train[ idx ]);
            atomicAdd(loss, powf((activations[idx] - Y_train[idx]), 2));
        }
        else if (loss_id==1){
            float eps { 1e-7f };
            float y_hat { fminf(fmaxf(activations[ idx ], eps), 1.0f-eps) };
            if (activation_id==3){
                errors[ idx ] = activations[ idx ] - Y_train[ idx ];
                atomicAdd(loss, - (Y_train[ idx ] * logf( y_hat )));
                return;
            }
            errors[ idx ] = -Y_train [ idx ] / y_hat;
            atomicAdd(loss, - (Y_train[ idx ] * logf( y_hat )));  
        }
        else if (loss_id==2){
            float eps { 1e-7f };
            float y_hat { fminf(fmaxf(activations[ idx ], eps), 1.0f-eps) };
            if (activation_id==0){
                errors[ idx ] = activations[ idx ] - Y_train[ idx ];
                atomicAdd(loss, - ( (Y_train[ idx ] * logf( y_hat ) ) + ( 1.0f - Y_train [ idx ]) * logf(1.0f - y_hat ) ));
                return;
            }
            errors[ idx ] = - (Y_train[ idx ] / y_hat) -( (1.0f - Y_train[ idx ]) / (1.0f - y_hat) );
            atomicAdd(loss, - ( (Y_train[ idx ] * logf( y_hat ) ) + ( 1.0f - Y_train [ idx ]) * logf(1.0f - y_hat) ) );
        }

        if (activation_id==0) errors[ idx ] *= derivatives::sigmoidDerivative(activations[ idx ]);
        else if (activation_id==1) errors[ idx ] *= derivatives::reluDerivative(preActivations[ idx ]);
        else if (activation_id==2) errors[ idx ] *= 1.0f;
    }
    
}

 __global__ void ActivationDerivative(float* partial_errors, float* preActivations, float* activations,
    int activation_id, int batch_size, int output_size){
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * output_size + col;

    if (col<output_size && row<batch_size){
        if (activation_id==0) partial_errors[ idx ] *= derivatives::sigmoidDerivative(activations[ idx ]);
        else if (activation_id==1) partial_errors[ idx ] *= derivatives::reluDerivative(preActivations[ idx ]);
        else if (activation_id==2) partial_errors[ idx ] *= 1.0f;
    }  
 }

__global__ void updateWeightsKernel(float* w, float* dw, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) w[idx] -= lr * dw[idx];
}

__global__ void updateBiasesKernel(float* b, float* db, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] -= lr * db[idx];
}


__global__ void printLoss(float* loss, int batch_size, int output_size){
    printf("Loss: %f\n\n", 1000*(*loss));
}

__global__ void debugPrint(
    float* array,
    int rows,
    int cols){

    int idx = threadIdx.x;
    for (int row {}; row<rows; ++row){
        for (int col {}; col<cols; ++col){
            printf("%.6f ",array[row*cols+col]);
        }
        printf("\n");
    }
    printf("\n");
}
