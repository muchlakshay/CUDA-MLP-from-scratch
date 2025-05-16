#pragma once

#include "kernels.cuh"
#include <iostream>
#include <Eigen/Dense>
#include "utility.cuh"

namespace nnData{
    struct LayersMetaData{
        int input_size;
        int output_size;
        float* d_weights;
        float* d_weights_transposed;
        float* d_bias;
        float* d_errors;
        float* d_weights_gradients;
        float* d_bias_gradients;
        float* d_activations;
        float* d_preActivations;
        float* d_maxLogits;
        float* d_sumLogits;
    };

    int numLayers;
    vector<LayersMetaData> layersMetaData;
    float* d_X_train;
    float* d_Y_train;
    float* loss;

}

void loadDataToGPU(const MatrixXd& X_train, const MatrixXd& Y_train, bool debug_print = false){

    auto alloc1 { cudaMalloc(&nnData::d_X_train, X_train.rows() * X_train.cols() * sizeof(float)) };
    utility::CUDA_CHECK(alloc1);
    auto alloc2 { cudaMalloc(&nnData::d_Y_train, Y_train.rows() * Y_train.cols() * sizeof(float)) };
    utility::CUDA_CHECK(alloc2);
    auto X_train_flat { utility::flatten(X_train) };
    auto Y_train_flat { utility::flatten(Y_train) };

    auto copy1 { cudaMemcpy(nnData::d_X_train, X_train_flat.data(), X_train.rows() * X_train.cols() * sizeof(float), cudaMemcpyHostToDevice) }; 
    utility::CUDA_CHECK(copy1);
    auto copy2 { cudaMemcpy(nnData::d_Y_train, Y_train_flat.data(), Y_train.rows() * Y_train.cols() * sizeof(float), cudaMemcpyHostToDevice) };
    utility::CUDA_CHECK(copy2);


    if (debug_print){
        cout<<"\nGPU X_train: \n";
        debugPrint<<<1,1>>>(nnData::d_X_train, X_train.rows(), X_train.cols());
        cudaDeviceSynchronize();
        cout<<"\nGPU Y_train: \n";
        debugPrint<<<1,1>>>(nnData::d_Y_train, Y_train.rows(), Y_train.cols());
        cudaDeviceSynchronize();
    }
}

void freeGPUDataMem(){
    cudaFree(nnData::d_X_train);
    cudaFree(nnData::d_Y_train);
}

void freeGPU(){
    for (int i {}; i<nnData::numLayers; ++i){
        cudaFree(nnData::layersMetaData[i].d_weights);
        cudaFree(nnData::layersMetaData[i].d_weights_transposed);
        cudaFree(nnData::layersMetaData[i].d_bias);
        cudaFree(nnData::layersMetaData[i].d_errors);
        cudaFree(nnData::layersMetaData[i].d_weights_gradients);
        cudaFree(nnData::layersMetaData[i].d_bias_gradients);
        cudaFree(nnData::layersMetaData[i].d_activations);
        cudaFree(nnData::layersMetaData[i].d_preActivations);
        cudaFree(nnData::layersMetaData[i].d_maxLogits);
        cudaFree(nnData::layersMetaData[i].d_sumLogits);
    }
    cudaFree(nnData::loss);
    // freeGPUDataMem();
}

void initLayersGPU(const vector<MatrixXd>& weights, const vector<VectorXd>& biases, int batch_size){

    nnData::numLayers = weights.size();

    for (int i {}; i<nnData::numLayers; ++i){

        auto weights_flat { utility::flatten(weights[i]) };
        auto weights_transposed_flat { utility::flatten( MatrixXd(weights[i].transpose()) )};
        auto biases_flat { utility::flatten(biases[i]) };

        nnData::LayersMetaData layer;

        layer.input_size = weights[i].cols();
        layer.output_size = weights[i].rows();

        auto alloc1 { cudaMalloc(&layer.d_weights, weights[i].rows() * weights[i].cols() * sizeof(float)) };
        utility::CUDA_CHECK(alloc1);
        auto alloc2 { cudaMalloc(&layer.d_weights_transposed, weights[i].rows() * weights[i].cols() * sizeof(float)) };
        utility::CUDA_CHECK(alloc2);
        auto aclloc3 { cudaMalloc(&layer.d_bias, biases[i].size() * sizeof(float)) };
        utility::CUDA_CHECK(aclloc3);
        auto alloc4 { cudaMalloc(&layer.d_errors, weights[i].rows() * batch_size * sizeof(float)) };
        utility::CUDA_CHECK(alloc4);
        auto alloc5 { cudaMalloc(&layer.d_weights_gradients, weights[i].rows() * weights[i].cols() * sizeof(float)) };
        utility::CUDA_CHECK(alloc5);
        auto alloc6 { cudaMalloc(&layer.d_bias_gradients, biases[i].size() * sizeof(float)) };
        utility::CUDA_CHECK(alloc6);
        auto alloc7 { cudaMalloc(&layer.d_activations, batch_size * weights[i].rows() * sizeof(float)) };
        utility::CUDA_CHECK(alloc7);
        auto alloc8 { cudaMalloc(&layer.d_preActivations, batch_size * weights[i].rows() * sizeof(float)) };
        utility::CUDA_CHECK(alloc8);
        auto alloc9 { cudaMalloc(&nnData::loss, sizeof(float)) };
        utility::CUDA_CHECK(alloc9);
        auto alloc10 { cudaMalloc(&layer.d_maxLogits, batch_size * sizeof(float)) };
        utility::CUDA_CHECK(alloc10);
        auto alloc11 { cudaMalloc(&layer.d_sumLogits, batch_size * sizeof(float)) };
        utility::CUDA_CHECK(alloc11);

        auto copy1 { cudaMemcpy(layer.d_weights, weights_flat.data(), weights[i].rows() * weights[i].cols() * sizeof(float), cudaMemcpyHostToDevice) };
        utility::CUDA_CHECK(copy1);
        auto copy2 { cudaMemcpy(layer.d_weights_transposed, weights_transposed_flat.data(), weights[i].rows() * weights[i].cols() * sizeof(float), cudaMemcpyHostToDevice) };
        utility::CUDA_CHECK(copy2);
        auto copy3 { cudaMemcpy(layer.d_bias, biases_flat.data(), biases[i].size() * sizeof(float), cudaMemcpyHostToDevice) };
        utility::CUDA_CHECK(copy3);
        nnData::layersMetaData.push_back(layer);

    }   
}

void printParams(){
    for (int i {}; i<nnData::numLayers; ++i){
        if (i==0) continue;
        cout<<"\nWeights GPU Layer "<<i+1<<": "<<"\n";
        debugPrint<<<1,1>>>(nnData::layersMetaData[i].d_weights, nnData::layersMetaData[i].output_size, nnData::layersMetaData[i].input_size);
        cudaDeviceSynchronize();
        cout<<"Weights Transposed GPU Layer "<<i+1<<": "<<"\n";
        debugPrint<<<1,1>>>(nnData::layersMetaData[i].d_weights_transposed, nnData::layersMetaData[i].input_size, nnData::layersMetaData[i].output_size);
        cudaDeviceSynchronize();
        cout<<"Bias GPU Layer "<<i+1<<": "<<"\n";
        debugPrint<<<1,1>>>(nnData::layersMetaData[i].d_bias, 1, nnData::layersMetaData[i].output_size);
        cudaDeviceSynchronize();

    }
}

void forwardPassGPU(float* X_train, int batch_size,
    const vector<string>& activation_fn, bool debug_print = false){

    float* input_to_layer {X_train};

    for (int i {}; i<nnData::numLayers; ++i){
        auto& layer {nnData::layersMetaData[i]};

        float* A { input_to_layer };
        float* B { layer.d_weights_transposed };
        float* C { layer.d_preActivations };

        float* D { layer.d_bias };
        float* E { layer.d_activations };

        int input_size {layer.input_size};
        int output_size {layer.output_size};

        const int BLOCK_SIZE {32};
        dim3 BLOCKS (BLOCK_SIZE, BLOCK_SIZE);
        dim3 GRIDS ( (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (batch_size  + BLOCK_SIZE - 1) / BLOCK_SIZE );

        MatMul<<<GRIDS, BLOCKS>>>(A, B, C, batch_size, input_size, output_size);
        cudaDeviceSynchronize();
        utility::CUDA_CHECK(cudaGetLastError());

        int activation_id {};
        if (activation_fn[i] == "sigmoid") activation_id = 0;
        else if (activation_fn[i] == "relu") activation_id = 1;
        else if (activation_fn[i] == "linear") activation_id = 2;
        else if (activation_fn[i] == "softmax") activation_id = 3;
        
        if (activation_id==3){
            cudaMemset(layer.d_maxLogits, 0, batch_size * sizeof(float));
            cudaMemset(layer.d_sumLogits, 0, batch_size * sizeof(float));
            
            dim3 BLOCK_SIZE (32, 32);
            dim3 GRID_SIZE ( (output_size + 32 - 1) / 32,
                             (batch_size  + 32 - 1) / 32 );
            addBias<<<GRID_SIZE, BLOCK_SIZE>>>(C, D, output_size, batch_size);
            utility::CUDA_CHECK(cudaGetLastError());

            int THREADS  { 1 };
            int BLOCKS { batch_size };
            maxLogit<<<BLOCKS, THREADS>>>(C, layer.d_maxLogits, layer.d_sumLogits,
            batch_size, output_size);
            cudaDeviceSynchronize();
            utility::CUDA_CHECK(cudaGetLastError());

            // cout<<"Max Logits:"<<i+1<<"\n";
            // debugPrint<<<1,1>>>(layer.d_maxLogits, batch_size, 1);
            // cudaDeviceSynchronize();
            // cout<<"Sum:"<<i+1<<"\n";
            // debugPrint<<<1,1>>>(layer.d_sumLogits, batch_size, 1);
            // cudaDeviceSynchronize();
        }

        // cout<<"PreAvtivation:\n";
        // debugPrint<<<1,1>>>(C, batch_size, output_size);

        addBiasAndActivation<<<GRIDS, BLOCKS>>>(C, D, E,
        batch_size, output_size, activation_id, layer.d_maxLogits, layer.d_sumLogits);
        cudaDeviceSynchronize();
        utility::CUDA_CHECK(cudaGetLastError());

        if (debug_print){
            cout<<"\nLayer "<<i+1<<":\n";
            cout<<"\nWeighted Sum GPU: \n";
            debugPrint<<<1,1>>>(C, batch_size, output_size);
            cudaDeviceSynchronize();
            utility::CUDA_CHECK(cudaGetLastError());
            cout<<"\nActivations GPU: \n";
            debugPrint<<<1,1>>>(E, batch_size, output_size);
            cudaDeviceSynchronize();
            utility::CUDA_CHECK(cudaGetLastError());
        }

        input_to_layer = E;

    }

}

void backpropagateGPU(float* Y_train, float* X_train, int batch_size,
     const vector<string>& activation_fn, const string& loss_function, int labels_len,
     bool verbose = false){

    cudaMemset(nnData::loss, 0, sizeof(float));

    for (int i {nnData::numLayers-1}; i>=0; --i){
        auto& layer {nnData::layersMetaData[i]};

        cudaMemset(layer.d_weights_gradients, 0, layer.input_size * layer.output_size * sizeof(float));
        cudaMemset(layer.d_bias_gradients, 0, layer.output_size * sizeof(float));
        cudaMemset(layer.d_errors, 0, layer.output_size * batch_size * sizeof(float));


        int activation_id {};
        if (activation_fn[i] == "sigmoid") activation_id = 0;
        else if (activation_fn[i] == "relu") activation_id = 1;
        else if (activation_fn[i] == "linear") activation_id = 2;
        else if (activation_fn[i] == "softmax") activation_id = 3;

        int loss_id {};
        if (loss_function == "MSE") loss_id = 0;
        else if (loss_function == "cross_entropy") loss_id = 1;
        else if (loss_function == "binary_cross_entropy") loss_id = 2;

        if (i==nnData::numLayers-1){
            float* A { layer.d_activations };
            float* B { Y_train };
            float* C { layer.d_errors };

            float* D { layer.d_preActivations };

            int input_size {layer.input_size};
            int output_size {layer.output_size};

            const int BLOCK_SIZE {32};
            dim3 BLOCKS (BLOCK_SIZE, BLOCK_SIZE);
            dim3 GRIDS ( (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (batch_size  + BLOCK_SIZE - 1) / BLOCK_SIZE );
                
            ComputeOutputError<<<GRIDS, BLOCKS>>>(A, B, C, D, activation_id, batch_size,
            output_size, nnData::loss, labels_len, loss_id);
            cudaDeviceSynchronize();
            utility::CUDA_CHECK(cudaGetLastError());

                // if (verbose){
                //     cout<<"\nOutput Error GPU:"<<i+1<<"\n";
                //     debugPrint<<<1,1>>>(C, batch_size, output_size);
                //     cudaDeviceSynchronize();
                //     utility::CUDA_CHECK(cudaGetLastError());
                // }
        }
        else{

            auto& nextLayer {nnData::layersMetaData[i+1]};
            float* A { nextLayer.d_errors };
            float* B { nextLayer.d_weights };
            float* C { layer.d_errors };

            float* D { layer.d_preActivations };
            float* E { layer.d_activations };

            int next_input_size {nextLayer.input_size};
            int next_output_size {nextLayer.output_size};

            const int BLOCK_SIZE {32};
            dim3 BLOCKS (BLOCK_SIZE, BLOCK_SIZE);
            dim3 GRIDS ( (next_input_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         (batch_size  + BLOCK_SIZE - 1) / BLOCK_SIZE );
            
            MatMul<<<GRIDS, BLOCKS>>>(A, B, C, batch_size, next_output_size, next_input_size);
            cudaDeviceSynchronize();
            utility::CUDA_CHECK(cudaGetLastError());

            ActivationDerivative<<<GRIDS, BLOCKS>>>(C, D, E, activation_id, batch_size, next_input_size);
            cudaDeviceSynchronize();
            utility::CUDA_CHECK(cudaGetLastError());

            // if (verbose && i!=0){
            //     cout<<"\nErrors Hidden GPU: \n";
            //     debugPrint<<<1,1>>>(C, batch_size, next_input_size);
            //     cudaDeviceSynchronize();
            //     utility::CUDA_CHECK(cudaGetLastError());
            // }
        }

        float* A { layer.d_errors };
        float* B { ( i==0 ) ? X_train : nnData::layersMetaData[i-1].d_activations };
        float* C { layer.d_weights_gradients };

        float* D { layer.d_bias_gradients };

        int input_size {layer.input_size};
        int output_size {layer.output_size};

        const int BLOCK_SIZE {32};
        dim3 BLOCKS (BLOCK_SIZE, BLOCK_SIZE);
        dim3 GRIDS ( (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (output_size  + BLOCK_SIZE - 1) / BLOCK_SIZE );
        
        ComputeWeightsGradients<<<GRIDS, BLOCKS>>>(A, B, C, batch_size, output_size, input_size, 1.0f);
        cudaDeviceSynchronize();
        utility::CUDA_CHECK(cudaGetLastError());

        int BLOCK_BIAS { 256 };
        int GRID_BIAS { (output_size + BLOCK_BIAS - 1) / BLOCK_BIAS };

        ComputeBiasGradients<<<GRID_BIAS, BLOCK_BIAS>>>(A, D, batch_size, output_size,
        1.0f, nnData::loss, nnData::layersMetaData[nnData::numLayers-1].output_size);
        cudaDeviceSynchronize();
        utility::CUDA_CHECK(cudaGetLastError());

        // {
        if (verbose && i!=0){
            cout<<"\nWeights Gradients GPU Layer "<<i+1<<": "<<"\n";
            debugPrint<<<1,1>>>(C, output_size, input_size);
            cudaDeviceSynchronize();
            cout<<"Bias Gradients GPU Layer "<<i+1<<": "<<"\n";
            debugPrint<<<1,1>>>(D, 1, output_size);
            cudaDeviceSynchronize();
        }
        // }
    }
}

void updateWeightsGPU(float learning_rate, const string& optimizer){

    for (int i{}; i<nnData::numLayers; ++i){
        
        auto& layer {nnData::layersMetaData[i]};
        if (optimizer=="SGD"){
            float* W { layer.d_weights };
            float* W_grad { layer.d_weights_gradients };
    
            float* B { layer.d_bias };
            float* B_grad { layer.d_bias_gradients };
    
            int w_length { layer.input_size * layer.output_size };
            int b_length { layer.output_size };
    
            updateWeightsKernel<<<ceil(w_length/256.0f), 256>>>(W, W_grad, learning_rate, w_length);
            updateBiasesKernel<<<ceil(b_length/256.0f), 256>>>(B, B_grad, learning_rate, b_length);
            cudaDeviceSynchronize();

            utility::CUDA_CHECK(cudaGetLastError());

            // cout<<"update weights GPU\n";
            // debugPrint<<<1,1>>>(W, layer.output_size, layer.input_size);
            // cudaDeviceSynchronize();
            // cout<<"upadte bias GPU\n";
            // debugPrint<<<1,1>>>(B, 1, layer.output_size);
            // cudaDeviceSynchronize();
        }
    }
}


void saveParametersToHost(vector<MatrixXd>& weights, vector<VectorXd>& biases){

    assert(weights.size() == nnData::numLayers);
    assert(biases.size() == nnData::numLayers);

    for (int i{}; i<nnData::numLayers; ++i){
        auto& layer {nnData::layersMetaData[i]};

        int in {layer.input_size};
        int out {layer.output_size};

        vector<float> weights_vec (out * in);
        vector<float> biases_vec (out);
        
        auto copy1 { cudaMemcpy(weights_vec.data(), layer.d_weights, out * in * sizeof(float), cudaMemcpyDeviceToHost) };
        utility::CUDA_CHECK(copy1);
        auto copy2 { cudaMemcpy(biases_vec.data(), layer.d_bias, out * sizeof(float), cudaMemcpyDeviceToHost) };
        utility::CUDA_CHECK(copy2);

        weights[i] = utility::toMatrix(weights_vec, out, in);
        biases[i] = utility::toVector(biases_vec, out);

    }

}
