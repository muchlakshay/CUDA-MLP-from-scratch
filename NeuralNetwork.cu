#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>
#include "loadcsv.cuh"
#include "nnBackend.cuh"
#include "load_mnist.cuh"
#include <chrono>

using namespace std;
using namespace Eigen;

class NeuralNetwork{

    vector<MatrixXd> weights;
    vector<VectorXd> bias;
    vector<int> layer_sizes;
    vector<string> activation_fn;
    vector<MatrixXd> layers_weighted_sum;
    vector<MatrixXd> layers_activations;
    vector<MatrixXd> layers_error;
    vector<MatrixXd> layers_gradients;
    vector<VectorXd> layers_gradients_bias;

    string loss_function {};
    int batch_size {};
    float learning_rate {0.0f};
    float lr_shrink_factor {};
    bool verbose;
    string optimizer {};

    MatrixXd init(int row, int col){
        if (col == 0) {
            throw std::invalid_argument("Number of input neurons cannot be zero.");
        }        
        random_device rd;
        mt19937 mt {rd()};

        MatrixXd mat(row, col);
        
        float limit = sqrt(6.0 / (col + row));
        uniform_real_distribution<float> range(-limit, limit);
            
        for (int i {}; i<row; ++i){
            for (int j {}; j<col; ++j){
                mat(i, j) = range(mt);
            }
        }
        return mat;
    }
    
    double calculate_loss(MatrixXd Y_train, MatrixXd output){
        double loss {};
        if (loss_function=="MSE"){
            // cout<<"Y: "<<Y_train<<" O: "<<output<<"\n";
            loss = (Y_train - output).array().square().mean();
        }
        return loss;
    }

    MatrixXd softmax(const MatrixXd& z) {
        MatrixXd result = z;

        for (int i = 0; i < z.rows(); ++i) {
            double row_max = z.row(i).maxCoeff();
            VectorXd exp_row = (z.row(i).array() - row_max).exp();
            // cout<<i+1<<" max: "<<row_max<<" sum: "<<exp_row.sum()<<"\n";
            result.row(i) = exp_row / exp_row.sum();
        }
        return result;
    }

    void forwardPassCPU(const MatrixXd& x_train, bool debug_print=false){
        layers_weighted_sum.clear();
        layers_activations.clear();

        MatrixXd input_to_layer {x_train};
        for (int i {}; i<layer_sizes.size()-1; ++i){

            MatrixXd weighted_sum { input_to_layer * weights[i].transpose() };

            weighted_sum.rowwise() += bias[i].transpose();
            MatrixXd activations (weighted_sum.rows(), weighted_sum.cols());

            if (activation_fn[i]=="sigmoid") 
                activations = weighted_sum.unaryExpr([](double z){return 1.0/(1+exp(-z));});
            else if (activation_fn[i]=="relu")
                activations = weighted_sum.unaryExpr([](double z){return max(0.0, z);});
            else if (activation_fn[i]=="linear")
                activations = weighted_sum;
            else if (activation_fn[i]=="softmax")
                activations = softmax(weighted_sum);

            layers_weighted_sum.push_back(weighted_sum);
            layers_activations.push_back(activations);
            input_to_layer = activations;

            if (debug_print){
                cout<<"\nLayer "<<i+1<<":\n";
                cout<<"\nWeighted Sum: \n"<<weighted_sum<<"\n";
                cout<<"\nActivations: \n"<<activations<<"\n";
            }
        }
    }
    double sigmoidDerivative(double z) {
        return z * (1.0 - z);
    }
    double reluDerivative(double z){
        return (z>0.0) ? 1.0 : 0.0;
    }

    void backpropagateCPU(const MatrixXd& Y_train, const MatrixXd& X_train,
        const vector<string>& activation_fn, bool debug_print = false){

        layers_error.clear();
        layers_gradients.clear();
        layers_gradients_bias.clear();

        layers_error.assign(layer_sizes.size() - 1, MatrixXd());  
        layers_gradients.assign(layer_sizes.size() - 1, MatrixXd()); 
        layers_gradients_bias.assign(layer_sizes.size() - 1, VectorXd());

        
        for (int i = layer_sizes.size()-2 ; i>=0; --i){
            if (i==layer_sizes.size()-2){
                if (loss_function=="MSE"){
                    layers_error[i] = 2.0*(layers_activations[i] - Y_train);
                    if (activation_fn[i]=="sigmoid"){
                       layers_error[i] = layers_error[i].cwiseProduct(
                        (layers_activations[i].unaryExpr([this](double z){ return sigmoidDerivative(z); })) );
                    }
                    else if (activation_fn[i]=="relu"){
                        layers_error[i] = layers_error[i].cwiseProduct(
                            (layers_weighted_sum[i].unaryExpr([this](double z){ return reluDerivative(z); })) );
                    }

                }
                else if (loss_function=="cross_entropy"){
                    if (activation_fn[i]=="softmax"){
                        layers_error[i] = layers_activations[i]-Y_train;
                        // if (debug_print)
                        // cout<<"CPU OPT Errors: "<<" Layer "<<i+1<<"\n"<<layers_error[i]<<"\n";
                    }
                }
            }
            else{

                layers_error[i] = layers_error[i+1] * weights[i+1];
                if (activation_fn[i]=="sigmoid"){
                        layers_error[i] = layers_error[i].cwiseProduct(
                        (layers_activations[i].unaryExpr([this](double z){ return sigmoidDerivative(z); })) );
                }
                else if (activation_fn[i]=="relu"){
                    layers_error[i] = layers_error[i].cwiseProduct(
                    (layers_weighted_sum[i].unaryExpr([this](double z){ return reluDerivative(z); })) );
                }
                // if (debug_print && i!=0)
                // cout<<"CPU HDDN ERR: \n"<<layers_error[i]<<"\n";
            }
            
            auto prev_layer_activations = (i==0) ? X_train : layers_activations[i-1];

            // Gradient clipping

            layers_gradients[i] = (layers_error[i].transpose() * prev_layer_activations)/batch_size;
            layers_gradients_bias[i] = (layers_error[i].colwise().sum()/batch_size).transpose(); 

            layers_gradients[i] = layers_gradients[i].unaryExpr([](double g) {
                return max(-1.0f, min(1.0f, g));
            });
            layers_gradients_bias[i] = layers_gradients_bias[i].unaryExpr([](double g) {
                return max(-1.0f, min(1.0f, g));
            });

            if (debug_print){
                // if (i==0);s
                // else {
                    cout<<"\nLayer "<<i+1<<" Gradient: \n"<<layers_gradients[i]<<"\n";
                    cout<<"\nLayer "<<i+1<<" Gradient Bias: \n"<<layers_gradients_bias[i]<<"\n";
                }
            } 
        }
    

    void updateWeightsCPU(float learning_rate){
        for (int i {}; i<layer_sizes.size()-1; ++i){

            if (optimizer=="SGD"){
                weights[i] -= (learning_rate * layers_gradients[i]);
                bias[i] -= (learning_rate * layers_gradients_bias[i]);
            }

            // cout<<"update weights: \n"<<weights[i]<<"\n";
            // cout<<"update bias: \n"<<bias[i]<<"\n";
        }
    }

    void learnGPU(MatrixXd& X_train, MatrixXd& Y_train, int epochs){

        initLayersGPU(weights, bias, batch_size);
        float lr { learning_rate };

        // loadDataToGPU(X_train, Y_train , false);
        cout<<"Learning...\n";
        for (int epoch {1}; epoch<=epochs; ++epoch){

            auto t0 { chrono::high_resolution_clock::now() };

            utility::shuffleData(X_train , Y_train);

            auto t1 { chrono::high_resolution_clock::now() };

            loadDataToGPU(X_train, Y_train , false);

            auto t2 { chrono::high_resolution_clock::now() };

            if(verbose) cout<<"[Epoch "<<epoch<<"] ";

            int startIdxX {};
            int startIdxY {};
            int rowIdx {};

            for (int i{}; i<ceil(X_train.rows()/static_cast<float>(batch_size)); ++i){
                int current_batch_size {min(batch_size, static_cast<int>(X_train.rows())-rowIdx)};

                // forwardPassCPU(X_train.block(rowIdx, 0, current_batch_size, X_train.cols()), false);

                forwardPassGPU( (nnData::d_X_train + startIdxX ),
                                 current_batch_size, activation_fn, false);

                // backpropagateCPU(Y_train.block(rowIdx, 0, current_batch_size, Y_train.cols()),
                // X_train.block(rowIdx, 0, current_batch_size, X_train.cols()),
                // activation_fn, true);

                backpropagateGPU( (nnData::d_Y_train + startIdxY), ( nnData::d_X_train + startIdxX ),
                                   current_batch_size, activation_fn, loss_function, Y_train.cols(), false);

                // updateWeightsCPU(lr);
                
                updateWeightsGPU(lr, optimizer);

                startIdxX += X_train.cols() * current_batch_size;
                startIdxY += Y_train.cols() * current_batch_size;
                rowIdx += current_batch_size;

            }

            if (epoch%5==0 && epoch!=0){
                lr*= lr_shrink_factor;
            }
            
            auto t3 { chrono::high_resolution_clock::now() };
            if (verbose) {
                auto shuffle_duration = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
                auto load_duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
                auto training_duration = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
                auto toatl_duration = chrono::duration_cast<chrono::milliseconds>(t3 - t0).count();

                // cout<<"Load : "<<shuffle_duration<<"ms | ";
                cout<<"Load: "<<load_duration<<"ms | ";
                cout<<"Train: "<<training_duration<<"ms | ";
                cout<<"Total: "<<toatl_duration<<"ms | ";

                printLoss<<<1,1>>>(nnData::loss, batch_size, Y_train.cols());
                cudaDeviceSynchronize();
            }
            freeGPUDataMem();

        }
        cout<<"Learning Complete!";
        // printParams();
        cudaDeviceSynchronize();
        saveParametersToHost(weights, bias);
        freeGPU();
    }

public:

    enum class TrainingDevice{
        GPU,CPU
    };

    void learn(MatrixXd& X_train, MatrixXd& Y_train, int epochs, TrainingDevice device){

        assert(X_train.cols()==layer_sizes[0] && X_train.rows()==Y_train.rows());

        if ( device==TrainingDevice::GPU && !utility::isCudaAvailable() ){
                std::cerr << "CUDA device not found. Exiting, switching to CPU mode.\n";
                exit(1);
        }
        else if (device==TrainingDevice::GPU) {
            learnGPU(X_train, Y_train, epochs);
            return;
        }

        cout<<"Learning...\n";
        float lr { learning_rate };
        for (int epoch {1}; epoch<=epochs; ++epoch){
            
            utility::shuffleData(X_train, Y_train);
            if (verbose) cout<<"\nEpoch: "<<epoch<<"\n";
            
            float loss {};
            float lr { learning_rate };
            int rowIdx {};

            for (int i{}; i<ceil(X_train.rows()/static_cast<float>(batch_size)); ++i){
                int current_batch_size {min(batch_size, static_cast<int>(X_train.rows())-rowIdx)};


                forwardPassCPU(X_train.block(rowIdx, 0, current_batch_size, X_train.cols()), false);

                backpropagateCPU(Y_train.block(rowIdx, 0, current_batch_size, Y_train.cols()),
                X_train.block(rowIdx, 0, current_batch_size, X_train.cols()),
                activation_fn, false);

                updateWeightsCPU(lr);

                if (epoch%5==0 && epoch!=0){
                    lr*= lr_shrink_factor;
                }
                rowIdx += current_batch_size;

                if (verbose && i==ceil(X_train.rows()/batch_size)-1){
                cout<<"Loss: "<<calculate_loss(Y_train.block(rowIdx, 0, current_batch_size, Y_train.cols()), layers_activations.back())<<"\n";
                }
            }
        }
        cout<<"Learning Complete!";
    }

    void assemble(const string& Loss, const string& Optimizer, int BatchSize,
                  double LearningRate, float LR_shrink_factor=1.0f, bool Verbose=true){
        
        loss_function = Loss;
        optimizer = Optimizer;
        batch_size = BatchSize;
        learning_rate = LearningRate;
        lr_shrink_factor = LR_shrink_factor;
        verbose = Verbose;
    }

    void input(int size){layer_sizes.push_back(size);}

    void extend(int size, const string& activation){
        assert(layer_sizes.size() != 0);

        int in {layer_sizes.back()};
        int out {size};
        layer_sizes.push_back(size);
        activation_fn.push_back(activation);
        weights.push_back(init(out, in));
        bias.push_back(VectorXd::Constant(out, 0));
    }

    const auto& getWeights(){return weights;}
    const auto& getBias(){return bias;}
    void info(){
        for (int i {}; i < layer_sizes.size(); ++i){
            if (i == 0){
                cout<<"\nLayer "<<i+1<<" (Input Layer):\n";
                cout<<"Neurons: "<<layer_sizes[i]<<"\n\n";
                continue;
            }
            cout<<"Layer "<<i+1<<":\n";
            cout<<"Neurons: "<<layer_sizes[i]<<"\n"
                <<"Activation Function: "
                <<activation_fn[i-1]<<"\n\n";
        }

        int params {};
        int num_weights {};
        int num_biases {};

        for (int i {}; i<weights.size(); ++i){
            num_weights += weights[i].size();
            num_biases  += bias[i].size();
            params      += weights[i].size() + bias[i].size();
        }
        
        cout<<"Total Weights: "<<num_weights<<"\nTotal Biases: "
            <<num_biases<<"\nTotal Parameters: "<<params<<"\n";
    }   

    MatrixXd predict(const MatrixXd& to_pred){
        forwardPassCPU(to_pred);
        return layers_activations.back();
    }
    void predictGPU(const MatrixXd& to_pred){
        vector<float> data_vec {utility::flatten(to_pred)};
        float* to_pred_ptr;
        auto alloc { cudaMalloc(&to_pred_ptr, to_pred.rows()*to_pred.cols()*sizeof(float)) };
        utility::CUDA_CHECK(alloc);
        auto copy { cudaMemcpy(to_pred_ptr, data_vec.data(), to_pred.rows()*to_pred.cols()*sizeof(float), cudaMemcpyHostToDevice) };
        utility::CUDA_CHECK(copy);

       forwardPassGPU(to_pred_ptr, to_pred.rows(), activation_fn, false);
        debugPrint<<<1,1>>>(nnData::layersMetaData[nnData::numLayers-1].d_activations, to_pred.rows(), 10);
        cudaDeviceSynchronize();
        cudaFree(to_pred_ptr);
    }
};


int main(){

    // Eigen::MatrixXd X(10, 2);
    // X <<  0.1,  0.2,
    //       0.2,  0.1,
    //       0.9,  1.0,
    //       1.0,  0.9,
    //       0.85, 0.95,
    //      -0.9, -0.8,
    //      -1.0, -0.9,
    //      -0.85,-0.75,
    //       0.0,  0.0,
    //       1.0, -1.0;

    // Eigen::MatrixXd Y(10, 3);
    // Y <<  1, 0, 0,
    //     1, 0, 0,
    //     0, 1, 0,
    //     0, 1, 0,
    //     0, 1, 0,
    //     0, 0, 1,
    //     0, 0, 1,
    //     0, 0, 1,
    //     1, 0, 0,
    //     0, 1, 0;

    // auto data {load_csv_eigen("iris.csv", "Species", 0.8)};
    // MatrixXd X_train {data.X_train};
    // MatrixXd X_test {data.X_test};
    // // normalizeMatrixXd(X_train);
    // // normalizeMatrixXd(X_test);
    // MatrixXd Y_train {utility::toOneHot(data.Y_train, 3)};
    // MatrixXd Y_test {utility::toOneHot(data.Y_test, 3)};

    // // for (int i {}; i<X_train.rows(); ++i){
    // //     cout<<"X: "<<X_train.row(i)<<" Y: "<<Y_train.row(i)<<"\n";
    // // }
    // // cout<<X_train;
    // // cout<<Y_train;

    // NeuralNetwork nn;
    // nn.input(X_train.cols());
    // nn.extend(16, "sigmoid");
    // nn.extend(16, "sigmoid");
    // nn.extend(3, "softmax");

    // nn.assemble("cross_entropy", "SGD", 12, 0.001f, 1.0f, true);
    // nn.learn(X_train, Y_train, 1000, NeuralNetwork::TrainingDevice::CPU);
    // cout<<Y_test;
    // cout<<nn.predict(X_test);

    MatrixXd X_train = load_mnist_images("DATA/MNIST/train-images.idx3-ubyte");
    // MatrixXd X_train_ = X_train.block(0, 0, 50, X_train.cols());
    MatrixXd Y_train = load_mnist_labels("DATA/MNIST/train-labels.idx1-ubyte");
    // MatrixXd Y_train_ = Y_train.block(0, 0, 50, Y_train.cols());
    MatrixXd X_test = load_mnist_images("DATA/MNIST/t10k-images.idx3-ubyte");
    // MatrixXd X_test_ = X_test.block(0, 0, 50, X_test.cols());
    MatrixXd Y_test = load_mnist_labels("DATA/MNIST/t10k-labels.idx1-ubyte");
    // MatrixXd Y_test_ = Y_test.block(0, 0, 50, Y_test.cols());

    // cout<<X_train_;

    NeuralNetwork model;
    model.input(X_train.cols());
    model.extend(30, "relu");
    model.extend(16, "relu");
    model.extend(10, "softmax");
    model.assemble("cross_entropy", "SGD", X_train.rows(), 0.5f, 1.0f, true);
    // model.learn(X_train_, Y_train_, 200, NeuralNetwork::TrainingDevice::CPU);
    model.learn(X_train, Y_train, 40, NeuralNetwork::TrainingDevice::GPU);


    // model.predictGPU(X_test);
    // freeGPU();

    // cout<<"CPU\n";

    auto pred = (model.predict(X_test));
    for (int i {}; i<pred.rows(); ++i){
        cout<<"Actual: "<<Y_test.row(i)<<" Predicted: "<<pred.row(i)<<"\n";
    }

    auto w {model.getWeights()};
    for (int i {1}; i<3; ++i){
        cout<<w[i]<<"\n\n";
    }











    // EigenDataT data { load_csv_eigen("DATA/housing.csv", "median_house_value", 0.8) };
    // MatrixXd X_train {data.X_train};
    // normalizeMatrixXd(X_train);
    // // MatrixXd Y_train {utility::toOneHot(data.Y_train, 2)};
    // MatrixXd Y_train { toMatrixXd(data.Y_train) };
    // MatrixXd X_test {data.X_test};
    // normalizeMatrixXd(X_test);
    // // MatrixXd Y_test {utility::toOneHot(data.Y_test,2 )};
    // MatrixXd Y_test { toMatrixXd(data.Y_test) };

    // // cout<<"X_train length: "<<X_train.rows()<<"\n";
    // // cout<<"Y_train length: "<<Y_train.rows()<<"\n";

    // NeuralNetwork nn;
    // nn.input(X_train.cols());
    // nn.extend(16, "relu");
    // nn.extend(16, "relu");
    // nn.extend(1, "linear");

    // nn.assemble("MSE", "SGD", 256, 0.0001, 1.0f, true);
    
    // nn.learn(X_train, Y_train, 200, NeuralNetwork::TrainingDevice::GPU);
    // // auto pred_r = nn.predict(X_train).unaryExpr([](double z){return (z>0.5) ? 1.0 : 0.0;});
    // auto pred = nn.predict(X_train);

    // int correct {};
    // for (int i {}; i<Y_train.rows(); ++i){
    //     cout<<"Actual: "<<Y_train.row(i)<<" Predicted: "<<pred.row(i)<<"\n";
    //     // if (Y_train.row(i) == pred_r.row(i)){
    //     //     ++correct;
    //     // }
    // }
    // cout<<"Accuracy: "<<static_cast<float>(correct)/Y_train.rows()*100<<"\n";
    return 0;
}
