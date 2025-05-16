#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>

using namespace std;
using namespace Eigen;

namespace utility{
    vector<float> flatten(const MatrixXd& mat){
        vector<float> flat (mat.rows()*mat.cols());

        for (int i {}; i<mat.rows(); ++i){
            for (int j {}; j<mat.cols(); ++j) flat[i*mat.cols()+j] = mat(i, j);
        }
        return flat;
    }
    
    vector<float> flatten(const VectorXd& vec){
        vector<float> flat;
        for (int i {}; i<vec.size(); ++i){
            flat.push_back(vec(i));
        }
        return flat;
    }

    MatrixXd toMatrix(vector<float>& vec, int rows, int cols){
        MatrixXd mat (rows, cols);
        for (int i {}; i<mat.rows(); ++i){
            for (int j {}; j<mat.cols(); ++j){
                mat(i, j) = vec[i*cols+j];
            }
        }
        // std::vector<double> vec_d(vec.begin(), vec.end());
        // using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        // using RowMajorMap = Eigen::Map<RowMajorMatrix>;
        // RowMajorMap mat(vec_d.data(), rows, cols);
        return mat;
    }

    VectorXd toVector(vector<float>& vec, int elements){
        VectorXd vector (elements);
        for (int i {}; i<elements; ++i){
            vector(i) = vec[i];
        }
        return vector;
    }

    void CUDA_CHECK(cudaError_t err){
        if (err!=cudaSuccess) {
            cout<<"Cuda Error: "<<cudaGetErrorString(err)
                <<" In File: "  <<__FILE__
                <<" In Line: "  <<__LINE__;
            exit(EXIT_FAILURE);   
        }
    }

    bool isCudaAvailable() {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        return (err == cudaSuccess && deviceCount > 0);
    }

    void shuffleData(MatrixXd& X_train, MatrixXd& Y_train){

        random_device rd;
        mt19937 mt{rd()};

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm (X_train.rows());
        perm.setIdentity();
        shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), mt);

        X_train = perm * X_train;
        Y_train = perm * Y_train;
    }

    MatrixXd toOneHot(VectorXd& labels, int num_labels){
        MatrixXd one_hot {MatrixXd::Zero(labels.rows(), num_labels)};
        for (std::size_t i {}; i<labels.rows(); ++i){
            one_hot(i, static_cast<int>(labels(i))) = 1.0;
        }
        return one_hot;
    }
}
