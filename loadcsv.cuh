#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <numeric>

struct EigenDataT {
    std::vector<std::string> header;
    Eigen::MatrixXd X_train;
    Eigen::VectorXd Y_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXd Y_test;
};

EigenDataT load_csv_eigen(const std::string& filename, const std::string& target_column, double training_ratio) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    EigenDataT data;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file or no header found.");
    }

    std::stringstream header_stream(line);
    std::string col;
    while (std::getline(header_stream, col, ',')) {
        data.header.push_back(col);
    }

    int target_index = -1;
    for (size_t i = 0; i < data.header.size(); ++i) {
        if (data.header[i] == target_column) {
            target_index = static_cast<int>(i);
            break;
        }
    }

    if (target_index == -1) {
        throw std::runtime_error("Target column '" + target_column + "' not found.");
    }

    std::vector<std::vector<double>> rows;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;

        while (std::getline(ss, value, ',')) {
            if (value.empty() || value == " ") row.push_back(0.0);
            else row.push_back(std::stod(value));
        }

        if (row.size() != data.header.size()) {
            throw std::runtime_error("Row size does not match header size.");
        }

        rows.push_back(std::move(row));
    }

    file.close();

    size_t num_rows = rows.size();
    size_t num_features = data.header.size() - 1;

    Eigen::MatrixXd X(num_rows, num_features);
    Eigen::VectorXd Y(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        Y(i) = rows[i][target_index];

        int col_idx = 0;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            if (static_cast<int>(j) == target_index) continue;
            X(i, col_idx++) = rows[i][j];
        }
    }

    std::vector<size_t> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end()); 

    Eigen::MatrixXd X_shuffled(num_rows, num_features);
    Eigen::VectorXd Y_shuffled(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        X_shuffled.row(i) = X.row(indices[i]);
        Y_shuffled(i) = Y(indices[i]);
    }

    size_t train_size = static_cast<size_t>(training_ratio * num_rows);
    data.X_train = X_shuffled.topRows(train_size);
    data.Y_train = Y_shuffled.head(train_size);
    data.X_test  = X_shuffled.bottomRows(num_rows - train_size);
    data.Y_test  = Y_shuffled.tail(num_rows - train_size);

    return data;
}

void normalizeMatrixXd(Eigen::MatrixXd& matrix) {
    matrix = matrix.rowwise() - matrix.colwise().mean();
    matrix = matrix.array().rowwise() / matrix.colwise().norm().array();
}

Eigen::MatrixXd toMatrixXd(const Eigen::VectorXd& vec){
    Eigen::MatrixXd matrix (vec.size(), 1);
    for (int i {}; i<vec.size(); ++i){
         matrix(i, 0) = vec[i];
    }
    return  matrix;
}
