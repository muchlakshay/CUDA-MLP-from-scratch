#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

// Helper function to reverse the byte order (for big endian to little endian conversion)
inline uint32_t reverse_bytes(uint32_t val) {
    return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | 
           ((val >> 8) & 0xFF00) | ((val >> 24) & 0xFF);
}

MatrixXd load_mnist_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }
    
    // Read magic number and number of items
    uint32_t magic_number, num_items;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    
    // Convert from big endian to little endian if needed
    magic_number = reverse_bytes(magic_number);
    num_items = reverse_bytes(num_items);
    
    // Check magic number
    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file: " + file_path);
    }
    
    // Read all labels
    std::vector<uint8_t> labels(num_items);
    file.read(reinterpret_cast<char*>(labels.data()), num_items);
    
    // Convert to one-hot encoded matrix
    MatrixXd one_hot_labels(num_items, 10);
    one_hot_labels.setZero();
    
    for (int i = 0; i < num_items; ++i) {
        uint8_t label = labels[i];
        if (label >= 10) {
            throw std::runtime_error("Invalid label value: " + std::to_string(label));
        }
        one_hot_labels(i, label) = 1.0;
    }
    
    return one_hot_labels;
}

MatrixXd load_mnist_images(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }
    
    // Read magic number and dimensions
    uint32_t magic_number, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    // Convert from big endian to little endian if needed
    magic_number = reverse_bytes(magic_number);
    num_images = reverse_bytes(num_images);
    rows = reverse_bytes(rows);
    cols = reverse_bytes(cols);
    
    // Check magic number
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file: " + file_path);
    }
    
    // Read all image data
    std::vector<uint8_t> images(num_images * rows * cols);
    file.read(reinterpret_cast<char*>(images.data()), num_images * rows * cols);
    
    // Convert to MatrixXd (normalized to 0-1 range)
    MatrixXd image_matrix(num_images, rows * cols);
    
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            image_matrix(i, j) = images[i * rows * cols + j] / 255.0;
        }
    }
    
    return image_matrix;
}

#endif // MNIST_LOADER_H
