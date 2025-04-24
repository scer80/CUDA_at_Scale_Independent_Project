#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include <mnist_dataloader.hpp>


MnistDataloader::MnistDataloader(
    const std::string& training_images_filepath,
    const std::string& training_labels_filepath,
    const std::string& test_images_filepath,
    const std::string& test_labels_filepath
)
    : training_images_filepath(std::move(training_images_filepath)),
      training_labels_filepath(std::move(training_labels_filepath)),
      test_images_filepath(std::move(test_images_filepath)),
      test_labels_filepath(std::move(test_labels_filepath))
{}

ImageData MnistDataloader::read_training_data() {
    return read_images_labels(training_images_filepath, training_labels_filepath);
}

ImageData MnistDataloader::read_test_data() {
    return read_images_labels(test_images_filepath, test_labels_filepath);
}

uint32_t MnistDataloader::read_header(const std::vector<char>& buffer, size_t position) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value = (value << 8) | static_cast<uint8_t>(buffer[position + i]);
    }
    return value;
}

ImageData MnistDataloader::read_images_labels(const std::string& images_filepath, const std::string& labels_filepath) {
    // Read labels
    std::ifstream labels_file(labels_filepath, std::ios::binary);
    if (!labels_file) {
        throw std::runtime_error("Cannot open labels file: " + labels_filepath);
    }
    std::vector<char> labels_buffer((std::istreambuf_iterator<char>(labels_file)), 
                                  std::istreambuf_iterator<char>());
    labels_file.close();
    if (labels_buffer.size() < 8) {
        throw std::runtime_error("Labels file is too small: " + labels_filepath);
    }
    uint32_t magic = read_header(labels_buffer, 0);
    if (magic != 2049) {
        throw std::runtime_error("Magic number mismatch in labels file, expected 2049, got " + std::to_string(magic));
    }
    uint32_t num_labels = read_header(labels_buffer, 4);
    std::vector<uint8_t> labels(labels_buffer.begin() + 8, labels_buffer.end());
    // Read images
    std::ifstream images_file(images_filepath, std::ios::binary);
    if (!images_file) {
        throw std::runtime_error("Cannot open images file: " + images_filepath);
    }
    std::vector<char> images_buffer((std::istreambuf_iterator<char>(images_file)), 
                                  std::istreambuf_iterator<char>());
    images_file.close();
    if (images_buffer.size() < 16) {
        throw std::runtime_error("Images file is too small: " + images_filepath);
    }
    magic = read_header(images_buffer, 0);
    if (magic != 2051) {
        throw std::runtime_error("Magic number mismatch in images file, expected 2051, got " + std::to_string(magic));
    }
    uint32_t num_images = read_header(images_buffer, 4);
    uint32_t rows = read_header(images_buffer, 8);
    uint32_t cols = read_header(images_buffer, 12);
    if (num_images != num_labels) {
        throw std::runtime_error("Number of images and labels don't match");
    }
    // Extract image data
    const size_t image_size = rows * cols;
    const size_t data_start = 16;
    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(image_size));
    for (size_t i = 0; i < num_images; ++i) {
        size_t start = data_start + i * image_size;
        if (start + image_size > images_buffer.size()) {
            throw std::runtime_error("Image data exceeds file size");
        }
        std::copy(images_buffer.begin() + start, 
                 images_buffer.begin() + start + image_size, 
                 images[i].begin());
    }
    return {images, labels};
}

