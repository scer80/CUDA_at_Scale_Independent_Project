#ifndef _MNIST_DATALOADER_H_
#define _MNIST_DATALOADER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_WIDTH 28


struct ImageData {
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;
};


class MnistDataloader {
public:
    MnistDataloader(
        const std::string& training_images_filepath,
        const std::string& training_labels_filepath,
        const std::string& test_images_filepath,
        const std::string& test_labels_filepath
    );

    ImageData read_training_data();
    ImageData read_test_data();

private:
    std::string training_images_filepath;
    std::string training_labels_filepath;
    std::string test_images_filepath;
    std::string test_labels_filepath;

    uint32_t read_header(const std::vector<char>& buffer, size_t position);
    ImageData read_images_labels(const std::string& images_filepath, const std::string& labels_filepath);
};

#endif
