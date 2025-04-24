#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <mnist_dataloader.hpp>

using namespace std;


tuple<string, int, string> parseCommandLineArgs(int argc, char* argv[]) {
    vector<string> args(argv, argv + argc);

    string dataset = "train";
    int index = 0;
    string out = "out";

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--dataset") {
            dataset = args[i + 1];
        }
        if (args[i] == "--index") {
            index = std::stoi(args[i + 1]);
        }
        if (args[i] == "--out") {
            out = args[i + 1];
        }
    }
    return {dataset, index, out};
}


void saveSample(const vector<uint8_t>& image, const string& dataset, int index, int label, const string& dst_folder) {
    filesystem::create_directories(dst_folder);
    string dst_filename = (
        dst_folder + "/" + \
        dataset + "." + to_string(index).insert(0, 6 - to_string(index).length(), '0') + "." + to_string(label) + ".png"
    );

    cv::Mat img(MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH, CV_8UC1, const_cast<uint8_t*>(image.data()));
    cv::imwrite(dst_filename, img);
}


int main(int argc, char* argv[]) {
    auto [dataset, index, out] = parseCommandLineArgs(argc, argv);
    ImageData data;

    try {
        MnistDataloader loader(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            "data/mnist/t10k-images-idx3-ubyte",
            "data/mnist/t10k-labels-idx1-ubyte"
        );

        if (dataset == "test") {
            data = loader.read_test_data();
        } else {
            data = loader.read_training_data();
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    

    auto label = static_cast<int>(data.labels[index]);
    auto image = data.images[index];

    cout << image.size() << " " << label << endl;
    saveSample(image, dataset, index, label, out);

    return 0;
}
