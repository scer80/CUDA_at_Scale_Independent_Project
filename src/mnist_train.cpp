#include <filesystem>
#include <iostream>
#include <unordered_map>

#include <mnist_dataloader.hpp>

using namespace std;


tuple<int> parseCommandLineArgs(int argc, char* argv[]) {
    vector<string> args(argv, argv + argc);

    int epochs = 1;

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--epochs") {
            epochs = stoi(args[i + 1]);
        }
    }
    return {epochs};
}

int main(int argc, char* argv[]) {
    auto [epochs] = parseCommandLineArgs(argc, argv);

    auto data = unordered_map<string, ImageData>();

    try {
        MnistDataloader loader(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
            "data/mnist/t10k-images-idx3-ubyte",
            "data/mnist/t10k-labels-idx1-ubyte"
        );

        
        data["test"] = loader.read_test_data();
        data["train"] = loader.read_training_data();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
