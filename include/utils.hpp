#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <string>

template <typename T>
void print_2d(const string& name, const T* matrix, int rows, int cols) {
    std::cout << name << std::endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << matrix[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}


#endif
