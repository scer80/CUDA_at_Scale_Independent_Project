CC := g++
NVCC := $(CUDA_PATH)/bin/nvcc

DATA_DIR = data
GZ_DIR := $(DATA_DIR)/gz
MNIST_DIR := $(DATA_DIR)/mnist

SRC_DIR := src
INCLUDES := -Iinclude -I/usr/local/cuda-11.8/targets/x86_64-linux/include -I/usr/include/opencv4
LIB_DIRS := -L/usr/local/cuda-11.8/targets/x86_64-linux/lib -L/usr/lib/x86_64-linux-gnu
LIBS := -lopencv_core -lopencv_imgcodecs -lopencv_highgui

DOWNLOAD_COMMAND=wget --no-check-certificate
MNIST_URL=https://ossci-datasets.s3.amazonaws.com/mnist
MNIST_FILES := \
	train-images-idx3-ubyte.gz \
	train-labels-idx1-ubyte.gz \
	t10k-images-idx3-ubyte.gz \
	t10k-labels-idx1-ubyte.gz



.PHONY: download clean

download:
	mkdir -p $(GZ_DIR) $(MNIST_DIR)
	@for file in $(MNIST_FILES); do \
		if [ ! -f "$(GZ_DIR)/$$file" ]; then \
			echo "Downloading $$file..."; \
			$(DOWNLOAD_COMMAND) -P $(GZ_DIR) $(MNIST_URL)/$$file || exit 1; \
		else \
			echo "$$file already exists, skipping."; \
		fi; \
	done
	@for file in $(MNIST_FILES); do \
		gunzip -c $(GZ_DIR)/$$file > $(MNIST_DIR)/$${file%.gz}; \
	done

%.o: $(SRC_DIR)/%.cpp
	$(CC) $(INCLUDES) -c $< -o $@

mnist_export: mnist_export.o mnist_dataloader.o
	$(CC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)


clean:
	@echo "Cleaning."
	rm *.o
