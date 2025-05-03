export WORKSPACE=$(shell pwd)
export USER_NAME=$(shell whoami)
export USER_UID=$(shell id -u)
export USER_GID=$(shell id -g)

CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

DATA_DIR = data
MNIST_DIR := $(DATA_DIR)/MNIST
MNIST_GZ_DIR := $(MNIST_DIR)/gz
MNIST_RAW_DIR := $(MNIST_DIR)/raw

SRC_DIR := src
INCLUDES := \
	-Iinclude \
	-I/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/include \
	-I/usr/include/opencv4 \
	-I/usr/local/lib/python3.10/dist-packages/include \
	-I/opt/cudnn-frontend/samples/cpp
LIB_DIRS := -L/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/ -L/usr/lib/x86_64-linux-gnu
LIBS := \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui \
	-lcudnn -lcudnn_graph -lcudnn_cnn -lcudart \
	-lcublas
# 
# -lCatch2 \

DOWNLOAD_COMMAND=wget --no-check-certificate
MNIST_URL=https://ossci-datasets.s3.amazonaws.com/mnist
MNIST_FILES := \
	train-images-idx3-ubyte.gz \
	train-labels-idx1-ubyte.gz \
	t10k-images-idx3-ubyte.gz \
	t10k-labels-idx1-ubyte.gz

PROJECT_NAME := cuda
DEV_SERVICE := dev
DOCKER_COMPOSE_FILE := docker/cuda.docker-compose.yml
DOCKER_COMPOSE_CMD := docker compose --project-name $(PROJECT_NAME) --file $(DOCKER_COMPOSE_FILE)

build:
	$(DOCKER_COMPOSE_CMD) \
		--progress plain \
		build

shell:
	$(DOCKER_COMPOSE_CMD) up -d $(DEV_SERVICE) \
	&& $(DOCKER_COMPOSE_CMD) exec $(DEV_SERVICE) /bin/bash

stop:
	$(DOCKER_COMPOSE_CMD) down

.PHONY: download clean

download:
	mkdir -p $(MNIST_GZ_DIR) $(MNIST_RAW_DIR)
	@for file in $(MNIST_FILES); do \
		if [ ! -f "$(MNIST_GZ_DIR)/$$file" ]; then \
			echo "Downloading $$file..."; \
			$(DOWNLOAD_COMMAND) -P $(MNIST_GZ_DIR) $(MNIST_URL)/$$file || exit 1; \
		else \
			echo "$$file already exists, skipping."; \
		fi; \
	done
	@for file in $(MNIST_FILES); do \
		gunzip -c $(MNIST_GZ_DIR)/$$file > $(MNIST_RAW_DIR)/$${file%.gz}; \
	done

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(INCLUDES) -c $< -o $@

%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(INCLUDES) -c $< -o $@

mnist_export: mnist_export.o mnist_dataloader.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

mnist_train: mnist_train.o mnist_dataloader.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

temp: temp.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

cudnn_backend: cudnn_backend.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

gemini_reduction: gemini_reduction.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

use_linear: use_linear.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

use_activation: use_activation.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

use_mlp: use_mlp.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

use_softmax: use_softmax.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

use_nll_loss: $(SRC_DIR)/use_nll_loss.cu
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

use_fused_softmax_nll_loss: $(SRC_DIR)/use_fused_softmax_nll_loss.cu
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

sgemm: sgemm.o
	$(NVCC) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

clean:
	@echo "Cleaning."
	-rm *.o
