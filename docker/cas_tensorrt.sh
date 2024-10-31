#!/bin/bash

sudo docker run --privileged -d -p 51000:51000 --gpus all supervisely/cas_tensorrt