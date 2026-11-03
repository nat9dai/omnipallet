# Dependencies for D2SLAM (D2VINS quad-fisheye VIO), CPU build, ROS 1 Noetic + Foxy bridge; run.sh mounts the source.
# CPU only: the RTX 5080 (sm_120) needs TensorRT >= 10.8, which the upstream CNN code predates.
FROM osrf/ros:noetic-desktop-full-focal
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ARG CMAKE_VERSION=3.24.1
ARG ONNX_VERSION=1.18.0
ARG FAISS_VERSION=1.7.4

RUN apt-get update && apt-get install -y --no-install-recommends \
      wget curl git unzip build-essential gdb \
      libatlas-base-dev libeigen3-dev libgoogle-glog-dev libsuitesparse-dev libglib2.0-dev \
      libyaml-cpp-dev libdw-dev libboost-program-options-dev \
      python3-catkin-tools python3-pip \
      ros-noetic-image-transport-plugins ros-noetic-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# CMake >= 3.17 for faiss (focal ships 3.16); install next to, not over, the system one
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh -O /tmp/c.sh \
    && sh /tmp/c.sh --skip-license --prefix=/usr/local && rm /tmp/c.sh && /usr/local/bin/cmake --version

# Ceres 2.1 (HKUST fork: needs both ceres::Manifold and LocalParameterization)
RUN git clone --depth 1 -b D2SLAM https://github.com/HKUST-Swarm/ceres-solver /tmp/ceres \
    && cmake -S /tmp/ceres -B /tmp/ceres/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
       -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
    && cmake --build /tmp/ceres/build -j"$(nproc)" --target install && rm -rf /tmp/ceres

# ONNX Runtime CPU x64: linked by loop_cnn but never called in the CPU build
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz -O /tmp/ort.tgz \
    && tar -xzf /tmp/ort.tgz -C /tmp && cp -r /tmp/onnxruntime-linux-x64-${ONNX_VERSION}/include/* /usr/local/include/ \
    && cp -r /tmp/onnxruntime-linux-x64-${ONNX_VERSION}/lib/* /usr/local/lib/ && rm -rf /tmp/ort.tgz /tmp/onnxruntime-*

# LCM (link time; runtime only for loops or multi-drone)
RUN git clone --depth 1 -b v1.4.0 https://github.com/lcm-proj/lcm /tmp/lcm \
    && cmake -S /tmp/lcm -B /tmp/lcm/build -DCMAKE_BUILD_TYPE=Release -DLCM_ENABLE_TESTS=OFF -DLCM_ENABLE_EXAMPLES=OFF \
       -DLCM_ENABLE_JAVA=OFF -DLCM_ENABLE_LUA=OFF -DLCM_ENABLE_PYTHON=OFF \
    && cmake --build /tmp/lcm/build -j"$(nproc)" --target install && rm -rf /tmp/lcm

# faiss CPU (AVX2 on x86)
RUN git clone --depth 1 -b v${FAISS_VERSION} https://github.com/facebookresearch/faiss.git /tmp/faiss \
    && /usr/local/bin/cmake -S /tmp/faiss -B /tmp/faiss/build -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 \
       -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF \
    && make -C /tmp/faiss/build -j"$(nproc)" faiss faiss_avx2 && make -C /tmp/faiss/build install && rm -rf /tmp/faiss

# OpenGV (HKUST fork)
RUN git clone --depth 1 https://github.com/HKUST-Swarm/opengv /tmp/opengv \
    && cmake -S /tmp/opengv -B /tmp/opengv/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/opengv/build -j"$(nproc)" --target install && rm -rf /tmp/opengv

# backward-cpp header and spdlog 1.12
RUN wget -q https://raw.githubusercontent.com/bombela/backward-cpp/v1.6/backward.hpp -O /usr/local/include/backward.hpp \
    && wget -q https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.tar.gz -O /tmp/spd.tgz \
    && tar -xzf /tmp/spd.tgz -C /tmp && cmake -S /tmp/spdlog-1.12.0 -B /tmp/spdlog-1.12.0/build -DCMAKE_BUILD_TYPE=Release \
       -DSPDLOG_BUILD_EXAMPLE=OFF \
    && cmake --build /tmp/spdlog-1.12.0/build -j"$(nproc)" --target install && rm -rf /tmp/spd.tgz /tmp/spdlog-1.12.0

# libtorch CPU: headers are included by superpoint_common.h even in the CPU build
RUN wget -q "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip" -O /tmp/lt.zip \
    && unzip -q /tmp/lt.zip -d /opt && rm /tmp/lt.zip

# ROS 2 Foxy + ros1_bridge, so one container bridges D2VINS to the host's ROS 2 Humble over DDS
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends ros-foxy-ros-base ros-foxy-ros1-bridge \
       ros-foxy-sensor-msgs ros-foxy-nav-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig
ENV SWARM_WS=/root/swarm_ws
WORKDIR /root/swarm_ws
