// ------------------------------------------------------------------
// Deep Iterative Matching Network
// Modified from MATLAB Faster R-CNN (https://github.com/shaoqingren/faster_rcnn)
// ------------------------------------------------------------------

#include "gpu_flow.hpp"
#include <iostream>

const int CAFFE_CUDA_NUM_THREADS = 512;

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void flow_kernel(const int nthreads,
                           const float *depth_src, const float *depth_tgt,
                           int height, int width,
                           float* KT, float* Kinv,
                           float* flow, float* valid) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int batch_idx = index / width / height;
    float d_src = depth_src[index];
    float x = (w*Kinv[0] + h*Kinv[1] + Kinv[2])*d_src;
    float y = (w*Kinv[3] + h*Kinv[4] + Kinv[5])*d_src;
    float z = d_src;
    if (d_src > 1E-3){
        KT += batch_idx * 12;
        float x_proj = x*KT[0] + y*KT[1] + z*KT[2] + KT[3];
        float y_proj = x*KT[4] + y*KT[5] + z*KT[6] + KT[7];
        float z_proj = x*KT[8] + y*KT[9] + z*KT[10] + KT[11] + 1E-15;
        float w_proj = x_proj / z_proj;
        float h_proj = y_proj / z_proj;
        int w_proj_i = round(w_proj);
        int h_proj_i = round(h_proj);
        if (w_proj>=0 && w_proj<=width-1 && h_proj>=0 && h_proj<=height-1){
            float d_tgt = depth_tgt[(batch_idx*height+h_proj_i)*width+w_proj_i];
            if (abs(z_proj - d_tgt) < 3E-3) {
                flow[((batch_idx*2+0)*height+h)*width+w] = h_proj-h;
                flow[((batch_idx*2+1)*height+h)*width+w] = w_proj-w;
                valid[index] = 1;
                return;
            }
        }
    }
    flow[((batch_idx*2+0)*height+h)*width+w] = 0;
    flow[((batch_idx*2+1)*height+h)*width+w] = 0;
    valid[index] = 0;
    return;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _flow(float* flow, float* valid, float* depth_src, float* depth_tgt,
          float* KT, float* Kinv, int batch_size, int height,
          int width, int device_id) {
  _set_device(device_id);

  float* depth_src_dev = NULL;
  float* depth_tgt_dev = NULL;
  float* KT_dev = NULL;
  float* Kinv_dev = NULL;
  float* flow_dev = NULL;
  float* valid_dev = NULL;

  CUDA_CHECK(cudaMalloc(&depth_src_dev,
                        batch_size * height * width * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(depth_src_dev,
                        depth_src,
                        batch_size * height * width * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&depth_tgt_dev,
                        batch_size * height * width * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(depth_tgt_dev,
                        depth_tgt,
                        batch_size * height * width * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&KT_dev,
                        batch_size * 12 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(KT_dev,
                        KT,
                        batch_size * 12 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&Kinv_dev,
                        9 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(Kinv_dev,
                        Kinv,
                        9 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&flow_dev,
                        batch_size * 2 * height * width * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&valid_dev,
                        batch_size * height * width * sizeof(float)));

  const int count = batch_size * height * width;
  flow_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, depth_src_dev, depth_tgt_dev,
                                 height, width, KT_dev, Kinv_dev, flow_dev, valid_dev);

  CUDA_CHECK(cudaMemcpy(flow,
                        flow_dev,
                        batch_size * 2 * height * width * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemcpy(valid,
                        valid_dev,
                        batch_size * 1 * height * width * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(depth_src_dev));
  CUDA_CHECK(cudaFree(depth_tgt_dev));
  CUDA_CHECK(cudaFree(KT_dev));
  CUDA_CHECK(cudaFree(Kinv_dev));
  CUDA_CHECK(cudaFree(flow_dev));
  CUDA_CHECK(cudaFree(valid_dev));
}