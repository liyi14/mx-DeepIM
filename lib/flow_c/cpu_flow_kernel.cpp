// ------------------------------------------------------------------
// Deep Iterative Matching Network
// ------------------------------------------------------------------

#include "cpu_flow.hpp"
#include <iostream>

void flow_kernel(const int nthreads,
                           const float *depth_src, const float *depth_tgt,
                           int height, int width,
                           float* KT, float* Kinv,
                           float* flow) {
  for (int index = 0; index < nthreads; index++) {
    int w = index % width;
    int h = (index / width) % height;
    int batch_idx = index / width / height;
    float d_src = depth_src[index];
    float d_tgt = depth_tgt[index];
    float x = (w*Kinv[0] + h*Kinv[1] + Kinv[2])*d_src;
    float y = (w*Kinv[3] + h*Kinv[4] + Kinv[5])*d_src;
    float z = d_src;
    KT += batch_idx * 12;
    float x_proj = x*KT[0] + y*KT[1] + z*KT[2] + KT[3];
    float y_proj = x*KT[4] + y*KT[5] + z*KT[6] + KT[7];
    float z_proj = x*KT[8] + y*KT[9] + z*KT[10] + KT[11];
    float w_proj = x_proj / z_proj;
    float h_proj = y_proj / z_proj;
    if (d_src > 1E-3 && (d_src - d_tgt) < 3E-3) {
        flow[((batch_idx*2+0)*height+h)*width+w] = h_proj-h;
        flow[((batch_idx*2+1)*height+h)*width+w] = w_proj-w;
    }
    else {
        flow[((batch_idx*2+0)*height+h)*width+w] = 0;
        flow[((batch_idx*2+1)*height+h)*width+w] = 0;
    }
  }
}

void flow_cpp(float* flow_gt, float* depth_src, float* depth_tgt,
          float* KT, float* Kinv, int batch_size, int height,
          int width, int device_id) {
  const int count = batch_size * height * width;
  flow_kernel(count, depth_src, depth_tgt,
                                 height, width, KT, Kinv, flow_gt);
}