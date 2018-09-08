void flow_cpp(float* flow_gt, float* depth_src, float* depth_tgt,
           float* KT, float* Kinv, int batch_size, int height,
           int width, int device_id);