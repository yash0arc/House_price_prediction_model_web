import tensorflow as tf

print("CUDA built:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices('GPU'))
