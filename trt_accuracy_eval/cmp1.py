from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Conversion Parameters 
conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP32)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    conversion_params=conversion_params)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Save the model to the disk 
converter.save(output_saved_model_dir)