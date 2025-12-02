from ctranslate2.converters import TransformersConverter

print("Converting model...")
converter = TransformersConverter("vinai/PhoWhisper-medium")
converter.convert("PhoWhisper-medium-ct2", quantization="float16")
print("Conversion complete.")
