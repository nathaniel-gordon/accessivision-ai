# Deployment

The target deployment is on-device inference on Apple Neural Engine (ANE) via
CoreML. `src/deployment/coreml_export.py` handles graph conversion; INT8
palettisation with block size 32 gives ~2x size reduction with <1% quality
loss and ~480 ms latency on iPhone 15 Pro.

TFLite export path is also available for Android targets but is not the
primary deployment path.
