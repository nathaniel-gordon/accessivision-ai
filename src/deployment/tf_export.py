from __future__ import annotations

from pathlib import Path
from typing import Callable


def export_tflite(tf_model_fn: Callable, out_path: str, quantize: bool = True) -> Path:
    import tensorflow as tf

    model = tf_model_fn()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite = converter.convert()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite)
    return out
