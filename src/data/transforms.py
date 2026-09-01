from __future__ import annotations

from typing import Callable

from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DIFF_MEAN = (0.5, 0.5, 0.5)
DIFF_STD = (0.5, 0.5, 0.5)


def build_image_transform(size: int, clip_mean: bool = False) -> Callable:
    mean, std = (CLIP_MEAN, CLIP_STD) if clip_mean else (DIFF_MEAN, DIFF_STD)
    return transforms.Compose(
        [
            transforms.Resize(size, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def tf_input_pipeline(manifest: str, image_size: int, batch_size: int, shuffle: bool = True):
    import tensorflow as tf

    def _parse(line):
        ex = tf.io.parse_single_example(
            line,
            features={
                "image_path": tf.io.FixedLenFeature([], tf.string),
                "prompt": tf.io.FixedLenFeature([], tf.string),
                "variant_id": tf.io.FixedLenFeature([], tf.int64),
            },
        )
        img = tf.io.read_file(ex["image_path"])
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [image_size, image_size], antialias=True)
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        return img, ex["prompt"], ex["variant_id"]

    ds = tf.data.TFRecordDataset([manifest])
    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds
