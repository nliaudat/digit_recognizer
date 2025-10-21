import tensorflow as tf
import parameters as params

def build_augmentation_pipeline():
    """Create a tf.keras.Sequential augmentation pipeline based on parameters.py"""
    layers = []
    # Rotation
    if getattr(params, 'AUGMENTATION_ROTATION_RANGE', 0) > 0:
        rotation_factor = params.AUGMENTATION_ROTATION_RANGE / 360.0
        layers.append(tf.keras.layers.RandomRotation(
            factor=rotation_factor,
            fill_mode='nearest',
            name='random_rotation'
        ))
    # Translation
    if getattr(params, 'AUGMENTATION_WIDTH_SHIFT_RANGE', 0) > 0 or getattr(params, 'AUGMENTATION_HEIGHT_SHIFT_RANGE', 0) > 0:
        layers.append(tf.keras.layers.RandomTranslation(
            height_factor=getattr(params, 'AUGMENTATION_HEIGHT_SHIFT_RANGE', 0),
            width_factor=getattr(params, 'AUGMENTATION_WIDTH_SHIFT_RANGE', 0),
            fill_mode='nearest',
            name='random_translation'
        ))
    # Zoom
    if getattr(params, 'AUGMENTATION_ZOOM_RANGE', 0) > 0:
        layers.append(tf.keras.layers.RandomZoom(
            height_factor=params.AUGMENTATION_ZOOM_RANGE,
            width_factor=params.AUGMENTATION_ZOOM_RANGE,
            fill_mode='nearest',
            name='random_zoom'
        ))
    # Brightness
    if getattr(params, 'AUGMENTATION_BRIGHTNESS_RANGE', [1.0, 1.0]) != [1.0, 1.0]:
        min_delta = params.AUGMENTATION_BRIGHTNESS_RANGE[0] - 1.0
        max_delta = params.AUGMENTATION_BRIGHTNESS_RANGE[1] - 1.0
        layers.append(tf.keras.layers.RandomBrightness(
            factor=(min_delta, max_delta),
            value_range=(0, 1),
            name='random_brightness'
        ))
    # Contrast
    layers.append(tf.keras.layers.RandomContrast(
        factor=0.1,
        name='random_contrast'
    ))
    # Flips
    if getattr(params, 'AUGMENTATION_HORIZONTAL_FLIP', False):
        layers.append(tf.keras.layers.RandomFlip(
            mode='horizontal',
            name='random_horizontal_flip'
        ))
    if getattr(params, 'AUGMENTATION_VERTICAL_FLIP', False):
        layers.append(tf.keras.layers.RandomFlip(
            mode='vertical',
            name='random_vertical_flip'
        ))
    return tf.keras.Sequential(layers, name='augmentation_pipeline')
