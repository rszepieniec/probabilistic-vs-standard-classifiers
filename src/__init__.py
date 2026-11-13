import os

# TensorFlow Probability działa z Keras 2 (tf-keras). Musi być ustawione przed importem tensorflow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
