"""
DL model service — loads the CNN+LSTM+Attention model, scaler, and metadata once at startup.
Exposes: predict(ticker) → PredictionResult
"""
import os, pickle, logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "dl")
ARTIFACTS = os.path.abspath(ARTIFACTS)

_model   = None
_scaler  = None
_meta    = None
_results = None
_history = None


def _register_custom_layers():
    """Register TemporalAttention so Keras can deserialize the saved model."""
    import tensorflow as tf
    from tensorflow.keras.layers import Layer

    @tf.keras.utils.register_keras_serializable()
    class TemporalAttention(Layer):
        """Bahdanau-style temporal attention (must match training definition exactly)."""

        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.W = self.add_weight(
                name="attn_W", shape=(input_shape[-1], self.units),
                initializer="glorot_uniform", trainable=True,
            )
            self.b = self.add_weight(
                name="attn_b", shape=(self.units,),
                initializer="zeros", trainable=True,
            )
            self.v = self.add_weight(
                name="attn_v", shape=(self.units,),
                initializer="glorot_uniform", trainable=True,
            )

        def call(self, inputs):
            score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            attention_weights = tf.nn.softmax(
                tf.tensordot(score, self.v, axes=[[-1], [0]]), axis=1
            )
            context = tf.reduce_sum(
                inputs * tf.expand_dims(attention_weights, -1), axis=1
            )
            return context

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config


def _load():
    global _model, _scaler, _meta, _results, _history
    if _model is not None:
        return

    import tensorflow as tf  # lazy import — heavy
    _register_custom_layers()

    _model = tf.keras.models.load_model(
        os.path.join(ARTIFACTS, "final_model.keras"), compile=False
    )
    with open(os.path.join(ARTIFACTS, "scaler.pkl"), "rb") as f:
        _scaler = pickle.load(f)
    with open(os.path.join(ARTIFACTS, "preprocessing_meta.pkl"), "rb") as f:
        _meta = pickle.load(f)
    with open(os.path.join(ARTIFACTS, "final_results.pkl"), "rb") as f:
        _results = pickle.load(f)
    _history = pd.read_csv(os.path.join(ARTIFACTS, "training_history.csv"))
    logger.info("DL model + artifacts loaded.")


def get_final_results() -> dict:
    _load()
    return _results


def get_training_history() -> list[dict]:
    _load()
    return _history.to_dict(orient="records")


def get_feature_importance() -> list[dict]:
    _load()
    importance_path = os.path.join(ARTIFACTS, "feature_importance.csv")
    df = pd.read_csv(importance_path)
    return df.to_dict(orient="records")


def predict_from_ohlcv(df_ohlcv: pd.DataFrame) -> dict:
    """
    df_ohlcv: DataFrame with open/high/low/close/volume columns, DatetimeIndex, sorted oldest→newest.
    Returns: {'prediction': 'UP'|'DOWN', 'probability': float, 'confidence': 'Low'|'Medium'|'High', 'threshold': float}
    """
    from app.services.feature_service import compute_dl_features, build_window

    _load()

    features = compute_dl_features(df_ohlcv)
    features = features.dropna()

    window_size = _meta["window_size"]  # 30
    X = build_window(features, window_size)

    # Scale using the saved MinMaxScaler (fitted on training set)
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = _scaler.transform(X_flat).reshape(X.shape)
    X_scaled = X_scaled.astype(np.float32)

    prob = float(_model.predict(X_scaled, verbose=0)[0][0])
    threshold = float(_results["final_threshold"])  # ≈ 0.325
    prediction = "UP" if prob >= threshold else "DOWN"

    diff = abs(prob - 0.5)
    if diff < 0.10:
        confidence = "Low"
    elif diff < 0.20:
        confidence = "Medium"
    else:
        confidence = "High"

    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "confidence": confidence,
        "threshold": round(threshold, 4),
    }
