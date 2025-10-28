import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
# 新 import
from tensorflow.lite.python.interpreter import Interpreter



# ─── モデル変換（1度だけ実行） ─────────────────────────
# EfficientNetB0 をロードして TFLite 化
model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 代表データなしでも最適化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("efficientnet_b0_int8.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite モデルを出力しました: efficientnet_b0_int8.tflite")


# ─── 推論＋デコード ────────────────────────────────────
# インタプリタ初期化
interpreter = Interpreter(model_path="efficientnet_b0_int8.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(img_path: str, top_k=3):
    # 1. 画像読み込み＋リサイズ
    img = Image.open(img_path).convert("RGB").resize(
        (input_details[0]['shape'][2], input_details[0]['shape'][1])
    )
    # 2. 前処理 (EfficientNet 用)
    x = np.array(img, dtype=np.float32)
    x = preprocess_input(x)              # [0,255]→[0,1]
    x = np.expand_dims(x, axis=0)        # (1,224,224,3)
    # 3. 推論
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])  # (1,1000)
    # 4. Keras のデコード関数で人間向けに変換
    decoded = decode_predictions(preds, top=top_k)[0]
    return decoded

if __name__ == "__main__":
    for name, label, prob in predict_tflite("image.png", top_k=5):
        print(f"{label}: {prob:.4f}")
