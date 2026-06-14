"""Quick test: run predict_esp32 on a re-exported v23 model to verify uint8 I/O path works."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark.predictor import TFLiteDigitPredictor

model_path = r"exported_models\10cls_RGB\digit_recognizer_v23_10cls_RGB_TQT_SOFTMAX_0606_1437\v23_full_integer_quant.tflite"

print(f"Loading {model_path}...")
predictor = TFLiteDigitPredictor(model_path)

inp = predictor.input_details[0]
out = predictor.output_details[0]
print(f"Input:  dtype={inp['dtype']}, shape={inp['shape']}")
print(f"Output: dtype={out['dtype']}, shape={out['shape']}")
if inp['dtype'] != np.uint8 or out['dtype'] != np.uint8:
    print("❌ Not uint8 I/O — wrong model!")
    sys.exit(1)

# Create synthetic test image (random uint8 RGB)
test_img = np.random.randint(0, 256, (32, 20, 3), dtype=np.uint8)

# Run both predict and predict_esp32
pred, conf, vec = predictor.predict(test_img)
print(f"\npredict()        → digit={pred}, conf={conf:.4f}")

pred_esp, conf_esp, vec_esp = predictor.predict_esp32(test_img)
print(f"predict_esp32()  → digit={pred_esp}, conf={conf_esp:.4f}")

# Run 5 more ESP32 simulations to see noise impact
confs = []
preds = []
for i in range(5):
    p, c, _ = predictor.predict_esp32(test_img)
    preds.append(p)
    confs.append(c)
print(f"\n5x ESP32 sims: preds={preds}, confs={[f'{c:.4f}' for c in confs]}")

print("\n✅ predict_esp32 works with uint8 I/O!")