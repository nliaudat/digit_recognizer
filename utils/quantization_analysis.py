# utils/quantization_analysis.py
"""
Comprehensive quantization analysis tools for evaluating:
- Quantization impact on accuracy
- Model size reduction
- Performance comparison between float and quantized models
- QAT effectiveness analysis
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

import parameters as params

class QuantizationAnalyzer:
    """Comprehensive analyzer for quantization impact assessment"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.analysis_results = {}
    
    def analyze_quantization_impact(self, keras_model, tflite_model_path: str, 
                                  x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive analysis of quantization impact
        """
        print("\n🔍 ANALYZING QUANTIZATION IMPACT")
        print("=" * 50)
        
        results = {}
        
        # 1. Evaluate original Keras model
        keras_accuracy = self._evaluate_keras_model(keras_model, x_test, y_test)
        results['keras_accuracy'] = keras_accuracy
        
        # 2. Evaluate quantized TFLite model
        tflite_accuracy = self._evaluate_tflite_model(tflite_model_path, x_test, y_test)
        results['tflite_accuracy'] = tflite_accuracy
        
        # 3. Calculate accuracy drop
        accuracy_drop = keras_accuracy - tflite_accuracy
        results['accuracy_drop'] = accuracy_drop
        
        # 4. Model size analysis
        size_analysis = self._analyze_model_sizes(keras_model, tflite_model_path)
        results.update(size_analysis)
        
        # 5. Performance analysis
        perf_analysis = self._analyze_performance(keras_model, tflite_model_path, x_test)
        results.update(perf_analysis)
        
        # 6. Structural Validation (Verifying scales/zero-points)
        structural_results = validate_tflite_structural_quantization(tflite_model_path, verbose=self.debug)
        results['structural_quantization'] = structural_results
        
        # 7. QAT effectiveness (if applicable)
        if params.USE_QAT:
            qat_effectiveness = self._analyze_qat_effectiveness(accuracy_drop)
            results.update(qat_effectiveness)
        
        self.analysis_results = results
        self._print_analysis_summary(results)
        
        return results
    
    def _evaluate_keras_model(self, model, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate Keras model accuracy"""
        try:
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            if self.debug:
                print(f"Keras model accuracy: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            print(f"X Keras model evaluation failed: {e}")
            return 0.0
    
    def _evaluate_tflite_model(self, tflite_path: str, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate TFLite model accuracy"""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            correct_predictions = 0
            total_samples = len(x_test)
            
            for i in range(total_samples):
                # Prepare input
                input_data = x_test[i:i+1].astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(output_data)
                
                # Compare with ground truth
                if params.MODEL_ARCHITECTURE == "original_haverland":
                    true_class = np.argmax(y_test[i])
                else:
                    true_class = y_test[i]
                
                if predicted_class == true_class:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_samples
            
            if self.debug:
                print(f"TFLite model accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            print(f"X TFLite model evaluation failed: {e}")
            return 0.0
    
    def _analyze_model_sizes(self, keras_model, tflite_path: str) -> Dict:
        """Analyze model size reduction from quantization"""
        # Get Keras model size (approximate)
        try:
            # Save temporary Keras model to get size
            temp_path = "temp_keras_model.keras"
            keras_model.save(temp_path)
            keras_size = os.path.getsize(temp_path) / 1024  # KB
            os.remove(temp_path)
        except:
            keras_size = 0
        
        # Get TFLite model size
        tflite_size = os.path.getsize(tflite_path) / 1024  # KB
        
        size_reduction = ((keras_size - tflite_size) / keras_size) * 100 if keras_size > 0 else 0
        
        if self.debug:
            print(f"📊 Model sizes - Keras: {keras_size:.1f} KB, TFLite: {tflite_size:.1f} KB")
            print(f"📉 Size reduction: {size_reduction:.1f}%")
        
        return {
            'keras_size_kb': keras_size,
            'tflite_size_kb': tflite_size,
            'size_reduction_percent': size_reduction
        }
    
    def _analyze_performance(self, keras_model, tflite_path: str, x_test: np.ndarray) -> Dict:
        """Compare inference performance between Keras and TFLite"""
        performance_results = {}
        
        # Keras inference time
        try:
            sample_data = x_test[:10]  # Use small sample for timing
            start_time = tf.timestamp()
            _ = keras_model.predict(sample_data, verbose=0)
            end_time = tf.timestamp()
            keras_time = (end_time - start_time).numpy() * 1000  # ms
            performance_results['keras_inference_ms'] = keras_time
        except Exception as e:
            print(f"!  Keras performance test failed: {e}")
            performance_results['keras_inference_ms'] = 0
        
        # TFLite inference time
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            
            sample_data = x_test[:10].astype(input_details[0]['dtype'])
            
            start_time = tf.timestamp()
            for i in range(len(sample_data)):
                interpreter.set_tensor(input_details[0]['index'], sample_data[i:i+1])
                interpreter.invoke()
            end_time = tf.timestamp()
            tflite_time = (end_time - start_time).numpy() * 1000  # ms
            performance_results['tflite_inference_ms'] = tflite_time
            
            # Calculate speedup
            if performance_results['keras_inference_ms'] > 0:
                speedup = performance_results['keras_inference_ms'] / tflite_time
                performance_results['inference_speedup'] = speedup
            else:
                performance_results['inference_speedup'] = 0
                
        except Exception as e:
            print(f"!  TFLite performance test failed: {e}")
            performance_results['tflite_inference_ms'] = 0
            performance_results['inference_speedup'] = 0
        
        return performance_results
    
    def _analyze_qat_effectiveness(self, accuracy_drop: float) -> Dict:
        """Analyze QAT training effectiveness"""
        effectiveness = {}
        
        if accuracy_drop < 0.01:
            effectiveness['qat_effectiveness'] = "Excellent"
            effectiveness['qat_rating'] = 5
        elif accuracy_drop < 0.02:
            effectiveness['qat_effectiveness'] = "Good"
            effectiveness['qat_rating'] = 4
        elif accuracy_drop < 0.05:
            effectiveness['qat_effectiveness'] = "Acceptable"
            effectiveness['qat_rating'] = 3
        else:
            effectiveness['qat_effectiveness'] = "Poor"
            effectiveness['qat_rating'] = 2
        
        effectiveness['qat_accuracy_drop'] = accuracy_drop
        
        return effectiveness
    
    def _print_analysis_summary(self, results: Dict):
        """Print comprehensive quantization analysis summary"""
        print("\n" + "="*60)
        print("📊 QUANTIZATION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"🎯 Accuracy Analysis:")
        print(f"   Keras Model:     {results.get('keras_accuracy', 0):.4f}")
        print(f"   TFLite Model:    {results.get('tflite_accuracy', 0):.4f}")
        print(f"   Accuracy Drop:   {results.get('accuracy_drop', 0):.4f}")
        
        print(f"\n📦 Size Analysis:")
        print(f"   Keras Size:      {results.get('keras_size_kb', 0):.1f} KB")
        print(f"   TFLite Size:     {results.get('tflite_size_kb', 0):.1f} KB")
        print(f"   Size Reduction:  {results.get('size_reduction_percent', 0):.1f}%")
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   Keras Inference: {results.get('keras_inference_ms', 0):.2f} ms")
        print(f"   TFLite Inference: {results.get('tflite_inference_ms', 0):.2f} ms")
        print(f"   Speedup:         {results.get('inference_speedup', 0):.1f}x")
        
        structural = results.get('structural_quantization', {})
        if structural:
            print(f"\n🧩 Structural Validation:")
            print(f"   Level:           {structural.get('quantization_level', 'Unknown')}")
            print(f"   Tensors Scanned: {structural.get('tensors_scanned', 0)}")
            print(f"   Validated:       {'✅ YES' if structural.get('is_valid', False) else '❌ NO'}")

        if params.USE_QAT:
            print(f"\n🎯 QAT Effectiveness:")
            print(f"   Rating:          {results.get('qat_effectiveness', 'N/A')}")
            print(f"   Accuracy Drop:   {results.get('qat_accuracy_drop', 0):.4f}")
        
        # Overall quantization success assessment
        accuracy_drop = results.get('accuracy_drop', 1.0)
        size_reduction = results.get('size_reduction_percent', 0)
        
        if accuracy_drop < 0.02 and size_reduction > 50:
            print(f"\n QUANTIZATION: EXCELLENT SUCCESS!")
        elif accuracy_drop < 0.05 and size_reduction > 30:
            print(f"\n QUANTIZATION: GOOD SUCCESS!")
        elif accuracy_drop < 0.1:
            print(f"\n!  QUANTIZATION: ACCEPTABLE")
        else:
            print(f"\nX QUANTIZATION: POOR RESULTS")
    
    def generate_quantization_report(self, output_dir: str):
        """Generate detailed quantization analysis report"""
        if not self.analysis_results:
            print("X No analysis results available. Run analyze_quantization_impact first.")
            return
        
        report_path = os.path.join(output_dir, "quantization_analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Quantization Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("QUANTIZATION SETTINGS:\n")
            f.write(f"  Quantization Enabled: {params.QUANTIZE_MODEL}\n")
            f.write(f"  QAT Enabled: {params.USE_QAT}\n")
            f.write(f"  ESP-DL Quantization: {params.ESP_DL_QUANTIZE}\n")
            f.write(f"  Calibration Samples: {params.QUANTIZE_NUM_SAMPLES}\n\n")
            
            f.write("ACCURACY ANALYSIS:\n")
            f.write(f"  Keras Model Accuracy: {self.analysis_results.get('keras_accuracy', 0):.4f}\n")
            f.write(f"  TFLite Model Accuracy: {self.analysis_results.get('tflite_accuracy', 0):.4f}\n")
            f.write(f"  Accuracy Drop: {self.analysis_results.get('accuracy_drop', 0):.4f}\n\n")
            
            f.write("SIZE ANALYSIS:\n")
            f.write(f"  Keras Model Size: {self.analysis_results.get('keras_size_kb', 0):.1f} KB\n")
            f.write(f"  TFLite Model Size: {self.analysis_results.get('tflite_size_kb', 0):.1f} KB\n")
            f.write(f"  Size Reduction: {self.analysis_results.get('size_reduction_percent', 0):.1f}%\n\n")
            
            f.write("STRUCTURAL VALIDATION:\n")
            structural = self.analysis_results.get('structural_quantization', {})
            f.write(f"  Quantization Level: {structural.get('quantization_level', 'N/A')}\n")
            f.write(f"  Is Valid Quantized Model: {structural.get('is_valid', False)}\n")
            f.write(f"  Tensors with Scales: {structural.get('tensors_with_scales', 0)}\n\n")

            f.write("PERFORMANCE ANALYSIS:\n")
            f.write(f"  Keras Inference Time: {self.analysis_results.get('keras_inference_ms', 0):.2f} ms\n")
            f.write(f"  TFLite Inference Time: {self.analysis_results.get('tflite_inference_ms', 0):.2f} ms\n")
            f.write(f"  Inference Speedup: {self.analysis_results.get('inference_speedup', 0):.1f}x\n\n")
            
            if params.USE_QAT:
                f.write("QAT EFFECTIVENESS:\n")
                f.write(f"  Effectiveness: {self.analysis_results.get('qat_effectiveness', 'N/A')}\n")
                f.write(f"  Accuracy Drop: {self.analysis_results.get('qat_accuracy_drop', 0):.4f}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            accuracy_drop = self.analysis_results.get('accuracy_drop', 1.0)
            if accuracy_drop > 0.1:
                f.write("  X Consider disabling quantization or improving QAT training\n")
            elif accuracy_drop > 0.05:
                f.write("  !  Quantization acceptable but consider QAT for better results\n")
            else:
                f.write("   Quantization successful - good balance of size and accuracy\n")
        
        print(f" Quantization analysis report saved: {report_path}")


# ---------------------------------------------------------------------------
# TFLite Structural Validation: Verify Scales and Zero-Points
# ---------------------------------------------------------------------------

def validate_tflite_structural_quantization(tflite_path: str, verbose: bool = False) -> Dict:
    """
    Scans internal TFLite tensors to verify if quantization parameters (scales/zero points) 
    are present and meaningful.
    """
    results = {
        'is_valid': False,
        'tensors_scanned': 0,
        'tensors_with_scales': 0,
        'quantization_level': 'Float'
    }
    
    try:
        if not os.path.exists(tflite_path):
            return results

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        details = interpreter.get_tensor_details()
        results['tensors_scanned'] = len(details)
        
        scaled_tensors = 0
        int8_tensors = 0
        
        if verbose:
            print(f"\n--- TFLite Structural Scan: {os.path.basename(tflite_path)} ---")
            
        for detail in details:
            q_params = detail.get('quantization_parameters')
            if q_params and len(q_params.get('scales', [])) > 0:
                scales = q_params['scales']
                # Check for non-trivial scales
                if any(s != 0.0 and s != 1.0 for s in scales):
                    scaled_tensors += 1
                    if detail['dtype'] in [np.int8, np.uint8, np.int32]:
                        int8_tensors += 1
                    
                    if verbose and scaled_tensors < 10: # Print first few
                        print(f"  Valid Tensor found: {detail['name']}")
                        print(f"    - Scale[0]: {scales[0]:.6f}")
                        print(f"    - Dtype: {detail['dtype']}")

        results['tensors_with_scales'] = scaled_tensors
        
        if scaled_tensors > 5: # Threshold for a "quantized model"
            results['is_valid'] = True
            if int8_tensors > (scaled_tensors * 0.5):
                results['quantization_level'] = 'INT8/Fully Quantized'
            else:
                results['quantization_level'] = 'Hybrid/Partial'
        
        return results

    except Exception as e:
        if verbose: print(f"Structural validation error: {e}")
        return results


# ---------------------------------------------------------------------------
# TQT-specific comparison: float ONNX vs TQT-quantized graph
# ---------------------------------------------------------------------------

def compare_float_vs_tqt(
    onnx_path: str,
    quant_ppq_graph,
    x_val,
    y_val,
):
    import numpy as np
    from utils.preprocess import preprocess_for_inference

    print("\n" + "=" * 60)
    print("📊 FLOAT ONNX vs TQT QUANTIZED -- COMPARISON")
    print("=" * 60)

    float_preds = None
    float_accuracy = 0.0
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)
        iname = session.get_inputs()[0].name
        x_f = preprocess_for_inference(x_val).astype("float32").transpose(0, 3, 1, 2)
        float_preds = np.array([
            session.run(None, {iname: x_f[i:i+1]})[0][0] for i in range(len(x_f))
        ])
        float_accuracy = float(np.mean(np.argmax(float_preds, axis=1) == y_val))
        print(f"   Float ONNX accuracy  : {float_accuracy:.4f} ({float_accuracy*100:.2f}%)")
    except ImportError:
        print("⚠️  onnxruntime not installed -- skipping float evaluation")
    except Exception as exc:
        print(f"⚠️  Float ONNX inference failed: {exc}")

    tqt_preds = None
    tqt_accuracy = 0.0
    try:
        import torch
        from esp_ppq.api import PPQTorchExecutor
        x_f = preprocess_for_inference(x_val).astype("float32").transpose(0, 3, 1, 2)
        x_t = torch.from_numpy(x_f)
        executor = PPQTorchExecutor(quant_ppq_graph, device="cpu")
        tqt_out = []
        for i in range(len(x_t)):
            with torch.no_grad():
                out = executor.forward(x_t[i:i+1])
            logits = out[0] if isinstance(out, (list, tuple)) else out
            tqt_out.append(logits.squeeze().cpu().numpy())
        tqt_preds = np.array(tqt_out)
        tqt_accuracy = float(np.mean(np.argmax(tqt_preds, axis=1) == y_val))
        print(f"   TQT quantized accuracy: {tqt_accuracy:.4f} ({tqt_accuracy*100:.2f}%)")
    except ImportError:
        print("⚠️  esp_ppq or torch not installed -- skipping TQT evaluation")
    except Exception as exc:
        print(f"⚠️  TQT graph inference failed: {exc}")

    mean_mse = float("nan")
    if float_preds is not None and tqt_preds is not None:
        n = min(len(float_preds), len(tqt_preds))
        mean_mse = float(np.mean(np.mean((float_preds[:n] - tqt_preds[:n]) ** 2, axis=1)))
        print(f"   Mean output MSE (float vs TQT): {mean_mse:.6f}")

    delta = tqt_accuracy - float_accuracy
    sign = "+" if delta >= 0 else ""
    print(f"   Accuracy delta (TQT - float): {sign}{delta*100:.2f}pp")
    if delta >= 0:
        print("✅ TQT improved or matched float accuracy after quantization.")
    else:
        print(f"⚠️  TQT dropped {abs(delta)*100:.2f}pp. Try lower TQT_LR.")

    return {
        "float_accuracy": float_accuracy,
        "tqt_accuracy": tqt_accuracy,
        "accuracy_delta": delta,
        "mean_output_mse": mean_mse,
    }


# Convenience function for easy usage
def analyze_quantization_impact(keras_model, tflite_model_path: str, 
                              x_test: np.ndarray, y_test: np.ndarray, 
                              debug: bool = False) -> Dict:
    """Convenience function for quick quantization analysis"""
    analyzer = QuantizationAnalyzer(debug=debug)
    return analyzer.analyze_quantization_impact(keras_model, tflite_model_path, x_test, y_test)