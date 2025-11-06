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
        
        # 6. QAT effectiveness (if applicable)
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
        
        with open(report_path, 'w') as f:
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

# Convenience function for easy usage
def analyze_quantization_impact(keras_model, tflite_model_path: str, 
                              x_test: np.ndarray, y_test: np.ndarray, 
                              debug: bool = False) -> Dict:
    """Convenience function for quick quantization analysis"""
    analyzer = QuantizationAnalyzer(debug=debug)
    return analyzer.analyze_quantization_impact(keras_model, tflite_model_path, x_test, y_test)