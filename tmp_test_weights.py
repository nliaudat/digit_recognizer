import os
os.environ['DIGIT_INPUT_CHANNELS'] = '1'

def test_v30_teacher():
    import parameters as params
    params.NB_CLASSES = 100
    params.INPUT_CHANNELS = 3
    params.update_derived_parameters()

    from utils.train_distill import load_distillation_data
    # load_distillation_data(100, "rgb") # This crashes!

    from models.digit_recognizer_v30_teacher import create_v30_teacher
    try:
        m = create_v30_teacher(num_classes=100, input_shape=(32, 20, 3), pretrained=True)
        print("✅ RGB Loaded successfully")
    except Exception as e:
        print(f"❌ RGB Failed with: {e}")

if __name__ == '__main__':
    print("Testing minimal")
    test_v30_teacher()
