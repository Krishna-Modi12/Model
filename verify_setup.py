import torch
import cv2
import numpy as np

def verify_pytorch_stack():
    print("=" * 60)
    print("PYTORCH DEEP LEARNING STACK VERIFICATION")
    print("=" * 60)
    
    # Check PyTorch & GPU
    print("\n--- Core Engine ---")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_status = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_status}")
    if cuda_status:
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Test a simple tensor operation on GPU
        x = torch.rand(5, 3).cuda()
        print("GPU Tensor Test: Success")
    else:
        print("CRITICAL: GPU NOT DETECTED BY PYTORCH")

    # Check Libraries
    print("\n--- Library Connectivity ---")
    libraries = [
        ("timm", "timm"),
        ("pytorch_lightning", "Lightning"),
        ("albumentations", "Albumentations"),
        ("facenet_pytorch", "FaceNet-PyTorch"),
        ("insightface", "InsightFace"),
        ("mediapipe", "MediaPipe"),
        ("sklearn", "Scikit-Learn"),
        ("colormath", "ColorMath"),
        ("mlflow", "MLflow")
    ]
    
    for module, name in libraries:
        try:
            __import__(module)
            print(f"{name:<20} [OK]")
        except ImportError:
            print(f"{name:<20} [MISSING]")

    # Specific Stack Tests
    print("\n--- Functional Verification ---")
    try:
        import timm
        m = timm.create_model('efficientnet_b0', pretrained=False)
        print("TIMM Model Creation:   [OK]")
    except Exception as e:
        print(f"TIMM Model Creation:   [FAILED] ({e})")

    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh()
        print("MediaPipe FaceMesh:    [OK]")
    except Exception as e:
        print(f"MediaPipe FaceMesh:    [FAILED] ({e})")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    verify_pytorch_stack()
