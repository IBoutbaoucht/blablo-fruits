import argparse
import cv2
from src.inference import HybridAgroSystem

def main():
    parser = argparse.ArgumentParser(description="AgroScan: Hybrid Crop Disease Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    # Initialize System
    system = HybridAgroSystem()

    # Run Analysis
    print(f"\n[INFO] Analyzing image: {args.image}...")
    results = system.analyze(args.image)

    # Display Results
    print(f"\n--- 🔍 DIAGNOSIS REPORT ---")
    for i, res in enumerate(results):
        print(f"Object {i+1}:")
        print(f"  • Diagnosis: {res['label']}")
        print(f"  • Confidence: {res['confidence']:.2f}")
        print(f"  • Location: {res['box']}")
        print("-" * 30)

    print("\n[INFO] Analysis Complete.")

if __name__ == "__main__":
    main()
