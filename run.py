"""
Simple script to run the Lab Report Digitization system
"""
import os
import sys

def check_dependencies():
    try:
        import cv2  # noqa
        import pytesseract  # noqa
        from pdf2image import convert_from_path  # noqa
        import fastapi  # noqa
        print("✅ All Python dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    dirs = [
        "data/input", "data/processed", "data/corrections",
        "data/training_data", "outputs", "models", "static", "modules"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Created necessary directories")

def main():
    print("Lab Report Digitization System")
    print("=" * 40)

    if not check_dependencies():
        return 1

    create_directories()

    print("🚀 Starting FastAPI server...")
    print("📱 Web interface: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")

    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
