import cv2
from inference_sdk import InferenceHTTPClient

# API ve model bilgileri
API_URL = "https://detect.roboflow.com"
API_KEY = "Gs0X6B7FVqulAniaOzPh"
MODEL_ID = "turk-isaret-dili/2"

# Inference HTTP Client oluştur
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

def infer_image(image):
    try:
        # İnferans yap
        result = client.infer(image, model_id=MODEL_ID)
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def main():
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Görüntüyü işaret dili modeline gönder
        result = infer_image(frame)
        if result is not None:
            # Sonuçları ekrana yazdır
            print(result)
        else:
            print("Inference failed.")

        # Görüntüyü ekrana yansıt
        cv2.imshow('Camera', frame)

        # 'q' tuşuna basıldığında döngüyü kır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()