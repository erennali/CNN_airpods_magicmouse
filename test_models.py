"""
Model Performance Comparison Script
Bu script üç modeli de test eder ve performanslarını karşılaştırır.
"""

import os
import time
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Model bilgileri
MODELS = {
    'Model 1 - VGG16 Transfer Learning': {
        'path': 'model1_transfer_learning.h5',
        'img_size': 224
    },
    'Model 2 - Basic CNN': {
        'path': 'model2_basic_cnn.h5',
        'img_size': 128
    },
    'Model 3 - Optimized CNN': {
        'path': 'model3_improved_cnn.h5',
        'img_size': 128
    }
}

CLASS_NAMES = ['AirPods', 'Magic Mouse']

def load_all_models():
    """Tüm modelleri yükle"""
    models = {}
    print("=" * 70)
    print("MODELLER YÜKLENİYOR...")
    print("=" * 70)
    
    for model_name, model_info in MODELS.items():
        try:
            if os.path.exists(model_info['path']):
                models[model_name] = load_model(model_info['path'])
                print(f"✅ {model_name} yüklendi")
            else:
                print(f"❌ {model_name} bulunamadı: {model_info['path']}")
        except Exception as e:
            print(f"❌ {model_name} yüklenemedi: {str(e)[:50]}")
    
    print(f"\nToplam {len(models)}/3 model yüklendi.\n")
    return models

def preprocess_image(img_path, img_size):
    """Görüntüyü modele uygun şekilde hazırla"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.LANCZOS)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, img_path, img_size):
    """Tek bir model ile tahmin yap"""
    try:
        preprocessed = preprocess_image(img_path, img_size)
        
        # İlk tahmin (warmup)
        _ = model.predict(preprocessed, verbose=0)
        
        # Gerçek tahmin (zaman ölçümü ile)
        start_time = time.time()
        predictions = model.predict(preprocessed, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return predicted_class, confidence, inference_time
    except Exception as e:
        print(f"  ❌ Hata: {str(e)[:50]}")
        return None, 0, 0

def test_single_image(models, img_path):
    """Tek bir görüntüyü tüm modellerle test et"""
    print("=" * 70)
    print(f"TEST: {os.path.basename(img_path)}")
    print("=" * 70)
    
    results = []
    
    for model_name, model in models.items():
        img_size = MODELS[model_name]['img_size']
        pred, conf, inf_time = predict_image(model, img_path, img_size)
        
        if pred:
            print(f"\n🤖 {model_name}")
            print(f"   Tahmin    : {pred}")
            print(f"   Güven     : {conf:.2f}%")
            print(f"   Süre      : {inf_time:.2f} ms")
            
            results.append({
                'model': model_name,
                'prediction': pred,
                'confidence': conf,
                'time': inf_time
            })
        else:
            print(f"\n❌ {model_name}: Tahmin yapılamadı")
    
    # En iyi sonucu bul
    if results:
        best = max(results, key=lambda x: x['confidence'])
        fastest = min(results, key=lambda x: x['time'])
        
        print("\n" + "=" * 70)
        print("ÖZET:")
        print(f"  🏆 En Yüksek Güven: {best['model']} ({best['confidence']:.2f}%)")
        print(f"  ⚡ En Hızlı Model  : {fastest['model']} ({fastest['time']:.2f} ms)")
        print("=" * 70 + "\n")

def test_directory(models, directory):
    """Bir klasördeki tüm görüntüleri test et"""
    image_files = [f for f in os.listdir(directory) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ {directory} klasöründe görüntü bulunamadı!")
        return
    
    print(f"\n📁 {directory} klasöründe {len(image_files)} görüntü bulundu.\n")
    
    for img_file in image_files[:3]:  # İlk 3 görüntüyü test et
        img_path = os.path.join(directory, img_file)
        test_single_image(models, img_path)
        time.sleep(0.5)

def compare_models_performance(models):
    """Modellerin genel performans karşılaştırması"""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 70)
    
    test_images = []
    
    # Dataset'ten test görüntüleri topla
    for class_name in ['airpods', 'magic_mouse']:
        class_dir = os.path.join('dataset', class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            test_images.extend(images[:2])  # Her sınıftan 2 görüntü
    
    if not test_images:
        print("❌ Test için görüntü bulunamadı!")
        return
    
    print(f"\n📊 {len(test_images)} görüntü ile test yapılıyor...\n")
    
    all_results = {model_name: {'times': [], 'correct': 0} 
                   for model_name in models.keys()}
    
    for img_path in test_images:
        true_label = 'AirPods' if 'airpods' in img_path else 'Magic Mouse'
        print(f"Test: {os.path.basename(img_path)} (Gerçek: {true_label})")
        
        for model_name, model in models.items():
            img_size = MODELS[model_name]['img_size']
            pred, conf, inf_time = predict_image(model, img_path, img_size)
            
            if pred:
                all_results[model_name]['times'].append(inf_time)
                if pred == true_label:
                    all_results[model_name]['correct'] += 1
                print(f"  {model_name}: {pred} ({conf:.1f}%) - {inf_time:.1f}ms")
        print()
    
    # Sonuçları özetle
    print("=" * 70)
    print("SONUÇLAR:")
    print("=" * 70)
    
    for model_name, results in all_results.items():
        if results['times']:
            avg_time = np.mean(results['times'])
            accuracy = (results['correct'] / len(test_images)) * 100
            print(f"\n🤖 {model_name}")
            print(f"   Doğruluk      : {accuracy:.1f}% ({results['correct']}/{len(test_images)})")
            print(f"   Ort. Süre     : {avg_time:.2f} ms")
            print(f"   Min/Max Süre  : {min(results['times']):.1f} / {max(results['times']):.1f} ms")
    
    print("\n" + "=" * 70)

def main():
    """Ana test fonksiyonu"""
    print("\n" + "=" * 70)
    print("CNN MODEL TEST VE KARŞILAŞTIRMA ARACI")
    print("Eren Ali Koca - 2212721021")
    print("=" * 70 + "\n")
    
    # Modelleri yükle
    models = load_all_models()
    
    if not models:
        print("❌ Hiçbir model yüklenemedi! Lütfen model dosyalarını kontrol edin.")
        return
    
    # Menü
    while True:
        print("\n" + "=" * 70)
        print("TEST SEÇENEKLERİ:")
        print("=" * 70)
        print("1. Tek görüntü test et")
        print("2. Dataset klasöründen test et (AirPods)")
        print("3. Dataset klasöründen test et (Magic Mouse)")
        print("4. Kapsamlı performans karşılaştırması")
        print("5. Root klasördeki test görüntülerini kullan")
        print("0. Çıkış")
        print("=" * 70)
        
        choice = input("\nSeçiminiz (0-5): ").strip()
        
        if choice == '1':
            img_path = input("Görüntü yolu: ").strip()
            if os.path.exists(img_path):
                test_single_image(models, img_path)
            else:
                print("❌ Dosya bulunamadı!")
                
        elif choice == '2':
            test_directory(models, 'dataset/airpods')
            
        elif choice == '3':
            test_directory(models, 'dataset/magic_mouse')
            
        elif choice == '4':
            compare_models_performance(models)
            
        elif choice == '5':
            # Root klasördeki test görüntüleri
            test_images = [f for f in os.listdir('.') 
                          if f.startswith('IMG_') and f.lower().endswith('.jpg')]
            if test_images:
                print(f"\n📁 {len(test_images)} test görüntüsü bulundu.\n")
                for img in test_images[:3]:
                    test_single_image(models, img)
            else:
                print("❌ Root klasörde test görüntüsü bulunamadı!")
                
        elif choice == '0':
            print("\n👋 Çıkılıyor...\n")
            break
        else:
            print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program kullanıcı tarafından sonlandırıldı.\n")
    except Exception as e:
        print(f"\n❌ Kritik hata: {e}")
        import traceback
        traceback.print_exc()

