# BLM5103 - Bilgisayarla Görme Ödevi 2

Bu proje, verilen videolardaki 9 robotun 60 saniyelik süre boyunca hangi saniyelerde hareket ettiğini tespit eder. Sistem, homografi ile görüntüyü düzleştirir, Harris köşe tespiti ile hareket analizini yapar ve saniyelik hareket bilgilerini `.txt` dosyasına yazar.

##  Dosya Yapısı

```
.
├── robot_motion_detection.py   # Ana Python kodu
├── odev2-videolar/             # Videolar ve referans txt dosyaları
│   ├── tusas-odev2-test1.mp4
│   ├── tusas-odev2-referans1.txt
│   └── ...
├── requirements.txt            # Kütüphane gereksinimleri
└── README.md                   # Bu belge
```

##  Gereksinimler

Python 3.6 veya üzeri kurulu olmalıdır. Projede kullanılan temel kütüphaneler aşağıdaki gibidir:

- `opencv-python`
- `numpy`

Kurulumu aşağıdaki komutla gerçekleştirebilirsiniz:

```bash
pip install -r requirements.txt
```

##  requirements.txt

```
opencv-python
numpy
```

## ▶Kullanım

Aşağıdaki gibi çalıştırabilirsiniz:

```bash
python robot_motion_detection.py
```

> Not: `VIDEO_PATH` ve `OUTPUT_TXT` ortam değişkenleriyle video ve çıktı yolu ayarlanabilir. Varsayılan olarak `odev2-videolar/tusas-odev2-test1.mp4` işlenir ve `tusas-odev2-ogr1.txt` çıktısı üretilir.

## Çıktı Formatı

Çıktı dosyası aşağıdaki gibi yapılandırılmıştır:

```
Saniye	Robot-1	Robot-2	...	Robot-9
1    	0    	1    	...	0
2    	0    	0    	...	1
...
60   	1    	0    	...	0
```

##  Değerlendirme

Oluşan `txt` dosyası, referans dosyayla `odev2-kontrol.py` gibi karşılaştırıcı araçlarla test edilmelidir. Bu sayede doğru tahmin sayısı belirlenir (maksimum 540).

Herhangi bir sorunda `robot_motion_detection.py` dosyasını doğrudan çalıştırarak video işleyebilir ve görselleştirmeyi inceleyebilirsiniz. Tüm hareket eden robotlar yeşil çerçeveyle gösterilecektir.
