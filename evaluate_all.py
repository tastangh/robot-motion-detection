import os
import subprocess

def dosyalariOku(filename):
    matrix = []
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            if parts:
                row = list(map(int, parts[1:]))
                matrix.append(row)
    return matrix

def sonuclariKarsilastir(matrix1, matrix2):
    same_count = 0
    total = 0
    for row1, row2 in zip(matrix1, matrix2):
        for val1, val2 in zip(row1, row2):
            total += 1
            if val1 == val2:
                same_count += 1
    return same_count, total

# =======================
# Ayarlar
# =======================
base_dir = "odev2-videolar"
robot_script = "robot_motion_detection.py"

for i in range(1, 5):
    video = f"{base_dir}/tusas-odev2-test{i}.mp4"
    output_txt = f"{base_dir}/tusas-odev2-ogr{i}.txt"
    reference_txt = f"{base_dir}/tusas-odev2-referans{i}.txt"

    print(f"\n[INFO] Video {i} işleniyor...")

    # Kodun içinde değişken olarak VIDEO_PATH ve OUTPUT_TXT tanımlı olduğu için subprocess ile argüman aktarmak yerine, bunları dışarıdan değiştirmek gerekebilir.
    # Alternatif olarak: `robot_motion_detection.py`'yi parametre alabilir hale getiririz.
    os.environ["VIDEO_PATH"] = video
    os.environ["OUTPUT_TXT"] = output_txt

    # robot_motion_detection.py aynı klasörde olmalı
    subprocess.run(["python", robot_script], check=True)

    sonuc_ref = dosyalariOku(reference_txt)
    sonuc_ogr = dosyalariOku(output_txt)
    dogruSay, toplam = sonuclariKarsilastir(sonuc_ref, sonuc_ogr)

    print(f"[SKOR] tusas-odev2-test{i}.mp4 için doğruluk: {dogruSay} / {toplam} ({dogruSay / toplam * 100:.2f}%)")
