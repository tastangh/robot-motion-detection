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

ref = dosyalariOku("odev2-videolar/tusas-odev2-referans1.txt")
ogr = dosyalariOku("odev2-videolar/tusas-odev2-ogr1.txt")

yanlislar = []

for saniye in range(len(ref)):
    for robot in range(9):
        if ref[saniye][robot] != ogr[saniye][robot]:
            yanlislar.append((saniye+1, robot+1, ref[saniye][robot], ogr[saniye][robot]))

print(f"Toplam yanlış sayısı: {len(yanlislar)}")
print("Yanlışlar (Saniye, Robot, Doğru Değer, Senin Değerin):")
for y in yanlislar:
    print(f"Saniye {y[0]:02d}, Robot-{y[1]} → Doğru: {y[2]} Senin: {y[3]}")
