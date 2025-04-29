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
    return same_count

dosya1 = "tusas-odev2-referans.txt"
dosya2 = "tusas-odev2-ogr.txt"

sonuc_ref = dosyalariOku(dosya1)
sonuc_ogr = dosyalariOku(dosya2)

dogruSay = sonuclariKarsilastir(sonuc_ref, sonuc_ogr)

print("Skor: " + str(dogruSay) + "/540")