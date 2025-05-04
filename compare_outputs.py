def parse_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
        data = []
        for line in lines:
            parts = line.strip().split()
            row = [int(x) for x in parts[1:]]  # Skip "1)", "2)", etc.
            data.append(row)
        return data

def compare_outputs(ref_path, student_path):
    ref = parse_file(ref_path)
    student = parse_file(student_path)

    total = 0
    correct = 0
    errors = []

    for sec in range(len(ref)):
        for robot in range(9):
            total += 1
            r_val = ref[sec][robot]
            s_val = student[sec][robot]
            if r_val == s_val:
                correct += 1
            else:
                errors.append(f"[HATA] Saniye {sec+1}, Robot-{robot+1}: Beklenen={r_val}, Bulunan={s_val}")

    print(f"\n✅ Doğru: {correct} / {total}  | Skor: {round(correct/total*540)} / 540")
    if errors:
        print("\n❌ Hatalar:")
        for e in errors:
            print(e)
    else:
        print("🎉 Tüm eşleşmeler doğru!")

if __name__ == "__main__":
    compare_outputs("odev2-videolar/tusas-odev2-referans1.txt", "odev2-videolar/tusas-odev2-ogr1.txt")
