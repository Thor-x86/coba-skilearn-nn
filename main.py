import csv
from sklearn.neural_network import MLPClassifier

print('Memproses berkas data.csv...')

# Buka data.csv
list_data = list()
with open('data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for tiap_baris in csv_reader:
        # Hiraukan baris kosong
        if not tiap_baris:
            continue
        list_data.append(tiap_baris)

jumlah_kolom = len(list_data[0])

# Sel masih berupa string, kita perlu konversi
# sel tersebut menjadi float (atau int khusus label)
for baris in list_data:
    for indeks_kolom in range(jumlah_kolom):
        sel = baris[indeks_kolom].strip()
        if indeks_kolom < jumlah_kolom - 1:
            # Selain label
            baris[indeks_kolom] = float(sel)
        else:
            # Untuk label
            baris[indeks_kolom] = int(sel)

# Pisahkan label dan data
list_label = list()
for baris in list_data:
    label = baris[jumlah_kolom-1]
    list_label.append(label)
    del(baris[jumlah_kolom-1])

print('Melatih Neural Network...')
neural_network = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)
neural_network.fit(list_data, list_label)

# Proses prediksi, Anda masukkan nilai ke terminal nanti
while True:
    print('')

    # Minta data
    masukan = input('Masukkan %i angka dibatasi spasi: ' % (jumlah_kolom-1))

    # Masukan kosong artinya keluar
    if len(masukan) == 0:
        print('Berhenti.')
        print('')
        input('[Tekan ENTER]')
        break

    list_angka = masukan.split(' ')

    # Pastikan jumlah data sesuai dengan data training
    if len(list_angka) != jumlah_kolom-1:
        print('Harus %i angka!' % (jumlah_kolom-1))
        continue

    # Konversi data ke float, peringati jika terdeteksi
    # bukan angka
    try:
        for indeks in range(len(list_angka)):
            list_angka[indeks] = float(list_angka[indeks])
    except ValueError:
        print('Mohon masukkan angka saja!')
        continue

    print('Memproses...')
    hasil = neural_network.predict([list_angka])
    print('')
    print('ID Label adalah %i' % hasil)
