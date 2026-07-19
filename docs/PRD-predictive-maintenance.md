# PRD: Predictive Maintenance / Anomaly Detection Portfolio Project

## Problem Statement

Kandidat (CS student, menerima offer sebagai AI Engineer di tim AI/ML/DL PT Premier Engineering Indonesia) perlu mengirim CV dalam 3 hari, namun belum memiliki portofolio proyek AI/ML yang bisa ditunjukkan ke HR. Kandidat juga baru mulai belajar AI/ML dari nol, sehingga proyek harus feasible dikerjakan sendiri dalam waktu singkat sambil tetap terlihat mumpuni dan relevan dengan domain industri (electrical control, power transmission, hydraulic system, industrial IoT) yang menjadi bisnis inti perusahaan target.

Tanpa portofolio yang solid dan relevan, kandidat berisiko sulit membuktikan kapabilitas AI Engineer-nya saat proses rekrutmen maupun negosiasi gaji.

## Solution

Membangun dan men-deploy sebuah sistem **Predictive Maintenance / Anomaly Detection** berbasis data sensor mesin industri, menggunakan dataset publik AI4I 2020, dengan dua lapisan model (klasifikasi jenis kegagalan + deteksi anomali), ditampilkan melalui dashboard interaktif, di-deploy secara publik, dan didokumentasikan lengkap — selesai dalam 3 hari sebagai bahan portofolio CV.

## User Stories

1. Sebagai kandidat pelamar kerja, saya ingin memiliki proyek yang tema-nya relevan dengan bisnis industrial IoT perusahaan target, sehingga HR/hiring manager melihat kesesuaian langsung dengan kebutuhan tim.
2. Sebagai kandidat pemula di AI/ML, saya ingin menggunakan tech stack yang populer, gratis, dan mudah dipelajari, sehingga saya bisa menyelesaikan proyek dalam 2-3 hari tanpa terjebak kompleksitas berlebihan.
3. Sebagai kandidat, saya ingin proyek mendeteksi kegagalan mesin dari data sensor (suhu, kecepatan rotasi, torsi/beban, keausan alat), sehingga proyek menunjukkan use case nyata predictive maintenance.
4. Sebagai kandidat, saya ingin ada dua lapisan deteksi (klasifikasi kegagalan yang sudah dikenal + anomaly detection untuk pola yang belum pernah dilabeli), sehingga proyek menunjukkan pemahaman lebih dalam soal skenario dunia nyata (tidak semua kegagalan sudah pernah terjadi/terlabeli sebelumnya).
5. Sebagai kandidat, saya ingin dashboard yang mensimulasikan monitoring real-time dari data historis, sehingga demo terlihat interaktif dan mendekati kondisi operasional sungguhan.
6. Sebagai kandidat, saya ingin dashboard menampilkan status risiko dengan indikator warna (hijau/kuning/merah), sehingga nilai bisnis proyek (early warning system) mudah dipahami orang non-teknis sekalipun.
7. Sebagai kandidat, saya ingin proyek di-deploy secara publik (bukan hanya jalan di lokal), sehingga saya bisa mencantumkan link demo langsung di CV/LinkedIn.
8. Sebagai kandidat, saya ingin evaluasi model difokuskan pada recall dan F1-score (bukan hanya accuracy), sehingga saya bisa menjelaskan dengan tepat mengapa metrik itu lebih relevan untuk kasus data yang imbalance seperti predictive maintenance.
9. Sebagai kandidat, saya ingin README project mencantumkan hasil kuantitatif (metrik model) dan batasan proyek secara jujur, sehingga proyek terlihat kredibel dan tidak overclaim di mata reviewer teknis.
10. Sebagai reviewer/HR yang membaca CV, saya ingin bisa langsung membuka demo dan melihat kode di GitHub, sehingga saya dapat menilai kemampuan kandidat tanpa perlu instalasi apapun.
11. Sebagai reviewer teknis (calon rekan tim AI/ML/DL), saya ingin melihat dokumentasi yang menjelaskan alasan pemilihan dataset, model, dan metrik, sehingga saya bisa menilai pemahaman kandidat, bukan sekadar hasil akhirnya.

## Implementation Decisions

- **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI/Kaggle), dipilih dibanding CWRU Bearing Dataset (vibrasi mentah) karena format tabular lebih cepat dikerjakan dalam 2-3 hari dan tidak membutuhkan signal processing (FFT, ekstraksi fitur getaran) yang menambah kompleksitas bagi pemula. Batasan ini dicatat secara eksplisit: dataset tidak memiliki data vibrasi mentah; kolom `Torque` digunakan sebagai proxy beban/electrical stress, bukan data arus motor asli.
- **Fitur input:** Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Type (kualitas produk).
- **Model klasifikasi:** Random Forest atau XGBoost dengan `class_weight='balanced'` untuk menangani imbalance kelas (failure ~3% dari data), memprediksi status Machine Failure (dan idealnya tipe failure spesifik: Heat Dissipation, Power Failure, Overstrain, Tool Wear, Random Failure).
- **Model anomaly detection:** Isolation Forest, dilatih hanya dari data kondisi normal (unsupervised), menghasilkan anomaly score untuk menangkap pola tidak wajar di luar pola failure yang sudah dilabeli.
- **Logika risk level (gabungan kedua model):**
  - Merah: model klasifikasi memprediksi failure = True
  - Kuning: klasifikasi normal, tapi anomaly score di atas threshold (top 5% anomaly score tertinggi)
  - Hijau: normal di kedua model
- **Simulasi real-time:** dashboard mereplay data historis (test set) baris demi baris untuk mensimulasikan streaming sensor, bukan koneksi ke sensor IoT sungguhan.
- **Dashboard:** Streamlit — menampilkan line chart historis tiap sensor (Plotly), kartu status risiko terkini, dan log riwayat alert.
- **Tech stack:** Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit — seluruhnya gratis dan populer saat ini.
- **Deployment:** GitHub (kode publik) + Streamlit Community Cloud (hosting gratis, deploy langsung dari repo).
- **Struktur proyek:** dipisah antara notebook eksplorasi (`notebooks/`), modul training (`src/`), model tersimpan (`models/`), dan aplikasi dashboard (`app.py`) agar mudah direview.

## Testing Decisions

- **Evaluasi model klasifikasi:** precision, recall, dan F1-score per kelas — bukan accuracy, karena distribusi kelas sangat imbalance (accuracy tinggi bisa dicapai model yang selalu menebak "normal"). Recall menjadi prioritas karena false negative (gagal mendeteksi kerusakan) memiliki biaya lebih tinggi dalam konteks predictive maintenance.
- **Confusion matrix** disimpan sebagai bukti visual hasil evaluasi, dilampirkan di README.
- **Split data:** train/test split dengan `stratify=y` untuk menjaga proporsi kelas failure di kedua subset, mengingat data sangat imbalance.
- **Validasi anomaly detection:** anomaly score dari Isolation Forest dibandingkan secara kualitatif dengan label failure yang ada, untuk melihat apakah skor tinggi berkorelasi dengan kasus failure (meski model ini unsupervised dan tidak dievaluasi dengan label secara langsung).
- **Prior art:** tidak ada test suite kode sebelumnya dalam konteks ini (proyek baru/personal), sehingga evaluasi difokuskan pada metrik model, bukan unit test perangkat lunak.

## Out of Scope

- Koneksi ke sensor IoT/hardware sungguhan — sepenuhnya simulasi dari data historis.
- Online learning / model yang otomatis update saat data baru masuk.
- Model deep learning (LSTM, Autoencoder neural network) — versi awal menggunakan model klasik (Random Forest/XGBoost, Isolation Forest) demi kecepatan pengerjaan dan kemudahan penjelasan saat interview.
- Generalisasi ke seluruh jenis mesin yang disuplai PT Premier Engineering — dataset merepresentasikan satu skenario mesin/proses saja.
- Data vibrasi mentah dan analisis sinyal (FFT, kurtosis, dsb.) — tidak termasuk karena keterbatasan dataset yang dipilih (Opsi A).
- Proyek kedua (RAG Chatbot untuk Q&A datasheet produk) — dibahas terpisah sebagai proyek portofolio lanjutan, bukan bagian dari spec ini.

## Further Notes

- **Timeline:** 3 hari — Hari 1 (data & preprocessing), Hari 2 (modeling & evaluasi), Hari 3 (dashboard, deployment, dokumentasi).
- **README wajib mencantumkan:** problem statement, penjelasan dataset & batasannya, alasan pemilihan pendekatan/model, hasil kuantitatif (metrik model), link demo live, screenshot/GIF, dan instruksi menjalankan proyek secara lokal.
- Kejujuran soal batasan proyek (bukan real IoT, torque sebagai proxy, dataset merepresentasikan satu skenario) sengaja ditonjolkan di dokumentasi karena menambah kredibilitas kandidat di mata reviewer teknis, bukan mengurangi nilai proyek.
- Proyek ini merupakan proyek portofolio pertama dari rencana dua proyek; proyek kedua (RAG Chatbot) sudah dibahas sebelumnya namun dieksekusi setelah proyek ini selesai.
