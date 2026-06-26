# Prediksi Outcome Putusan Kasasi Peradilan Anak (ABH) Berbasis Case-Based Reasoning: Studi Komparasi TF-IDF, IndoBERT, dan SVM
## Mahkamah Agung Republik Indonesia

> Implementasi Case-Based Reasoning untuk prediksi outcome putusan kasasi peradilan anak (ABH) di Mahkamah Agung RI — studi komparasi TF-IDF, IndoBERT, dan SVM dengan mekanisme Hybrid BERT+SVM sebagai konfigurasi terbaik

---

## Struktur Proyek

```
cbr-peradilan-anak/
│
├── Notebook/
│   ├── 01_preprocessing.ipynb         # Scraping & preprocessing PDF dari MA
│   ├── 02_case_representation.ipynb   # Ekstraksi metadata & text features
│   ├── 03_case_retrieval.ipynb        # Retrieval: TF-IDF vs IndoBERT vs SVM
│   ├── 04_case_reuse.ipynb            # Prediksi outcome: voting, weighted, SVM, hybrid
│   ├── 05_revisi_retain.ipynb         # Revisi & retensi kasus terverifikasi ke case base
│   └── 06_evaluation.ipynb            # Evaluasi metrik & error analysis
│
├── requirements_preprocessing.txt     # Dependensi environment 01 & 02
├── requirements_cbr.txt               # Dependensi environment 03, 04, 05 & 06
├── README.md
│
├── data/
│   ├── pdf/                           # PDF putusan MA (input scraping)
│   ├── raw/                           # Teks bersih paragraf utuh (case_NNN.txt)
│   ├── tokens/                        # Token per baris + metadata header (case_NNN.txt & .json)
│   │
│   ├── processed/                     # Output antar-notebook (case base, retrieval, embeddings)
│   │   ├── case_base.json
│   │   ├── case_base.csv
│   │   ├── retrieval_results.json
│   │   └── bert_embeddings.npy
│   │
│   ├── results/                       # Prediksi & solusi reuse
│   │   ├── case_solutions.json
│   │   └── predictions.csv
│   │
│   └── eval/                          # Output evaluasi — dibagi dua subfolder
│       ├── metrics/                   # JSON & CSV hasil evaluasi
│       │   ├── queries.json
│       │   ├── test_case_ids.json
│       │   ├── retrieval_comparison.csv
│       │   ├── retain_log.json
│       │   ├── retain_log.csv
│       │   ├── retrieval_metrics.csv
│       │   ├── prediction_metrics.csv
│       │   └── retain_comparison.csv
│       └── visualisasi/               # Grafik & gambar PNG
│           ├── distribusi_case_base.png
│           ├── retrieval_comparison.png
│           ├── reuse_comparison.png
│           └── evaluation_summary.png
│
├── progress_checkpoint_cbr.txt    # Checkpoint halaman terakhir scraper (auto-generated, di-.gitignore)
├── riwayat_terunduh_cbr.txt       # Riwayat nomor perkara yang sudah diunduh (auto-generated, di-.gitignore)
└── logs/                          # Log preprocessing (auto-generated)
    └── cleaning.log
```

---

## Pembagian Environment

Proyek ini menggunakan **dua environment terpisah** karena library scraping dan library deep learning memiliki konflik dependensi (terutama pada versi `numpy` dan `urllib3`).

```
Environment 1: env_preprocessing
    └── Untuk notebook: 01_preprocessing + 02_case_representation
    └── Berisi: seleniumbase, beautifulsoup4, pdfplumber, pandas, scikit-learn

Environment 2: env_cbr
    └── Untuk notebook: 03_case_retrieval + 04_case_reuse + 05_revisi_retain + 06_evaluation
    └── Berisi: torch, transformers (IndoBERT), scikit-learn, matplotlib
```

---

## Cara Menjalankan — Komputer Lokal

### Langkah 0 — Clone Repository

Buka **Command Prompt** (Windows) atau **Terminal** (Linux/macOS), lalu:

```bash
git clone https://github.com/username/cbr-peradilan-anak.git
cd cbr-peradilan-anak
```

Atau unduh ZIP dari GitHub → Extract → buka folder proyeknya.

---

### Langkah 1 — Buat Environment 1 (Preprocessing)

Pilih salah satu cara sesuai yang kamu punya:

#### Jika pakai Anaconda

Buka **Anaconda Prompt** (bukan CMD biasa), lalu ketik:

```bash
conda create -n env_preprocessing python=3.10
conda activate env_preprocessing
pip install -r requirements_preprocessing.txt
```

#### Jika pakai Python biasa (tanpa Anaconda)

Buka **Command Prompt**, lalu ketik:

```bash
python -m venv venv_preprocessing

# Aktifkan — Windows:
venv_preprocessing\Scripts\activate

# Aktifkan — Linux/macOS:
source venv_preprocessing/bin/activate

pip install -r requirements_preprocessing.txt
```

#### Daftarkan kernel ke Jupyter

Setelah environment aktif, jalankan perintah ini agar notebook bisa memilih environment ini:

```bash
python -m ipykernel install --user --name env_preprocessing --display-name "Python (Preprocessing)"
```

---

### Langkah 2 — Buat Environment 2 (CBR Pipeline)

#### Jika pakai Anaconda

Buka **Anaconda Prompt** baru (environment sebelumnya tidak perlu aktif), lalu:

```bash
conda create -n env_cbr python=3.10
conda activate env_cbr
pip install -r requirements_cbr.txt
```

> Jika punya GPU NVIDIA, install PyTorch versi CUDA terlebih dahulu sebelum requirements_cbr.txt:
> ```bash
> # Cek versi CUDA kamu dengan: nvidia-smi
> # Contoh untuk CUDA 11.8:
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> # Baru install sisanya:
> pip install -r requirements_cbr.txt
> ```

#### Jika pakai Python biasa

```bash
python -m venv venv_cbr

# Windows:
venv_cbr\Scripts\activate
# Linux/macOS:
source venv_cbr/bin/activate

pip install -r requirements_cbr.txt
```

#### Daftarkan kernel ke Jupyter

```bash
python -m ipykernel install --user --name env_cbr --display-name "Python (CBR)"
```

---

### Langkah 3 — Buka Jupyter dan Pilih Kernel yang Benar

Buka Jupyter Notebook atau JupyterLab:

```bash
jupyter notebook
# atau
jupyter lab
```

Saat membuka setiap notebook, pastikan kernel yang dipilih sudah benar:

| Notebook | Kernel yang harus dipilih |
|----------|--------------------------|
| `01_preprocessing.ipynb` | Python (Preprocessing) |
| `02_case_representation.ipynb` | Python (Preprocessing) |
| `03_case_retrieval.ipynb` | Python (CBR) |
| `04_case_reuse.ipynb` | Python (CBR) |
| `05_revisi_retain.ipynb` | Python (CBR) |
| `06_evaluation.ipynb` | Python (CBR) |

**Di JupyterLab:** klik nama kernel di pojok kanan atas → pilih kernel yang sesuai.
**Di VS Code:** klik tombol "Select Kernel" di pojok kanan atas notebook → pilih kernel.
**Di Jupyter Notebook klasik:** menu Kernel → Change Kernel → pilih yang sesuai.

---

### Langkah 4 — Jalankan Notebook Secara Berurutan

Semua file proyek berformat **`.ipynb` (Jupyter Notebook)**, bukan script `.py`.
Cara menjalankannya adalah membuka notebook di Jupyter/VS Code lalu menjalankan
cell satu per satu dari atas ke bawah, atau klik **Run All** untuk menjalankan semua cell sekaligus.

> **Jangan jalankan sebagai script Python biasa** (`python 03_case_retrieval.ipynb`) —
> format `.ipynb` tidak bisa dieksekusi langsung oleh Python interpreter.
> Jika ingin menjalankan dari terminal, gunakan:
> ```bash
> jupyter nbconvert --to notebook --execute 03_case_retrieval.ipynb
> ```

Urutan pipeline:

```
01_preprocessing       →  02_case_representation  →  03_case_retrieval
(env_preprocessing)       (env_preprocessing)         (env_cbr)
                                                            ↓
                                                      04_case_reuse (env_cbr)
                                                            ↓
                                                      05_revisi_retain (env_cbr)
                                                            ↓
                                                      06_evaluation (env_cbr)
```

Setiap notebook **wajib selesai dijalankan** sebelum lanjut ke notebook berikutnya karena outputnya saling bergantung.

---

### Mengapa Scraping TIDAK Disarankan di Google Colab?

Tahap 01 (scraping) **wajib dijalankan di komputer lokal**, bukan di Google Colab. Alasannya:

| Masalah | Penjelasan |
|---------|-----------|
| **Sesi timeout** | Colab memutus koneksi setelah ~90 menit tidak aktif. Scraping 40+ dokumen bisa memakan waktu lebih lama dan seluruh progress hilang tanpa bisa dilanjutkan. |
| **IP dibanned** | Colab berbagi IP publik dengan ribuan pengguna lain secara bersamaan. Server MA.go.id mendeteksi pola request berulang dari IP yang sama dan memblokir akses sementara. |
| **Storage tidak persisten** | File yang diunduh ke `/content/` Colab **hilang otomatis** saat sesi berakhir. Menyimpan ke Google Drive bisa dilakukan tapi lambat dan sering disconnect di tengah proses download. |
| **Browser automation sulit** | Situs MA menggunakan JavaScript untuk render halaman sehingga butuh Selenium + ChromeDriver. Setup ini di Colab memerlukan langkah tambahan yang rumit dan sering gagal. |
| **Rate limiting** | Colab tidak punya kontrol IP yang konsisten sehingga sulit menerapkan jeda (delay) yang tepat antar request untuk menghindari pemblokiran dari server MA. |

---

## Cara Menjalankan — Google Colab

Google Colab **hanya disarankan untuk notebook 02 sampai 06** (setelah PDF sudah ada di lokal dan diupload ke Google Drive).

### Persiapan

1. Jalankan `01_preprocessing.ipynb` di **komputer lokal** terlebih dahulu
2. Upload folder `data/` hasil scraping ke **Google Drive** kamu
3. Buka [colab.research.google.com](https://colab.research.google.com)
4. Upload notebook yang ingin dijalankan

### Setup di setiap sesi Colab

Karena Colab tidak menyimpan library yang diinstall antar sesi, setiap kali membuka Colab baru jalankan cell berikut di bagian paling atas notebook:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Arahkan ke folder proyek di Drive kamu
import os
os.chdir('/content/drive/MyDrive/cbr-peradilan-anak')  # sesuaikan path

# Install library yang dibutuhkan
# Untuk notebook 02 (env_preprocessing):
!pip install pdfplumber pandas scikit-learn

# Untuk notebook 03-06 (env_cbr):
!pip install transformers torch scikit-learn matplotlib pdfplumber
```

> **Catatan Colab:** Google Colab sudah menyediakan GPU gratis (T4). Untuk mengaktifkannya: menu Runtime → Change runtime type → Hardware accelerator → GPU. Ini akan mempercepat proses IndoBERT embedding di notebook 03 secara signifikan.

### Keterbatasan Colab yang perlu diketahui

- Library harus diinstall ulang setiap kali sesi baru dimulai
- Sesi Colab gratis dibatasi ~12 jam dan bisa terputus sewaktu-waktu
- Simpan selalu output penting ke Google Drive, bukan ke `/content/`
- File `bert_embeddings.npy` (~300MB) sebaiknya disimpan ke Drive agar tidak perlu generate ulang setiap sesi

---

## Ringkasan Hasil Evaluasi

*(Angka di bawah diambil langsung dari output eksekusi `06_evaluation.ipynb` pada test set 20% = 28 kasus; akan bervariasi sedikit jika notebook scraping dijalankan ulang karena dataset MA bertambah seiring waktu.)*

**Retrieval (Acc@k diukur dari label outcome top-k hasil retrieval):**

| Model | Acc@1 | Acc@3 | Acc@5 | F1-Weighted |
|-------|-------|-------|-------|-------------|
| TF-IDF | 0.857 | 0.964 | 0.964 | 0.857 |
| IndoBERT | 0.857 | 0.929 | 0.964 | 0.857 |
| SVM (TF-IDF, balanced) | 0.929 | — | — | 0.894 |

**Prediksi outcome / Case Reuse (4 algoritma, test set):**

| Algoritma | Accuracy | F1-Weighted |
|-----------|----------|-------------|
| Majority Vote | 0.893 | 0.842 |
| Weighted Similarity | 0.821 | 0.838 |
| SVM (balanced) | 0.929 | 0.894 |
| **Hybrid BERT+SVM** | **1.000** | **1.000** |

Pada tahap *retrieval* murni, TF-IDF dan IndoBERT relatif setara — TF-IDF justru sedikit lebih baik di Acc@3, sementara SVM (TF-IDF, `class_weight='balanced'`) unggul di Acc@1 berkat penanganan class imbalance yang lebih eksplisit. **Hybrid BERT+SVM** (kombinasi retrieval semantik IndoBERT + klasifikasi SVM dengan confidence threshold 0.40, dipakai sebagai default `metode='hybrid'` di `predict_outcome()` pada Notebook 04) menghasilkan performa terbaik pada tahap prediksi akhir. Skor 1.0 pada test set sekecil ini (28 kasus) kemungkinan juga dipengaruhi ukuran sampel yang kecil dan komposisi kelas yang sangat tidak seimbang, sehingga sebaiknya tidak ditafsirkan sebagai generalisasi mutlak — lihat bagian Limitasi.

---

## Mereplikasi untuk Domain / Pengadilan Lain

Proyek ini dirancang untuk putusan kasasi **peradilan anak (ABH) di Mahkamah Agung RI**, namun pipeline-nya bisa diadaptasi untuk domain lain (misal: putusan tipikor, perdata, pengadilan negeri tertentu). Berikut bagian kode yang perlu disesuaikan:

---

### `01_preprocessing.ipynb` — Scraping & Filter

| Variabel / Baris | Lokasi | Yang perlu diubah |
|-----------------|--------|-------------------|
| `BASE_URL` | Cell 1 | URL direktori putusan MA sesuai kategori yang dituju |
| `KATEGORI_URL` | Cell 1 | Path kategori di situs MA (misal `/kategori/tipikor.html`) |
| `TAHUN_MINIMAL` / `TAHUN_MAKSIMAL` | Cell 1 | Rentang tahun putusan yang ingin dikumpulkan |
| Filter kasasi murni | Cell 1 | Logika filter tingkatan perkara — ubah jika bukan kasasi (misal banding/PK) |
| Marker struktural validasi | Cell 12 | Daftar penanda struktural khas dokumen — tiap jenis putusan punya penanda berbeda |
| `'jenis' : 'putusan kasasi MA'` | Cell 16 | Label jenis dokumen untuk metadata |

---

### `02_case_representation.ipynb` — Ekstraksi Metadata & Fitur

| Variabel / Fungsi | Lokasi | Yang perlu diubah |
|------------------|--------|-------------------|
| `RE_NOMOR` (regex nomor perkara) | Cell 4 | Format nomor perkara berbeda per jenis & pengadilan |
| `RE_PASAL` (regex pasal) | Cell 4 | Pola pasal bisa berbeda tergantung jenis perkara |
| `RE_AMAR` (regex amar putusan) | Cell 4 | Kata kunci amar: `menolak`, `mengabulkan`, `memperbaiki` — sesuaikan dengan domain |
| Deteksi `outcome_kasasi` | Cell 8 | Label outcome (`DITOLAK`, `DIKABULKAN`, `DIPERBAIKI`) dan kata kunci pendeteksinya |
| `feat['memori_kasasi']` | Cell 10 | Pola regex penanda struktural memori kasasi — ganti sesuai struktur dokumen domain baru |
| Field representasi case | Cell 12 | Tambah/hapus field sesuai informasi yang relevan di domain baru |

---

### `03_case_retrieval.ipynb` — Retrieval

Umumnya **tidak perlu diubah** — TF-IDF, IndoBERT, dan SVM bersifat generik. Yang mungkin perlu disesuaikan:

- **Bobot field representasi teks** — jika domain baru punya field yang lebih/kurang informatif dari `memori_kasasi`
- **Label kelas** di SVM — otomatis mengikuti nilai `outcome_kasasi` dari case base

---

### `04_case_reuse.ipynb` — Prediksi Outcome

- **Label minority class** untuk `minority_boost` — ubah jika kelas minoritas di domain baru berbeda
- **Confidence threshold** (`>= 0.40`) pada Hybrid BERT+SVM — bisa di-tune sesuai distribusi kelas baru

---

> **Catatan:** Bagian yang paling krusial adalah regex di `02_case_representation.ipynb` Cell 4 dan 8.
> Kualitas ekstraksi metadata (terutama `outcome_kasasi`) langsung menentukan kualitas seluruh pipeline CBR.
> Disarankan uji regex terlebih dahulu pada 5–10 dokumen sampel sebelum menjalankan pipeline penuh.

## Replikasi Tanpa Scraping Ulang

Jika PDF putusan sudah tersedia (hasil scraping sebelumnya atau diperoleh dari sumber lain), pipeline bisa langsung dimulai dari **`02_case_representation.ipynb`** tanpa menjalankan NB01. Pastikan tiga hal berikut sudah terpenuhi sebelum menjalankan NB02:

---

### 1. Struktur folder harus sudah ada

```
data/
├── pdf/        ← file PDF putusan (wajib ada, dibaca NB02)
├── raw/        ← teks bersih per kasus (wajib ada, dibaca NB02)
└── tokens/     ← token per kasus (dibuat NB01, dibaca NB02 via log)
```

Jika folder belum ada, buat manual:

```bash
mkdir -p data/pdf data/raw data/tokens logs
```

---

### 2. File di `data/raw/` harus sesuai format NB01

NB02 membaca file `.txt` dari `data/raw/` yang dihasilkan NB01 dengan format:

```
# satu file per kasus: case_001.txt, case_002.txt, dst.
# isi: teks bersih paragraf utuh (lowercase, tanpa header/footer PDF)
```

Jika PDF diperoleh dari sumber lain, ekstrak teks terlebih dahulu menggunakan `pdfplumber` dan simpan ke `data/raw/` dengan nama file yang konsisten (`case_NNN.txt`).

---

### 3. File `logs/cleaning.log` harus ada

NB02 membaca file log hasil NB01 (`LOG_PATH`) untuk mengetahui pasangan `file_pdf ↔ file_output` per kasus. Format CSV-nya:

```
case_id, file_pdf, file_output, status, ...
```

Jika log tidak ada, NB02 akan gagal di Cell 12. Dua solusi:

- **Solusi A** — Buat log manual sesuai format di atas untuk setiap file PDF yang dimiliki
- **Solusi B** — Jalankan hanya bagian preprocessing NB01 (Cell 4–16, lewati Cell 1–3 yang berisi scraper) dengan menempatkan PDF secara manual ke `data/pdf/`

---

### Ringkasan cepat

| Yang harus disiapkan | Dari mana |
|----------------------|-----------|
| `data/pdf/*.pdf` | Hasil scraping / download manual dari direktori MA |
| `data/raw/*.txt` | Hasil ekstraksi teks NB01, atau ekstraksi manual dengan pdfplumber |
| `logs/cleaning.log` | Hasil NB01, atau dibuat manual sesuai format |
| `progress_checkpoint_cbr.txt` | Tidak dibutuhkan NB02+ — boleh tidak ada |
| `riwayat_terunduh_cbr.txt` | Tidak dibutuhkan NB02+ — boleh tidak ada |

> **Rekomendasi:** Daripada menyiapkan semua manual, jalankan saja **Cell 4–16 di NB01** (bagian preprocessing PDF, bukan scraping) setelah menempatkan PDF secara manual ke `data/pdf/`. Cell 1–3 (scraper) bisa dilewati sepenuhnya.

## Limitasi

1. **Class imbalance** — dari 150 kasus pada `case_base`: DITOLAK 124 (82.7%), TIDAK TERDETEKSI 11 (7.3%), DIPERBAIKI 10 (6.7%), DIKABULKAN 5 (3.3%). Di antara kelas berlabel valid (DITOLAK/DIPERBAIKI/DIKABULKAN saja), proporsinya ±89% / 7% / 4%. Model kesulitan mendeteksi minority class.
2. **DIPERBAIKI sulit terdeteksi** — secara linguistik ~95% identik dengan DITOLAK (hakim tetap menyatakan terdakwa bersalah, hanya vonis yang diubah), perbedaan hanya di 1–2 kalimat amar putusan. Rata-rata similarity (IndoBERT) DIPERBAIKI ke DITOLAK ±0.967, dan DIKABULKAN ke DITOLAK ±0.960.
3. **Keseragaman bahasa hukum** — seluruh putusan menggunakan template baku MA sehingga similarity antar kelas konsisten di atas 0.92.
4. **Sensor dokumen** — beberapa putusan anak memiliki nomor perkara dan nama terdakwa yang disensor sesuai UU SPPA Nomor 11 Tahun 2012 tentang Sistem Peradilan Pidana Anak.
5. **Test set kecil** — evaluasi akhir hanya menggunakan 28 kasus (20% dari 150 kasus berlabel valid), sehingga skor sempurna (1.0) pada Hybrid BERT+SVM kemungkinan dipengaruhi ukuran sampel yang kecil, bukan murni indikasi generalisasi sempurna.

---

## Format File — Notebook vs Script

Brief tugas menyebutkan output berupa file `.py` (misal: `03_retrieval.py`, `04_predict.py`). Pada proyek ini, **seluruh implementasi menggunakan format Jupyter Notebook (`.ipynb`)** dengan alasan:

- Setiap tahap pipeline dapat dieksekusi dan diinspeksi per-cell secara interaktif
- Output visualisasi (grafik, tabel) langsung tampil di dalam notebook
- Lebih mudah untuk debugging dan iterasi regex/model
- Format `.ipynb` diterima sebagai ekuivalen dari `.py` untuk tugas berbasis data science

Jika diperlukan dalam format `.py`, notebook dapat dikonversi dengan perintah:

```bash
# Pastikan environment sudah aktif, lalu:
jupyter nbconvert --to script 03_case_retrieval.ipynb
# Menghasilkan: 03_case_retrieval.py
```

Atau dijalankan langsung dari terminal tanpa konversi:

```bash
jupyter nbconvert --to notebook --execute 03_case_retrieval.ipynb
```

> Catatan jika diconvert ke `.py`: cell markdown berubah menjadi komentar, fungsi `display()` tidak tampil interaktif, dan error lebih sulit dilacak karena tidak ada output per-cell. Disarankan tetap menggunakan format `.ipynb`.

---

## Referensi

- UU Nomor 11 Tahun 2012 tentang Sistem Peradilan Pidana Anak (SPPA)
- IndoBERT: `indobenchmark/indobert-base-p1` — [HuggingFace](https://huggingface.co/indobenchmark/indobert-base-p1)
- Mahkamah Agung RI — [putusan.mahkamahagung.go.id](https://putusan3.mahkamahagung.go.id)
- Scikit-learn — [scikit-learn.org](https://scikit-learn.org)

---

## Informasi Proyek

- **Domain:** Hukum Pidana Anak — Kasasi Mahkamah Agung RI
- **Dataset:** 150 putusan kasasi
- **Bahasa:** Indonesia
- **Framework:** Python · Jupyter · scikit-learn · HuggingFace Transformers