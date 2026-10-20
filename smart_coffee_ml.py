#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Agro-Analytics
Prediksi Hasil Panen, Klasifikasi Mutu Biji Kopi, dan Klasterisasi Lahan
Petani Lokal Menggunakan Machine Learning
=========================================================================

Tugas kuliah "Introduction to Machine Learning".
Program ini menjalankan 6 tahap ML Workflow secara berurutan:

    Tahap 1  Data Collection (dataset sintetis 600 lahan kebun kopi)
    Tahap 2  Data Preprocessing (duplikat, missing value, encoding, scaling)
    Tahap 3  Data Splitting (80% training, 20% testing)
    Tahap 4  Supervised Learning (regresi hasil panen & klasifikasi mutu)
    Tahap 5  Unsupervised Learning (K-Means & Isolation Forest)
    Tahap 6  Inference (prediksi untuk satu kebun baru)

Seluruh data bersifat SINTETIS dan dibuat hanya untuk pembelajaran.

Cara menjalankan:
    python smart_coffee_ml.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # tanpa GUI, aman dijalankan di server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score,
                             silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
FITUR = ["ketinggian_mdpl", "curah_hujan_mm", "ph_tanah",
         "suhu_rata_rata", "dosis_pupuk_kg"]
TARGET_REG = "hasil_panen_kg"      # target regresi (numerik kontinu)
TARGET_CLF = "mutu_kopi"           # target klasifikasi ('Specialty'/'Reguler')
KOLOM_FLAG = "_anomali_injeksi"    # HANYA untuk validasi, tidak dipakai model
OUTPUT_GAMBAR = "visualisasi_ml_kopi.png"
OUTPUT_CSV = "dataset_kebun_kopi.csv"


def judul(teks):
    print("\n" + "=" * 78)
    print(f" {teks}")
    print("=" * 78)


# ===========================================================================
# TAHAP 1 - DATA COLLECTION (Dataset Sintetis)
# ===========================================================================
def buat_dataset(n_total=600, n_duplikat=12, n_outlier=10,
                 frac_missing=0.05, seed=RANDOM_STATE):
    """
    Membuat dataset sintetis 600 lahan kebun kopi lokal.

    Satuan:
        ketinggian_mdpl : meter di atas permukaan laut
        curah_hujan_mm  : curah hujan tahunan (mm/tahun)
        ph_tanah        : derajat keasaman tanah
        suhu_rata_rata  : suhu rata-rata harian (derajat Celsius)
        dosis_pupuk_kg  : dosis pupuk (kg per hektar per tahun)
        hasil_panen_kg  : hasil panen biji hijau (kg per hektar per musim)
        mutu_kopi       : 'Specialty' atau 'Reguler'

    Sengaja disisipkan masalah data yang realistis:
        - sekitar 5% nilai kosong (NaN) pada kolom fitur
        - data duplikat (baris yang tercatat dua kali)
        - data outlier/anomali (indikasi hama, tanah rusak, dll.)
    """
    rng = np.random.default_rng(seed)
    n_unik = n_total - n_duplikat

    # Tiga zona agroklimat laten (dataran rendah, menengah, tinggi).
    zona = rng.choice(3, size=n_unik, p=[0.30, 0.45, 0.25])
    alt = rng.normal(np.array([850, 1250, 1650])[zona],
                     np.array([80, 100, 90])[zona])
    alt = np.clip(alt, 600, 2000)
    suhu = 28.5 - 0.0065 * alt + rng.normal(0, 0.7, n_unik)  # makin tinggi makin dingin
    hujan = rng.normal(np.array([1700, 2000, 2350])[zona], 300)
    hujan = np.clip(hujan, 900, 3500)
    ph = np.clip(rng.normal(np.array([5.6, 5.9, 6.1])[zona], 0.45), 4.5, 7.2)
    dosis = np.clip(rng.normal(140, 45, n_unik), 30, 280)

    # Hasil panen: fungsi non-linear dari kondisi lahan + noise.
    f_alt = np.exp(-0.5 * ((alt - 1300) / 450) ** 2)
    f_hujan = np.exp(-0.5 * ((hujan - 2000) / 700) ** 2)
    f_ph = np.exp(-0.5 * ((ph - 6.0) / 0.8) ** 2)
    f_pupuk = 1 - np.exp(-dosis / 90)            # efek pupuk jenuh
    hasil = 1600 * f_alt * f_hujan * f_ph * (0.35 + 0.65 * f_pupuk)
    hasil = np.clip(hasil + rng.normal(0, 70, n_unik), 60, None)

    # Mutu kopi: dataran tinggi, pH mendekati 6, dan hujan moderat -> Specialty
    skor_mutu = (0.0055 * (alt - 1250)
                 - 0.6 * np.abs(ph - 6.1)
                 - 0.0004 * np.abs(hujan - 1900)
                 + rng.normal(0, 0.7, n_unik))
    ambang = np.quantile(skor_mutu, 0.60)
    mutu = np.where(skor_mutu > ambang, "Specialty", "Reguler").astype(object)

    df = pd.DataFrame({
        "ketinggian_mdpl": alt,
        "curah_hujan_mm": hujan,
        "ph_tanah": ph,
        "suhu_rata_rata": suhu,
        "dosis_pupuk_kg": dosis,
        "hasil_panen_kg": hasil,
        "mutu_kopi": mutu,
    })
    df[KOLOM_FLAG] = False

    # --- Sisipkan outlier/anomali (5 jenis kasus ekstrem) -----------------
    idx_out = rng.choice(n_unik, size=n_outlier, replace=False)
    for i, idx in enumerate(idx_out):
        jenis = i % 5
        if jenis == 0:    # serangan hama: kondisi normal tetapi panen anjlok
            df.loc[idx, "hasil_panen_kg"] *= 0.08
            df.loc[idx, "mutu_kopi"] = "Reguler"
        elif jenis == 1:  # tanah rusak/sangat asam
            df.loc[idx, "ph_tanah"] = rng.uniform(3.2, 3.8)
            df.loc[idx, "hasil_panen_kg"] *= 0.15
            df.loc[idx, "mutu_kopi"] = "Reguler"
        elif jenis == 2:  # ketinggian ekstrem
            df.loc[idx, "ketinggian_mdpl"] = rng.uniform(2300, 2700)
            df.loc[idx, "suhu_rata_rata"] = 28.5 - 0.0065 * df.loc[idx, "ketinggian_mdpl"]
            df.loc[idx, "hasil_panen_kg"] *= 0.30
        elif jenis == 3:  # pemupukan berlebihan
            df.loc[idx, "dosis_pupuk_kg"] = rng.uniform(520, 700)
            df.loc[idx, "hasil_panen_kg"] *= 0.50
            df.loc[idx, "mutu_kopi"] = "Reguler"
        else:             # curah hujan ekstrem
            df.loc[idx, "curah_hujan_mm"] = rng.uniform(4200, 5200)
            df.loc[idx, "hasil_panen_kg"] *= 0.40
        df.loc[idx, KOLOM_FLAG] = True

    # --- Sisipkan missing value (~5% sel pada kolom fitur) ---------------
    for kol in FITUR:
        mask = rng.random(n_unik) < frac_missing
        df.loc[mask, kol] = np.nan

    # --- Sisipkan data duplikat (salinan persis dari baris yang ada) ------
    idx_dup = rng.choice(n_unik, size=n_duplikat, replace=False)
    df = pd.concat([df, df.iloc[idx_dup]], ignore_index=True)

    # Acak urutan baris agar duplikat tidak berkumpul di akhir
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def tahap1_data_collection():
    judul("TAHAP 1 - DATA COLLECTION (Dataset Sintetis)")
    print("Pertanyaan data-driven: apakah kondisi lahan (ketinggian, hujan, pH,")
    print("suhu, pupuk) dapat dipelajari mesin untuk memprediksi hasil panen,")
    print("mutu kopi, dan mengelompokkan lahan secara otomatis?\n")

    df = buat_dataset()
    print(f"Dimensi dataset mentah : {df.shape[0]} baris x "
          f"{df.shape[1] - 1} kolom")
    print("\n5 data pertama:")
    print(df.drop(columns=KOLOM_FLAG).head().round(2).to_string())

    print("\nStatistik deskriptif (data mentah):")
    print(df[FITUR + [TARGET_REG]].describe().round(2).T.to_string())
    print("\nPerhatikan nilai min/max yang ekstrem (indikasi outlier), dan "
          "jumlah count yang\nlebih kecil dari total baris (indikasi missing "
          "value).")

    # Simpan dataset (tanpa kolom flag validasi) agar bisa dipakai ulang
    df.drop(columns=KOLOM_FLAG).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset disimpan ke: {OUTPUT_CSV}")
    return df


# ===========================================================================
# TAHAP 2 - DATA PREPROCESSING
# ===========================================================================
def tahap2_preprocessing_awal(df):
    """Pembersihan yang aman dilakukan sebelum split: duplikat & encoding."""
    judul("TAHAP 2 - DATA PREPROCESSING")

    # 2.1 Cek missing value
    print("[2.1] Missing value per kolom (sebelum penanganan):")
    miss = df.drop(columns=KOLOM_FLAG).isna().sum()
    tampil = pd.DataFrame({"jumlah_NaN": miss,
                           "persen": (miss / len(df) * 100).round(2)})
    print(tampil.to_string())
    total_sel = len(df) * len(FITUR)
    print(f"      Total NaN pada kolom fitur: {int(miss[FITUR].sum())} sel "
          f"({miss[FITUR].sum() / total_sel * 100:.2f}% dari sel fitur)")

    # 2.2 Hapus duplikat
    kolom_data = [c for c in df.columns if c != KOLOM_FLAG]
    n_dup = int(df.duplicated(subset=kolom_data).sum())
    sebelum = len(df)
    df = df.drop_duplicates(subset=kolom_data).reset_index(drop=True)
    print(f"\n[2.2] Data duplikat ditemukan: {n_dup} baris -> dihapus "
          f"({sebelum} -> {len(df)} baris)")

    # 2.3 Encoding label kategori
    encoder = LabelEncoder()
    df["mutu_kode"] = encoder.fit_transform(df[TARGET_CLF])
    mapping = {kelas: int(kode) for kode, kelas in enumerate(encoder.classes_)}
    print(f"\n[2.3] Encoding label '{TARGET_CLF}': {mapping}")
    print(df[TARGET_CLF].value_counts().to_string())
    return df, encoder


def tahap2_lanjutan_imputasi_scaling(X_train, X_test):
    """
    Imputasi median dan StandardScaler.

    Keduanya di-fit HANYA pada data training lalu diterapkan ke data testing,
    supaya informasi data testing tidak bocor ke proses training
    (mencegah data leakage).
    """
    judul("TAHAP 2 (lanjutan) - IMPUTASI MISSING VALUE & FEATURE SCALING")

    print("[2.4] Imputasi missing value dengan median (di-fit pada data training)")
    print(f"      NaN training sebelum : {int(X_train.isna().sum().sum())}")
    print(f"      NaN testing sebelum  : {int(X_test.isna().sum().sum())}")
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=FITUR, index=X_train.index)
    Xte_imp = pd.DataFrame(imputer.transform(X_test),
                           columns=FITUR, index=X_test.index)
    print(f"      NaN training sesudah : {int(Xtr_imp.isna().sum().sum())}")
    print(f"      NaN testing sesudah  : {int(Xte_imp.isna().sum().sum())}")
    print("      Nilai median yang dipakai:")
    for kol, med in zip(FITUR, imputer.statistics_):
        print(f"        {kol:<17}: {med:.2f}")

    print("\n[2.5] Feature scaling dengan StandardScaler (mean=0, std=1)")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_imp)
    Xte = scaler.transform(Xte_imp)
    print("      Rata-rata & std data training setelah scaling:")
    print(f"        mean ~ {np.round(Xtr.mean(axis=0), 3).tolist()}")
    print(f"        std  ~ {np.round(Xtr.std(axis=0), 3).tolist()}")

    return {"imputer": imputer, "scaler": scaler,
            "Xtr_imp": Xtr_imp, "Xte_imp": Xte_imp,
            "Xtr": Xtr, "Xte": Xte}


# ===========================================================================
# TAHAP 3 - DATA SPLITTING
# ===========================================================================
def tahap3_splitting(df):
    judul("TAHAP 3 - DATA SPLITTING (80% Training : 20% Testing)")
    train, test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE,
        stratify=df["mutu_kode"])   # proporsi Specialty/Reguler dijaga
    print(f"Total data bersih : {len(df)}")
    print(f"Data training     : {len(train)} ({len(train) / len(df) * 100:.0f}%)")
    print(f"Data testing      : {len(test)} ({len(test) / len(df) * 100:.0f}%)")
    print("Proporsi Specialty (kode 1) -> "
          f"training {train['mutu_kode'].mean() * 100:.1f}% | "
          f"testing {test['mutu_kode'].mean() * 100:.1f}%  (stratified)")
    return train, test


# ===========================================================================
# TAHAP 4 - SUPERVISED LEARNING
# ===========================================================================
def tahap4a_regresi(Xtr, Xte, ytr, yte):
    judul("TAHAP 4a - SUPERVISED LEARNING: REGRESI (Prediksi hasil_panen_kg)")
    lr = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                               n_jobs=-1).fit(Xtr, ytr)

    hasil = {}
    for nama, model in [("Linear Regression (baseline)", lr),
                        ("Random Forest Regressor", rf)]:
        pred = model.predict(Xte)
        hasil[nama] = (mean_absolute_error(yte, pred), r2_score(yte, pred))

    print(f"{'Model':<32}{'MAE (kg)':>12}{'R2 Score':>12}")
    print("-" * 56)
    for nama, (mae, r2) in hasil.items():
        print(f"{nama:<32}{mae:>12.2f}{r2:>12.4f}")

    pred_rf = rf.predict(Xte)
    mae, r2 = hasil["Random Forest Regressor"]
    cv = cross_val_score(rf, Xtr, ytr, cv=5, scoring="r2")
    print(f"\nModel utama: Random Forest Regressor")
    print(f"  MAE data testing : {mae:.2f} kg  "
          f"(rata-rata selisih prediksi vs aktual)")
    print(f"  R2 data testing  : {r2:.4f}  "
          f"(proporsi variasi hasil panen yang dijelaskan model)")
    print(f"  R2 cross-val 5x  : {cv.mean():.4f} +/- {cv.std():.4f} (data training)")
    return rf, pred_rf, mae, r2


def tahap4b_klasifikasi(Xtr, Xte, ytr, yte, encoder):
    judul("TAHAP 4b - SUPERVISED LEARNING: KLASIFIKASI (mutu_kopi)")
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                 n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred)
    rec = recall_score(yte, pred)
    f1 = f1_score(yte, pred)

    print("Kelas positif = 'Specialty' (kode 1)\n")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (dari yang diprediksi Specialty, "
          f"berapa yang benar)")
    print(f"  Recall    : {rec:.4f}  (dari Specialty sebenarnya, "
          f"berapa yang terdeteksi)")
    print(f"  F1-Score  : {f1:.4f}")

    cm = confusion_matrix(yte, pred)
    cm_df = pd.DataFrame(cm,
                         index=[f"Aktual {k}" for k in encoder.classes_],
                         columns=[f"Prediksi {k}" for k in encoder.classes_])
    print("\nConfusion Matrix:")
    print(cm_df.to_string())
    print("\nClassification Report:")
    print(classification_report(yte, pred, target_names=encoder.classes_,
                                digits=3))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(clf, Xtr, ytr, cv=skf, scoring="accuracy")
    print(f"Accuracy cross-val 5x (data training): "
          f"{cv.mean():.4f} +/- {cv.std():.4f}")
    return clf, cm, {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ===========================================================================
# TAHAP 5 - UNSUPERVISED LEARNING
# ===========================================================================
NAMA_ZONA = ["Zona A - Dataran Rendah", "Zona B - Dataran Menengah",
             "Zona C - Dataran Tinggi"]


def tahap5a_clustering(X_all_scaled, df_all_raw):
    judul("TAHAP 5a - UNSUPERVISED LEARNING: K-MEANS CLUSTERING (K=3)")
    print("Clustering hanya memakai 5 fitur lahan; label hasil panen dan mutu "
          "TIDAK dipakai.\n")
    km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    label = km.fit_predict(X_all_scaled)
    sil = silhouette_score(X_all_scaled, label)

    # Beri nama zona berdasarkan urutan ketinggian pusat klaster
    pusat_alt = df_all_raw.groupby(label)["ketinggian_mdpl"].mean()
    urutan = pusat_alt.sort_values().index.tolist()      # rendah -> tinggi
    peta_nama = {kid: NAMA_ZONA[r] for r, kid in enumerate(urutan)}
    nama_zona = np.array([peta_nama[k] for k in label])

    print(f"Silhouette Score : {sil:.4f}  (makin mendekati 1 makin terpisah)")
    print(f"Inertia          : {km.inertia_:.2f}\n")

    profil = df_all_raw[FITUR].groupby(nama_zona).mean()
    profil["jumlah_lahan"] = pd.Series(nama_zona).value_counts()
    profil["rata2_hasil_panen_kg"] = df_all_raw[TARGET_REG].groupby(nama_zona).mean()
    profil["persen_specialty"] = (
        df_all_raw["mutu_kode"].groupby(nama_zona).mean() * 100)
    print("Profil tiap zona agroklimat (rata-rata):")
    print(profil.sort_index().round(2).T.to_string())
    print("\n(Kolom hasil panen & % Specialty hanya dipakai untuk "
          "menafsirkan zona setelah\nclustering selesai, bukan untuk "
          "membentuknya.)")
    return km, label, nama_zona, peta_nama, sil


def tahap5b_anomaly(df_all_raw, flag_asli):
    judul("TAHAP 5b - UNSUPERVISED LEARNING: ANOMALY DETECTION (Isolation Forest)")
    fitur_anomali = FITUR + [TARGET_REG]
    iso = IsolationForest(n_estimators=300, contamination=0.03,
                          random_state=RANDOM_STATE)
    pred = iso.fit_predict(df_all_raw[fitur_anomali])   # -1 = anomali
    skor = iso.score_samples(df_all_raw[fitur_anomali])  # makin rendah makin aneh
    is_anom = pred == -1

    print(f"Parameter contamination = 0.03 (asumsi ~3% kebun berkondisi ekstrem)")
    print(f"Kebun terdeteksi anomali : {int(is_anom.sum())} dari {len(df_all_raw)}")

    daftar = df_all_raw.loc[is_anom, fitur_anomali].copy()
    daftar["skor_anomali"] = skor[is_anom]
    daftar = daftar.sort_values("skor_anomali")
    print("\n8 kebun paling anomali:")
    print(daftar.head(8).round(2).to_string())

    # Validasi: bandingkan dengan anomali yang sengaja disisipkan di Tahap 1
    tp = int((is_anom & flag_asli).sum())
    n_asli = int(flag_asli.sum())
    presisi = tp / max(int(is_anom.sum()), 1)
    recall = tp / max(n_asli, 1)
    print(f"\nValidasi terhadap {n_asli} anomali yang disisipkan di Tahap 1 "
          f"(label ini tidak dipakai saat training):")
    print(f"  Terdeteksi benar : {tp}/{n_asli}  (recall {recall:.2f}, "
          f"presisi {presisi:.2f})")
    print("  Catatan: anomali yang terlewat biasanya kasus dengan kondisi lahan "
          "tampak normal\n  (misalnya serangan hama yang hanya menurunkan panen), "
          "sedangkan kebun lain yang ikut\n  terdeteksi biasanya nilai ekstrem "
          "alami. Isolation Forest bekerja tanpa label, sehingga\n  hasilnya "
          "perlu dikonfirmasi lewat pengecekan lapangan.")
    return iso, is_anom, skor, {"tp": tp, "n_asli": n_asli,
                                "recall": recall, "presisi": presisi}


def tahap5c_visualisasi(yte_reg, pred_reg, mae, r2, cm, encoder, m_clf,
                        rf_reg, clf, km, peta_nama, X_all_scaled, nama_zona,
                        df_all_raw, is_anom, nama_file=OUTPUT_GAMBAR):
    judul("TAHAP 5c - VISUALISASI HASIL")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    palet = {NAMA_ZONA[0]: "#e67e22", NAMA_ZONA[1]: "#27ae60",
             NAMA_ZONA[2]: "#2980b9"}

    # (1) Regresi: aktual vs prediksi
    ax = axes[0, 0]
    ax.scatter(yte_reg, pred_reg, alpha=0.7, color="#2e86c1", edgecolor="white")
    lo, hi = min(yte_reg.min(), pred_reg.min()), max(yte_reg.max(), pred_reg.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Prediksi ideal")
    ax.set_xlabel("Hasil panen aktual (kg)")
    ax.set_ylabel("Hasil panen prediksi (kg)")
    ax.set_title(f"Regresi (Random Forest)\nMAE = {mae:.1f} kg | R2 = {r2:.3f}")
    ax.legend()

    # (2) Klasifikasi: confusion matrix
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=encoder.classes_, yticklabels=encoder.classes_,
                annot_kws={"size": 16})
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix Mutu Kopi\nAccuracy = {m_clf['acc']:.3f} | "
                 f"F1 = {m_clf['f1']:.3f}")

    # (3) Feature importance regresi vs klasifikasi
    ax = axes[0, 2]
    imp = pd.DataFrame({"Regresi (hasil panen)": rf_reg.feature_importances_,
                        "Klasifikasi (mutu)": clf.feature_importances_},
                       index=FITUR)
    imp.sort_values("Regresi (hasil panen)").plot.barh(
        ax=ax, color=["#2e86c1", "#e67e22"], width=0.75)
    ax.set_xlabel("Tingkat kepentingan fitur")
    ax.set_title("Feature Importance")

    # (4) K-Means pada proyeksi PCA 2D
    ax = axes[1, 0]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    P = pca.fit_transform(X_all_scaled)
    for z in NAMA_ZONA:
        m = nama_zona == z
        ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.75, color=palet[z], label=z)
    pusat = pca.transform(km.cluster_centers_)
    ax.scatter(pusat[:, 0], pusat[:, 1], marker="X", s=260, color="black",
               edgecolor="white", label="Pusat klaster")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}% variasi)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}% variasi)")
    ax.set_title("K-Means (K=3): Zona Agroklimat Lahan")
    ax.legend(fontsize=8)

    # (5) Profil pusat klaster (z-score)
    ax = axes[1, 1]
    profil = pd.DataFrame(km.cluster_centers_, columns=FITUR)
    profil.index = [peta_nama[i] for i in range(3)]
    profil = profil.loc[NAMA_ZONA]
    sns.heatmap(profil, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title("Profil Zona (pusat klaster, nilai terstandar)")
    ax.tick_params(axis="x", rotation=30)

    # (6) Anomaly detection
    ax = axes[1, 2]
    ax.scatter(df_all_raw.loc[~is_anom, "ketinggian_mdpl"],
               df_all_raw.loc[~is_anom, TARGET_REG],
               s=18, alpha=0.5, color="#95a5a6", label="Normal")
    ax.scatter(df_all_raw.loc[is_anom, "ketinggian_mdpl"],
               df_all_raw.loc[is_anom, TARGET_REG],
               s=70, marker="X", color="#c0392b", label="Anomali")
    ax.set_xlabel("Ketinggian (mdpl)")
    ax.set_ylabel("Hasil panen (kg)")
    ax.set_title(f"Isolation Forest: {int(is_anom.sum())} Kebun Anomali")
    ax.legend()

    fig.suptitle("Smart Agro-Analytics: Hasil Machine Learning pada Data "
                 "Kebun Kopi (Sintetis)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(nama_file, dpi=150)
    plt.close(fig)
    print(f"Visualisasi disimpan ke: {nama_file}")


# ===========================================================================
# TAHAP 6 - INFERENCE
# ===========================================================================
def tahap6_inference(prep, rf_reg, clf, encoder, km, peta_nama, iso):
    judul("TAHAP 6 - INFERENCE: PREDIKSI UNTUK KEBUN KOPI BARU")
    kebun_baru = {
        "ketinggian_mdpl": 1380,
        "curah_hujan_mm": 1950,
        "ph_tanah": 6.0,
        "suhu_rata_rata": 19.6,
        "dosis_pupuk_kg": 160,
    }
    print("Data kebun baru milik petani:")
    for k, v in kebun_baru.items():
        print(f"  {k:<17}: {v}")

    X_baru = pd.DataFrame([kebun_baru])[FITUR]
    X_imp = pd.DataFrame(prep["imputer"].transform(X_baru), columns=FITUR)
    X_scaled = prep["scaler"].transform(X_imp)   # pipeline yang sama dgn training

    hasil_panen = rf_reg.predict(X_scaled)[0]
    proba = clf.predict_proba(X_scaled)[0]
    kelas = encoder.classes_[int(np.argmax(proba))]
    zona = peta_nama[int(km.predict(X_scaled)[0])]

    # Cek keanomalian: Isolation Forest memakai fitur + hasil panen prediksi
    X_anom = X_imp.copy()
    X_anom[TARGET_REG] = hasil_panen
    status_anom = iso.predict(X_anom)[0]

    print("\nHasil prediksi model:")
    print(f"  Prediksi hasil panen : {hasil_panen:,.0f} kg per hektar")
    print(f"  Prediksi mutu kopi   : {kelas}")
    for k, p in zip(encoder.classes_, proba):
        print(f"      probabilitas {k:<10}: {p * 100:5.1f}%")
    print(f"  Zona agroklimat      : {zona}")
    print(f"  Status kondisi kebun : "
          f"{'ANOMALI - perlu dicek lapangan' if status_anom == -1 else 'Normal'}")


# ===========================================================================
# PROGRAM UTAMA
# ===========================================================================
def main():
    # Tahap 1
    df_mentah = tahap1_data_collection()

    # Tahap 2 (pembersihan awal)
    df, encoder = tahap2_preprocessing_awal(df_mentah)

    # Tahap 3
    train, test = tahap3_splitting(df)

    # Tahap 2 (lanjutan): imputasi & scaling, di-fit pada training saja
    prep = tahap2_lanjutan_imputasi_scaling(train[FITUR], test[FITUR])

    ytr_reg, yte_reg = train[TARGET_REG].values, test[TARGET_REG].values
    ytr_clf, yte_clf = train["mutu_kode"].values, test["mutu_kode"].values

    # Tahap 4
    rf_reg, pred_reg, mae, r2 = tahap4a_regresi(
        prep["Xtr"], prep["Xte"], ytr_reg, yte_reg)
    clf, cm, m_clf = tahap4b_klasifikasi(
        prep["Xtr"], prep["Xte"], ytr_clf, yte_clf, encoder)

    # Tahap 5: unsupervised memakai seluruh data bersih (tanpa label target)
    urut = df.index
    X_all_imp = pd.concat([prep["Xtr_imp"], prep["Xte_imp"]]).loc[urut]
    X_all_scaled = prep["scaler"].transform(X_all_imp)
    df_all_raw = X_all_imp.copy()
    df_all_raw[TARGET_REG] = df[TARGET_REG]
    df_all_raw["mutu_kode"] = df["mutu_kode"]
    flag_asli = df[KOLOM_FLAG].values

    km, _, nama_zona, peta_nama, _ = tahap5a_clustering(X_all_scaled, df_all_raw)
    iso, is_anom, _, _ = tahap5b_anomaly(df_all_raw, flag_asli)
    tahap5c_visualisasi(yte_reg, pred_reg, mae, r2, cm, encoder, m_clf,
                        rf_reg, clf, km, peta_nama, X_all_scaled, nama_zona,
                        df_all_raw, is_anom)

    # Tahap 6
    tahap6_inference(prep, rf_reg, clf, encoder, km, peta_nama, iso)

    judul("SELESAI")
    print(f"File keluaran: {OUTPUT_GAMBAR}, {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
