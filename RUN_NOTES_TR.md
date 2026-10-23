# Deney çalıştırma notu (çekirdek set)

Tarih: 23 Ekim 2026. Durum: çekirdek set (tuning dahil) tamamlandı, analiz çıktıları üretildi. Genişletilmiş set,
EV-GNN ve örnek gün şekli (F5) çalıştırılmadı.

## 1. Ortam

| | |
|---|---|
| Donanım | AMD Ryzen 7 7700 (8 çekirdek / 16 iş parçacığı; WSL'e 12 iş parçacığı ve 31 GB bellek ayrıldı), NVIDIA GeForce RTX 5070 Ti 16 GB (sürücü 617.14) |
| İşletim sistemi | Windows 11 Pro, WSL2 Ubuntu 24.04 (çekirdek 6.18.40.1-microsoft-standard-WSL2) |
| Python / PyTorch | Python 3.13.15, PyTorch 2.14.1+cu130 (CUDA 13.0), `requirements.txt` sürümleri |
| EV2Gym | `55ef368c` (sabitlenmiş sürüm; tüm `meta.json` dosyalarında `ev2gym_commit_matches: true`), kaynak kodu değiştirilmedi |

## 2. Çalıştırma

Aşamalar ayrı paralellik ayarlarıyla, sırayla çalıştırıldı (`--seeds 2`):

| Aşama | Komut | Süre |
|---|---|---|
| Hazırlık | `--stage scenarios,admission,conformal --jobs 10` | 1,6 sa |
| Tuning | `--stage tune,tune_select --jobs 2 --device cuda` | 7,9 sa (84 eğitim + 84 doğrulama) |
| Eğitim | `--stage train --jobs 2 --device cuda` | 10,2 sa (18 eğitim) |
| Değerlendirme | `--stage evaluate,oracle,merge --jobs 10` | 2,5 sa (298 değerlendirme + 43 oracle işi) |
| Analiz | `--stage analysis` | 6 dk |

Sonuç: 153.100 epizot, 30 yöntem, 43 test yapılandırması, 3.550 oracle epizodu, 18 eğitim logu. Başarısız iş yok.

## 3. Plandan sapmalar (yazar kararı veya zorunlu düzeltme)

1. **Seed sayısı 5 yerine 2 (yazar kararı).** Öğrenen her yöntem 2 seed ile (0, 1) eğitildi. `attn_safe_trainN50`
   da 3 yerine 2 seed ile eğitildi (`pipeline.py`: sabit 3 seed artık `--seeds` ile sınırlanıyor). Makalede ve
   README'de 5 ve 3 seed yazan yerler güncellenmeli. 2 seed ile seedler arası t güven aralıkları çok geniş
   (serbestlik derecesi 1).
2. **EV2Gym'in kuramadığı eğitim epizotları (`vec_env.py`).** EV2Gym'in `residential_loads.csv` dosyasında
   2.177 negatif değer var. Seçilen 10 hanenin toplamı bir adımda negatif olursa, yük tahmini gürültüsünün
   standart sapması negatif çıkıyor ve `reset` `ValueError: scale < 0` ile çöküyor. PV tarafında `abs()`
   kullanılıyor, yük tarafında kullanılmıyor. İlk tuning denemesinde 8 işin hepsi bu yüzden aynı noktada çöktü
   (bu işler `code/results/tuning_failed_scale_lt0_20261022/` klasörüne taşındı, tuning sıfırdan başlatıldı).
   Düzeltme: eğitim örnekleyicisi, kurulamayan bir epizot seed'i gelirse aynı rastgele akıştan yeni bir seed
   çekiyor ve bunu loglara yazıyor. EV2Gym değiştirilmedi. Test, validation ve kalibrasyon replay'leri sabit
   kaldı; hepsi sorunsuz kuruldu.
   - Gerçekleşen yeniden çekim sayısı: tuning'de iş başına 1 (84 işin 84'ünde), eğitimde iş başına 2
     (18 işin 18'inde). Toplamda 4 farklı epizot seed'i söz konusu, çünkü env seed akışları
     (run seed, env index) ile belirleniyor.
   - Makalede sınırlama olarak yazılmalı: eğitim sırasında EV2Gym'in simüle edemediği nadir günler atlandı.
3. **Oracle LP'sinin sayısal sorunları (`oracle.py`).** 3.550 episodun 3'ünde ana oracle LP'si (2. aşama, kâr
   maksimizasyonu) hem dual simplex hem interior-point ile "Unknown" döndü ve `profit_eur` boş kaldı; analiz bu
   yüzden durdu. Yedek zincire iki adım eklendi:
   - primal simplex: 2 episot (`private_N200_rho3` replay 8, `public_N50_rho4` replay 21), durum `Optimal(primal)`;
   - eksik enerji toleransını 1e-6'dan 1e-8'e sıkılaştırmak: 1 episot (`public_N200_rho4` replay 9), durum
     `Optimal(slack1e-8)`. Bu plan 1e-6 tanımına göre de olurlu; değer gerçek optimumun hafif altında
     (bu episotta farkın en fazla ~2 € olduğu tahmin ediliyor).

   Diğer 3.547 episot birebir aynı kaldı (karşılaştırıldı). Eski dosyalar
   `code/results/oracle_before_primal_fallback_20261023/` klasöründe.
   Önemli gözlem: bu episotlarda katı oracle değeri toleransa çok duyarlı (ör. `private_N200_rho3` replay 8:
   tolerans 1e-8 → 36,85 €, 1e-6 → 37,03 €, 1e-4 → 40,00 €). Admission tablosu istekleri ulaşılabilir maksimuma
   çektiği için olurlu küme bu episotlarda çok dar. Tartışmada bir yöntem notu olarak yer almalı.

## 4. Testler ve ölçümler

- `make test`: 98 geçti, 1 atlandı (`make_test_final.log`). Yeni test: `tests/test_vec_env.py`.
- Analiz testleri: 36 geçti, 1 başarısız. `test_smoke_never_written_into_analysis` testi `analysis/figures/`
  klasöründe hiç PDF olmamasını bekliyor; klasörde Şekil 1 ve artık gerçek şekiller olduğu için başarısız oluyor.
  Smoke verisini engelleyen asıl kontrol çalışıyor (`SystemExit`). Testin güncellenmesi gerekiyor.
- `python scripts/benchmark.py`: `benchmark.log`.

## 5. Yapılmayanlar

- Genişletilmiş set (ablasyonlar, w-duyarlılık, eMPC-24h, Gurobi eMPC), `--stage sensitivity`, EV-GNN.
  Bu nedenle T4/S2 gibi tablo ve şekillerde ablasyon satırları boş olabilir.
- Örnek gün şekli (F5). Bu şekil `--log-steps` ile ayrı bir değerlendirme gerektiriyor.
- Kod değişiklikleri commit edilmedi.

## 6. Paket içeriği

`gonderim_paketi_2026-10-23.zip` (README §9):
- `code/results/raw/**/{episodes.csv,train_log.csv,meta.json,config.yaml}`
- `code/results/tuning/*/selected*.yaml`, `code/results/tuning/*/*/validation.json`
- `code/results/test_sets/*.json`
- `code/results/logs/` (tüm iş logları) ve `run_logs/` (aşama logları, `progress.log`)
- `analysis/` çıktıları (tables, figures, stats, `findings_numbers.md`, `load_report.txt`)
- `make_test_final.log`, `analysis_tests_final.log`, `benchmark.log`
- `code_changes.patch` (3 kod dosyası ve yeni test) ve bu not

Checkpoint'ler (`checkpoint_final.pt`, 18 adet, ~263 MB) ayrı bir pakette:
`gonderim_checkpoints_2026-10-23.zip`.
