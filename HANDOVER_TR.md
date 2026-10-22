# Devir notu: deneyleri çalıştırma ve yazar görevleri

Tarih: 22 Ekim 2026. Durum: Faz 3 (kod altyapısı) ve Faz 5'in sonuç dışı kısımları tamamlandı. Deneyler senin bilgisayarında çalıştırılacak (G1-D5).

## 1. Klasörde ne var?

| Klasör | İçerik |
|---|---|
| `code/` | `v2gsafe` paketi; `train.py`, `evaluate.py`, `reproduce.sh`, `configs/`, 97 test (hepsi geçiyor), `README.md` (kurulum ve çalıştırma, İngilizce) |
| `analysis/` | Ham sonuçlardan tablo, şekil ve istatistik üreten scriptler (37 test geçiyor); kavramsal şekiller (`figures_conceptual/`); Şekil 1 (`figures/fig_architecture.*`) |
| `paper/manuscript/` | LaTeX makale (`main.tex`, `sections/`), `main.pdf`, anonim sürüm `main_anonymous.pdf` |
| `paper/supplementary/` | Ek dosya (tam ispatlar, ek tablo ve şekiller) |
| `paper/manuscript.md`, `paper/supplementary.md` | Okuma kolaylığı için Markdown kopyaları |
| `method/` | Yöntem belgeleri ve tüm tasarım kararları (`design_decisions.md`, DD1–DD44) |
| `review/` | İspat kontrolü, iki iç hakem turu, tutarlılık raporları |
| `literature/` | 114 doğrulanmış kaynak, makale matrisi, gap analizi |

## 2. Deneyleri çalıştırma (özet; ayrıntılar `code/README.md` içinde)

1. Kurulum: `cd code && python3 -m venv ../venv && source ../venv/bin/activate && pip install -r requirements.txt && bash scripts/install_ev2gym.sh && make test`. Python 3.10–3.13 ile çalışır.
2. Kısa duman testi (yaklaşık 25 dk): `./reproduce.sh --smoke --jobs 2`.
3. Ana deneyler:
   - GPU varsa: `./reproduce.sh --jobs 4 --device cuda`.
   - Yalnız CPU varsa: `--device` vermeden çalıştır.
   - Hiperparametre ayarı (tuning) otomatik olarak önce çalışır. Atlamak istersen hiperparametre dosyasını açıkça vermen gerekir.
4. Tahmini süreler:
   - Çekirdek set: yaklaşık 32 GPU-saat + 82–132 CPU-saat. GPU yoksa yaklaşık 130–185 düğüm-saat (16 çekirdek) + 82–132 CPU-saat.
   - Genişletilmiş set (isteğe bağlı): ek olarak yaklaşık 31–41 GPU-saat + 68–108 CPU-saat.
   - Bunlar tahmindir; bulut ortamındaki kısa zaman ölçümlerine dayanır.
5. İş birden fazla makineye seed veya yönteme göre bölünebilir (README §3).
6. Sonra şu komutu çalıştır: `python ../analysis/scripts/make_all.py --raw results/raw --out ../analysis`. Tablo ve şekiller `analysis/` altına yazılır. Makale bu dosyaları `\IfFileExists` ile otomatik alır; `make` ile yeniden derlemen yeterli.
7. Bana geri göndermen gerekenler: `results/raw/**/episodes.csv`, `train_log.csv`, `meta.json` ve `analysis/` çıktıları. Checkpoint dosyaları isteğe bağlı.

## 3. Senin yapman gerekenler (makale)

1. **Yenilik kontrolü (en önemli):** Şu çalışmaların tam metinleri okunmalı:
   - Orfanoudakis vd. 2026, "Can A.I. Revolutionize EV Dispatch?" (ECML PKDD 2025 çalıştayları, CCIS 2841);
   - Song vd. 2025 (Applied Energy 397, 126315);
   - Eland vd. 2026.

   Kütüphane erişimi yoksa yazarlardan preprint istenebilir. Makaledeki yenilik iddiası bu nedenle koşullu yazıldı.
2. **Başlık:** Çalışma başlığı DD30'da. Son başlık senin kararın.
3. **SAUCIS yapay zekâ politikası:**
   - Yapay zekânın taslak hâline getirdiği bölümler (Giriş, İlgili Çalışmalar, Yöntem, Özet) senin tarafından esaslı biçimde revize edilmeli.
   - Sonuçların yorumu ve Tartışma bölümü senin tarafından yazılmalı. Bu bölümlerdeki `\RES{}` yer tutucuları yalnızca senin deney çıktılarınla doldurulmalı.
   - Yapay zekâ beyanının son hâlini sen onaylamalısın (`paper/declarations.md`). İspatları bir insan uzman da kontrol ettiyse beyana eklenebilir.
4. **Şablon ve kaynak stili:**
   - SAUCIS LaTeX şablonunu (`saucis.cls`) ve `IEEEtran.bst` dosyasını `paper/manuscript/` klasörüne koy. Bu dosyalar buradan indirilemedi (ağ engeli). Belge, dosyaları bulursa otomatik olarak onları kullanır.
   - Şekiller dergi için EPS olarak dışa aktarılabilir.
5. **Bilgi formu ve yazar bilgileri:**
   - Yazar adları, ORCID, kurum ve yazar katkıları.
   - Finansman, etik beyanı, çıkar çatışması.
   - Donanım bilgisi ve kod deposu bağlantısı. Bunların hepsi `\TODO` olarak işaretli.
6. **Teknik kontrol:** Senaryo istatistiklerini ve kapalı form kontrollerini kendi ortamında yeniden üret (`method/profile_check.py`, `method/check_A1.py`).

## 4. Bilinen sınırlar

- EV2Gym'in kayıpsız enerji muhasebesi (DD1).
- Ayrılış zamanı ve istenen enerjinin bilinmesi varsayımı; konformal uzantı bunu kısmen gevşetiyor.
- Tek trafo, gerilim kısıtı yok.
- Yalnızca simülasyon.

Bunların hepsi makalenin Tartışma bölümünde yazılı.
