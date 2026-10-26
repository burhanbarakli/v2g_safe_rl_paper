# v2g_safe_rl_paper

Attention-Based Safe Reinforcement Learning with a Feasibility-Preserving Safety Layer for Size-Agnostic V2G
Charging Control (çalışma başlığı). Hedef dergi: SAUCIS.

## Klasörler

| Klasör | İçerik |
|---|---|
| `code/` | `v2gsafe` paketi, `train.py`, `evaluate.py`, `reproduce.sh`, testler. Kurulum ve çalıştırma: [`code/README.md`](code/README.md) |
| `code/results/` | Çekirdek çalışmanın ham sonuçları (aşağıya bakın) |
| `analysis/` | Tablo, şekil ve istatistikler; `findings_numbers.md` (yalnızca sayılar) |
| `paper/` | LaTeX makale ve ek dosya |
| `method/` | Yöntem belgeleri ve tasarım kararları (DD1–DD44) |
| `review/`, `literature/` | İç hakem turları, literatür matrisi ve kaynaklar |

Devir notu: [`HANDOVER_TR.md`](HANDOVER_TR.md). Deney çalıştırma notu: [`RUN_NOTES_TR.md`](RUN_NOTES_TR.md).

## Sonuçlar (çekirdek set, 23 Ekim 2026)

153.100 epizot, 30 yöntem, 43 test yapılandırması, 3.550 oracle epizodu, 18 eğitim. Ortam, süreler ve plandan
sapmalar `RUN_NOTES_TR.md` içinde.

> **Kapsam (yazar kararı):** Tüm sonuçlar öğrenen yöntem başına **2 seed** (0, 1) ile üretildi; plandaki 5 seed
> (ve `attn_safe_trainN50` için 3 seed) kullanılmadı. **Genişletilmiş set çalıştırılmadı ve çalıştırılmayacak**:
> ablasyonlar (`beta0`, `curriculum`, `small`, `penalty_dense`), ceza ağırlığı duyarlılığı (w = 1, w = 100),
> `--stage sensitivity` (SoC 0,8, Δt = 5 dk, aşınma maliyeti, deşarj fiyat çarpanı), eMPC-24h, Gurobi eMPC ve
> EV-GNN. Bu nedenle T4/Şekil S2'de ablasyon satırları, TS2'de ρ_tr = 3 kW dışındaki duyarlılık satırları ve
> örnek gün şekli (F5, `fig5_example_day`) yok. Makale ve `code/README.md` hâlâ 5 seed yazıyor; metin buna göre
> güncellenmeli.

**Depoda** (`code/results/`, README §9 gönderim içeriği):

- `raw/<exp_id>/episodes.csv`, `train_log.csv`, `meta.json`, `config.yaml`
- `raw/oracle__<test_cfg_id>/episodes.csv`
- `tuning/<method>/selected*.yaml`, `tuning/<method>/<tag>/validation.json`
- `test_sets/*.json` (replay seed'leri ve parmak izleri)
- `logs/` (her işin stdout/stderr çıktısı)
- `run_reports/`: `make test`, analiz testleri ve `scripts/benchmark.py` çıktıları; aşama logları
  (`pipeline/progress.log` ve aşama başına bir log); çalıştırmada kullanılan `run_core.sh` ve systemd birimi

**Google Drive'da** (git için fazla büyük ikili dosyalar):
[v2g_safe_rl_paper klasörü](https://drive.google.com/drive/folders/1FiZAGMjlq2j399njQTPmIriW8_I5lzbW)

| Dosya | Boyut | İçerik |
|---|---|---|
| `gonderim_checkpoints_2026-10-23.zip` | 276 MB | 18 `checkpoint_final.pt` (`code/results/raw/<exp_id>/` yapısında) |
| `results_caches_2026-10-23.zip` | 92 MB | `admission/`, `conformal/`, `test_sets/` önbellekleri (yeniden değerlendirme için) |
| `gonderim_paketi_2026-10-23.zip` | 36 MB | README §9 gönderim paketi (depodaki sonuçlar, analiz çıktıları, test ve benchmark logları, kod yaması) |
| `gonderim_checkpoints_penrew_2026-10-26.zip` | 31 MB | Tanı çalışması `attn_safe_penrew` için 2 `checkpoint_final.pt` (aşağıya bakın) |
| `gonderim_checkpoints_cheap_2026-10-26.zip` | 62 MB | Tanı çalışmaları `deepsets_safe_penrew` ve `attn_safe_penrew_entpen` için 4 `checkpoint_final.pt` |
| `SHA256SUMS.txt` | | Zip dosyalarının SHA-256 özetleri |

### Tanı çalışması: SA-Attn cezalı ödülle (26 Ekim 2026)

Performans analizi ([`review/performance_diagnosis_TR.md`](review/performance_diagnosis_TR.md)) sonrasında SA-Attn,
yalnızca ödül türü değiştirilerek (kâr yerine cezalı ödül, w_E = w_P = 10) 2 seed ile yeniden eğitildi ve aynı test
ızgarasında değerlendirildi: `code/configs/train_attn_safe_penrew.yaml`, sonuçlar
`code/results/raw/attn_safe_penrew__*`, loglar `code/results/run_reports/penrew/`. Aynı gün iki tanı çalışması daha
yapıldı (rapor §11): Deep Sets + cezalı ödül (`deepsets_safe_penrew`) ve SA-Attn + cezalı ödül + entropi cezası
(`attn_safe_penrew_entpen`); yapılandırmalar `code/configs/train_deepsets_safe_penrew.yaml`,
`code/configs/train_attn_safe_penrew_entpen.yaml`, loglar `code/results/run_reports/cheap/`.

Bu tanı çalışmaları `analysis/` tablo ve şekillerine ve makaleye **dahil değil**. `make_all.py` bu klasörler `raw/`
içindeyken çalıştırılırsa yöntemler genel stille tablolara girer; makale tablolarını yeniden üretmek için
`attn_safe_penrew*`, `deepsets_safe_penrew*` klasörlerini dışarıda bırakın.

Zip'leri depo kökünde açın; dosyalar `code/results/` altındaki yerlerine gelir. Ardından örneğin:

```bash
cd code
python evaluate.py --checkpoint results/raw/<exp_id>/checkpoint_final.pt --test-set configs/test_grid.yaml --jobs 8
python ../analysis/scripts/make_all.py --raw results/raw --out ../analysis
```

Depoda olmayanlar: ara checkpoint'ler (`checkpoint_last.pt`), tuning checkpoint'leri ve smoke sonuçları.
