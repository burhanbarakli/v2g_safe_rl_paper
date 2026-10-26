# Performans analizi: SA-Attn neden beklenen performansı vermedi?

Tarih: 26 Ekim 2026. Veri: çekirdek set (2 seed), `code/results/raw`. Ana ızgara: 3 profil × N ∈ {5, 10, 20, 50, 100, 200}
× ρ_tr ∈ {4, 7} kW (ρ_tr = 3 kW stres seviyesi ve literatür yapılandırması hariç). J^pen ortak gereksinime göre
(DD46): kâr − 10·ENS_shared − 10·aşım. Açık = yumuşak cezalı oracle − J^pen, şarj noktası ve episot başına €.
Betik: `review/performance_diagnosis.py` (WSL venv ile çalıştırıldı) (aşağıdaki tabloların hepsi bu betikten).

## Özet

SA-Attn'in geride kalmasının ana nedeni kâr değil, **olurlu olmayan (A5) episotlarda karşılanmayan enerji
cezası**. En iyi rakiplere göre açık farkının %58–83'ü bu cezadan geliyor. Bu cezanın tamamı olurlu olmayan adım içeren
episotlarda oluşuyor. Bu episotların payı %20; olurlu episotlarda SA-Attn'in açığı (0,88) fiyat eşiği kuralıyla aynı
düzeyde. Üç mekanizma birlikte çalışıyor:

1. **Ödül tasarımı:** SA-Attn (ve Deep Sets, MLP + katman) yalnızca kâr ödülüyle eğitiliyor (`base.yaml`:
   `reward.type: profit`). Karşılanmayan enerji eğitim ödülünde hiç görünmüyor. Politika, katmanın garantisine
   "güvenip" agresif V2G yapıyor ve garantinin koşulunu (aggregate olurluluk) bozan durumlara giriyor.
2. **Kapasiteyi görmeyen alt sınır:** Ayrılış alt sınırı her aracın kendi ulaşılabilirliğine bakıyor, trafonun
   ortak kapasitesine bakmıyor. Sıkı trafo (ρ_tr = 4) ve aynı anda ayrılan araçlar (workplace) bir araya gelince alt
   sınırların toplamı kapasiteyi aşıyor. Aynı katmanı kullanan eMPC de aynı oranda olurlu olmayan episot üretiyor; sorun
   RL'ye özgü değil, yapısal.
3. **Öğrenmenin ilerlememesi:** Eğitim boyunca kâr getirisi düz kalıyor, entropi artıyor, olurlu olmayan adım oranı
   yükseliyor. Attention mimarisi bu ortamda öğrenme avantajı sağlamıyor; Deep Sets en azından yavaşça iyileşiyor.

## 1. Yöntem karşılaştırması (ana ızgara, şarj noktası ve episot başına)

| Yöntem | Açık € | Kâr € | ENS cezası € | Şarj kWh | Deşarj kWh | Olurlu olmayan episot % | Alt sınırda % | Karşılanmayan araç % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-Attn | 1,273 | −2,058 | 0,428 | 22,0 | 10,7 | 19,8 | 34,3 | 1,03 |
| Deep Sets + katman | 0,946 | −1,868 | 0,291 | 20,5 | 9,2 | 12,8 | 31,5 | 0,69 |
| Attn + katman, alt sınırsız (cezalı ödül) | 0,824 | −1,945 | 0,091 | 19,9 | 8,5 | 0,0 | 5,8 | 0,41 |
| Attn cezalı, katman yalnızca kullanımda | 0,928 | −2,073 | 0,067 | 15,4 | 4,1 | 5,8 | 3,4 | 0,11 |
| MLP + katman | 1,271 | −2,030 | 0,442 | 19,8 | 8,6 | 15,9 | 40,1 | 1,69 |
| Fiyat eşiği + katman | 0,822 | −1,980 | 0,054 | 17,4 | 6,0 | 5,6 | 29,6 | 0,14 |
| ALAP + katman | 1,042 | −2,227 | 0,027 | 11,4 | 0,1 | 6,0 | 36,7 | 0,06 |
| LLF + katman | 1,091 | −2,303 | 0,000 | 11,4 | 0,0 | 0,0 | 0,0 | 0,00 |
| eMPC-HiGHS + katman | 1,157 | −1,954 | 0,416 | 22,9 | 11,6 | 19,8 | 57,6 | 1,08 |
| Oracle (bilgi) | — | −1,606 (katı) | — | 20,8 | 9,4 | — | — | — |

Notlar: Alt sınırsız varyantta olurlu olmayan adım tanım gereği oluşmaz (alt sınır yalnızca deşarj limiti). Cezalı
ikizin ENS'si DD46 sonrası ortak gereksinime göre hesaplandı.

## 2. Açık farkının ayrıştırması (SA-Attn eksi rakip; pozitif = SA-Attn kötü)

| Rakip | Açık farkı | Kârdan | ENS cezasından |
|---|---:|---:|---:|
| Fiyat eşiği + katman | +0,452 | +0,078 (%17) | +0,374 (%83) |
| Alt sınırsız varyant | +0,449 | +0,113 (%25) | +0,337 (%75) |
| Cezalı ikiz + katman | +0,345 | −0,016 | +0,361 (%105) |
| Deep Sets + katman | +0,328 | +0,190 (%58) | +0,138 (%42) |
| eMPC + katman | +0,116 | +0,104 | +0,012 |

## 3. Olurlu olmayan episotlar (SA-Attn)

| | Pay | Açık € | Kâr € | ENS cezası € |
|---|---:|---:|---:|---:|
| Olurlu olmayan adım yok | %80,2 | 0,879 | −2,008 | 0,000 |
| En az bir olurlu olmayan adım | %19,8 | 2,874 | −2,259 | 2,166 |

ENS cezasının **%100'ü** olurlu olmayan episotlardan geliyor (Sonuç 1 ile tutarlı: garanti koşullu olarak tutuyor).

## 4. Nerede bozuluyor? (açık € ve olurlu olmayan episot %)

| Profil | ρ_tr | SA-Attn açık | Deep Sets | Alt sınırsız | Fiyat eşiği | SA-Attn olurlu olmayan % | Deep Sets % | Fiyat eşiği % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| private | 4 | 1,577 | 1,242 | 0,951 | 0,915 | 24,4 | 15,1 | 7,8 |
| private | 7 | 1,237 | 0,899 | 1,075 | 0,997 | 1,4 | 0,8 | 0,2 |
| public | 4 | 1,158 | 0,997 | 0,840 | 0,853 | 18,2 | 12,3 | 8,4 |
| public | 7 | 0,892 | 0,707 | 0,824 | 0,832 | 2,0 | 1,4 | 0,8 |
| **workplace** | **4** | **2,209** | 1,383 | 0,750 | 0,714 | **64,3** | 43,9 | 14,8 |
| workplace | 7 | 0,566 | 0,445 | 0,504 | 0,619 | 8,3 | 3,0 | 1,4 |

- Sorun sıkı trafoda (ρ_tr = 4) yoğunlaşıyor; en kötüsü workplace (benzer saatlerde gelip giden araçlar).
- Gevşek trafoda (ρ_tr = 7) SA-Attn rekabetçi: workplace'te fiyat eşiğinden iyi, alt sınırsız varyanta yakın.
- N'ye göre: en zor N = 5 (açık 2,29); N ≥ 20'de SA-Attn ≈ 0,9, Deep Sets ≈ 0,7, fiyat eşiği ≈ 0,79.

## 5. Eğitim dinamiği

| Yöntem (seed) | Kâr getirisi: başlangıç → son | Entropi: başlangıç → son | Olurlu olmayan adım oranı: başlangıç → son |
|---|---|---|---|
| SA-Attn (0) | −32,1 → −31,7 | 0,76 → 1,04 | 0,009 → 0,014 |
| SA-Attn (1) | −31,8 → −32,4 | 0,73 → 1,23 | 0,009 → 0,013 |
| Deep Sets (0) | −32,4 → −30,1 | 0,81 → 1,32 | 0,006 → 0,010 |
| Deep Sets (1) | −31,6 → −29,9 | 0,85 → 1,29 | 0,007 → 0,009 |
| Alt sınırsız (0) | −30,7 → −29,9 | 0,48 → 0,17 | 0 → 0 |
| MLP + katman (0) | −32,6 → −32,1 | 0,96 → 1,33 | 0,004 → 0,004 |

(Eğitim karışımı N ∈ {10, 20}; değerler eğitimin ilk ve son %10'unun ortalaması, episot başına €.)

- SA-Attn'in kâr getirisi eğitim boyunca düz: politika, rastgele başlangıç + katman düzeyinin ötesine geçmiyor.
- Entropi artıyor. Katman pek çok ham aksiyonu aynı uygulanan güce eşlediği için (alt sınırda %34, ortalama projeksiyon
  mesafesi 0,52), ödül ham aksiyondaki değişime duyarsız kalıyor; bu, kredi atamasını zayıflatıyor ve entropi bonusunu
  dengeleyen bir sinyal bırakmıyor. Cezalı ödülle eğitilen alt sınırsız varyantta entropi düşüyor ve politika
  belirginleşiyor.
- Olurlu olmayan adım oranı eğitimde artıyor: kâr ödülü, garantinin koşulunu bozan davranışı cezalandırmıyor.
- Tuning yelpazesi dar (SA-Attn validation J^pen: en iyi −41,6, medyan −45,7); 0,5M adımlık tuning bütçesi
  hiperparametrenin büyük fark yaratmadığını gösteriyor.

## 6. Davranış: fazla V2G, öngörüsüz

SA-Attn şarj noktası başına 22,0 kWh şarj ve 10,7 kWh deşarj yapıyor; bu, mükemmel bilgili oracle'ın (20,8 / 9,4)
bile üzerinde. Ancak kârı (−2,06) daha az işlem yapan fiyat eşiği kuralından (−1,98) düşük. Öngörü olmadan yapılan
arbitraj kâr getirmiyor; üstelik deşarj, araçları geç başlama yörüngesine (alt sınıra) itiyor ve ayrılış öncesi
kapasite sıkışmasını büyütüyor. Aynı tablo eMPC + katmanda da görülüyor (23 / 11,6 kWh, %19,8 olurlu olmayan episot).

## 7. Attention neden Deep Sets'i geçmedi?

- Şarj noktaları arasındaki koordinasyonu katmanın ortak seviyesi (λ) zaten yapıyor; attention'ın noktalar arası
  etkileşimi burada ek bilgi taşımıyor.
- Aynı ödül ve katmanla Deep Sets eğitimde yavaşça iyileşiyor, attention iyileşmiyor; Deep Sets daha az olurlu olmayan
  episot üretiyor (%12,8'e karşı %19,8).
- Seed farkı küçük (SA-Attn 1,24 / 1,31; Deep Sets 0,99 / 0,90); sıralama seed gürültüsünden kaynaklanmıyor.
- Tuning: Deep Sets lr = 1e-3, attention lr = 3e-4 seçti; kısa tuning hızlı öğrenenleri kayırmış olabilir.

## 8. Ne yapılabilir? (yeni çalıştırma gerektirir; karar yazarın)

| Öneri | Beklenen etki | Maliyet (bu makinede) |
|---|---|---|
| **SA-Attn'i cezalı ödülle eğitmek** (`reward: {type: penalty}`), katman ve alt sınır aynı kalır | ENS cezası, alt sınırsız varyant ve cezalı ikizdeki düzeye inebilir (açık farkının %75–83'ü) | 1 yapılandırma satırı; 2 seed × ~1 sa eğitim + ~2 sa değerlendirme |
| Olurlu olmayan adım veya alt sınır açığı için yoğun ceza (`penalty_dense`) | Politikayı kapasite sıkışmasından uzak tutar | Aynı |
| **Kapasiteye duyarlı alt sınır** (trafonun gelecekteki ortak kapasitesine bakan ileri bakışlı sınır, G4 Option B) | Yapısal sorunu çözer; eMPC ve kurallara da fayda | Yöntem eki + yeni ispat kontrolü + ~6–8 GPU-sa |
| Deşarjı alt sınıra yakınken kısıtlamak | V2G'nin sıkışmaya katkısını azaltır | Küçük kod değişikliği + yeniden eğitim |
| Daha uzun eğitim, daha geniş tuning, entropi katsayısı 0 | Öğrenmenin ilerlemesine yardım edebilir; tek başına yetmesi beklenmez | Saatler |

## 9. Makale için anlamı

- Sonuçlar ana iddiaları destekliyor: trafo garantisi her koşulda tuttu, ayrılış garantisi koşullu olarak tuttu,
  sıfır-atış ölçekleme referansla aynı düzeyde.
- "Attention" iddiası desteklenmiyor; Option A (DD45) çerçevesi bununla tutarlı.
- Yeni bulgu olarak yazılabilir: kâr ödülü + garanti katmanı, koşulu bozan davranışı cezalandırmadığı için "katmana
  güvenen" bir politika üretiyor. Bu, güvenli RL literatüründe katmanla eğitimin bir tuzağı olarak tartışılabilir.
- Bu yorumlar SAUCIS politikası gereği yazarın kendi değerlendirmesiyle yazılmalı.

## 10. Test: SA-Attn cezalı ödülle (`attn_safe_penrew`, 26 Ekim 2026)

SA-Attn ile tek fark ödül türü (`configs/train_attn_safe_penrew.yaml`: `reward: {type: penalty}`, w_E = w_P = 10);
katman, alt sınır, admission, β ve ayarlı hiperparametreler (attn_safe tuning seçimi, ayrıca ayarlanmadı) aynı.
2 seed × 3M adım (RTX 5070 Ti, iki seed paralel, 2,2 sa), ardından 43 test yapılandırmasında değerlendirme.
Çalışma ve loglar: WSL `~/v2g_run/run_penrew.sh`, `~/v2g_logs/penrew/`. Önceki tüm sonuç dosyaları değişmedi
(karşılaştırıldı).

| | SA-Attn (kâr ödülü) | **SA-Attn (cezalı ödül)** | Deep Sets + katman | Alt sınırsız varyant | Fiyat eşiği + katman |
|---|---:|---:|---:|---:|---:|
| Açık € / şarj noktası | 1,273 | **0,909** | 0,946 | 0,824 | 0,822 |
| Kâr € | −2,058 | −2,016 | −1,868 | −1,945 | −1,980 |
| ENS cezası € | 0,428 | **0,106** | 0,291 | 0,091 | 0,054 |
| Olurlu olmayan episot % | 19,8 | **7,7** | 12,8 | 0 | 5,6 |
| Karşılanmayan araç % | 1,03 | **0,29** | 0,69 | 0,41 | 0,14 |
| Deşarj kWh | 10,7 | 7,8 | 9,2 | 8,5 | 6,0 |
| Seed 0 / 1 açık | 1,24 / 1,31 | 0,93 / 0,88 | 0,99 / 0,90 | 0,88 / 0,77 | — |

| Profil, ρ_tr | SA-Attn | Cezalı ödül | Olurlu olmayan %: SA-Attn → cezalı |
|---|---:|---:|---:|
| workplace, 4 | 2,209 | **0,923** | 64,3 → 21,4 |
| private, 4 | 1,577 | 1,043 | 24,4 → 11,2 |
| public, 4 | 1,158 | 0,927 | 18,2 → 9,4 |
| workplace, 7 | 0,566 | 0,574 | 8,3 → 1,9 |

| N | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| SA-Attn | 2,292 | 1,290 | 0,973 | 0,921 | 0,881 | 0,902 |
| Cezalı ödül | 1,176 | 0,854 | 0,826 | 0,850 | 0,826 | 0,854 |
| Deep Sets | 1,663 | 0,984 | 0,722 | 0,698 | 0,649 | 0,675 |
| Fiyat eşiği | 0,956 | 0,790 | 0,788 | 0,793 | 0,774 | 0,789 |

Eğitim: olurlu olmayan adım oranı artık eğitimde düşüyor (seed 0: 0,007 → 0,004; kâr ödülünde 0,009 → 0,014).
Entropi yine artıyor (0,82 → 1,03 ve 0,72 → 1,32); kâr getirisi yalnızca seed 0'da belirgin iyileşiyor.

**Sonuç.**
- Mekanizma 1 doğrulandı: ödülü değiştirmek tek başına açığı %29 (1,273 → 0,909), ENS cezasını %75
  (0,428 → 0,106), olurlu olmayan episotları %61 (19,8 → 7,7) azalttı. İki seed'de de aynı yönde.
- En büyük kazanç tam da sorunun yoğunlaştığı yerde: workplace/ρ=4 (2,21 → 0,92) ve N = 5 (2,29 → 1,18).
- Kalan fark: cezalı ödüllü SA-Attn hâlâ fiyat eşiği kuralının (0,822) ve alt sınırsız varyantın (0,824) gerisinde;
  N ≥ 20'de Deep Sets'in (≈ 0,65–0,72) gerisinde. Ortalama açıkta Deep Sets'i geçmesi küçük N'den geliyor.
- Mekanizma 2 (kapasiteyi görmeyen alt sınır) ve 3 (öğrenmenin zayıf ilerlemesi, artan entropi) devam ediyor.
  Kalan açık için kapasiteye duyarlı alt sınır ve Deep Sets'in cezalı ödülle denenmesi sıradaki adaylar.
- Attention avantajı bu testle de ortaya çıkmadı.

## 11. Ucuz deneyler: Deep Sets + cezalı ödül, SA-Attn + cezalı ödül + entropi cezası (26 Ekim 2026)

- `deepsets_safe_penrew` (`configs/train_deepsets_safe_penrew.yaml`): deepsets_safe ile tek fark ödül türü (cezalı);
  deepsets_safe tuning seçimi (lr 1e-3, minibatch 512, ent 0).
- `attn_safe_penrew_entpen` (`configs/train_attn_safe_penrew_entpen.yaml`): attn_safe_penrew + entropi cezası
  (`--set ppo.ent_coef=-0.01`; log-std üst sınırı kodda sabit olduğu ve yeni bir yapılandırma anahtarı tüm çalışmaların
  hash'ini değiştireceği için entropi doğrudan cezalandırıldı).
- 2 seed × 3M adım, iki eğitim paralel (Deep Sets 1,8 sa, SA-Attn 1,5 sa), sonra 43 yapılandırmada değerlendirme.
  Çalışma: WSL `~/v2g_run/run_cheap.sh`, loglar `~/v2g_logs/cheap/`. Önceki 34 yöntem sonuç dosyası değişmedi
  (karşılaştırıldı).

| Şarj noktası ve episot başına | SA-Attn | SA-Attn cezalı | SA-Attn cezalı + entropi cezası | Deep Sets | **Deep Sets cezalı** | Fiyat eşiği + katman |
|---|---:|---:|---:|---:|---:|---:|
| Açık € | 1,273 | 0,909 | 1,101 | 0,946 | **0,843** | 0,822 |
| Kâr € | −2,058 | −2,016 | −2,118 | −1,868 | −1,911 | −1,980 |
| ENS cezası € | 0,428 | 0,106 | 0,195 | 0,291 | 0,145 | 0,054 |
| Olurlu olmayan episot % | 19,8 | 7,7 | 12,3 | 12,8 | 7,6 | 5,6 |
| Karşılanmayan araç % | 1,03 | 0,29 | 0,25 | 0,69 | 0,26 | 0,14 |
| Seed 0 / 1 açık | 1,24 / 1,31 | 0,93 / 0,88 | 1,06 / 1,14 | 0,99 / 0,90 | 0,93 / 0,76 | — |

| N | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| SA-Attn cezalı | 1,176 | 0,854 | 0,826 | 0,850 | 0,826 | 0,854 |
| SA-Attn cezalı + entropi cezası | 1,678 | 1,007 | 0,930 | 0,948 | 0,941 | 0,941 |
| Deep Sets | 1,663 | 0,984 | 0,722 | 0,698 | **0,649** | **0,675** |
| Deep Sets cezalı | 1,259 | 0,818 | **0,716** | 0,715 | 0,701 | 0,715 |
| Fiyat eşiği + katman | **0,956** | **0,790** | 0,788 | 0,793 | 0,774 | 0,789 |

| Profil, ρ_tr | Deep Sets cezalı | Fiyat eşiği | Olurlu olmayan %: Deep Sets cezalı / fiyat eşiği |
|---|---:|---:|---:|
| workplace, 4 | 0,944 | 0,714 | 22,9 / 14,8 |
| private, 4 | 0,992 | 0,915 | 9,2 / 7,8 |
| public, 4 | 0,888 | 0,853 | 9,8 / 8,4 |
| private, 7 | 0,957 | 0,997 | 0,7 / 0,2 |
| public, 7 | 0,752 | 0,832 | 0,8 / 0,8 |
| workplace, 7 | 0,526 | 0,619 | 2,4 / 1,4 |

Eğitim: entropi cezası entropiyi 0,6/0,43'ten −1,30/−1,44'e düşürdü (aksiyon std ≈ 0,07); seed 1'de kâr getirisi
eğitimde kötüleşti (−31,4 → −33,0). Deep Sets cezalıda entropi yine arttı (0,89 → 1,33).

**Sonuçlar.**
1. Cezalı ödül iki mimaride de işe yarıyor: Deep Sets'te açık 0,946 → 0,843, ENS cezası yarıya indi (0,291 → 0,145),
   olurlu olmayan episotlar 12,8 → 7,6 %.
2. **En iyi öğrenen yöntem artık Deep Sets + cezalı ödül.** N ≥ 20'de fiyat eşiği kuralını geçiyor (0,70–0,72'ye karşı
   0,77–0,79) ve gevşek trafoda (ρ = 7) her profilde önde; ancak N ≤ 10'da ve sıkı trafoda (ρ = 4) geride, bu yüzden
   ortalamada hâlâ biraz geride (0,843'e karşı 0,822). N ≥ 50'de cezalı ödül Deep Sets'i biraz kötüleştiriyor
   (0,649–0,698'e karşı 0,701–0,715): kâr-ceza dengesi büyük istasyonda kârdan biraz feda ediyor.
3. Mimari: her karşılaştırmada Deep Sets attention'dan iyi; attention için avantaj çıkmadı.
4. **Entropi cezası (−0,01) zararlı:** politika fazla erken deterministik hâle geldi, açık 0,909 → 1,101 kötüleşti.
   Artan entropi zayıf öğrenmenin nedeni değil, belirtisi; bu yönden vazgeçilmeli (ya da çok daha küçük katsayı).
5. Kalan açık sıkı trafo + workplace + küçük N'de yoğunlaşıyor ve olurlu olmayan episotlarla birlikte gidiyor: bu,
   kapasiteyi görmeyen alt sınır (mekanizma 2) ile tutarlı. Sıradaki aday kapasiteye duyarlı alt sınır.
6. Deep Sets cezalıda seed farkı büyük (0,93 / 0,76); fiyat eşiğiyle ortalama farkı (0,02) 2 seed ile ayırt etmek
   için yetersiz.
