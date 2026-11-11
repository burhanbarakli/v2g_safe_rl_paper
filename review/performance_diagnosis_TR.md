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

## 12. Global bilgiye doğrudan erişim: `attn_gin_safe_penrew`, `attn_gctx_safe_penrew` (8–9 Kasım 2026)

**Soru.** Fiyat, fiyat tahmini, saat, trafo kapasitesi ve alt sınırların toplamı gözlemde yalnızca global token'da
bulunuyor. Deep Sets bu bilgiyi her şarj noktasının girdisine doğrudan ekliyor. SA-Attn'de ise bir şarj noktası bu
bilgiye ancak dikkat mekanizmasıyla global token'a bakarak ulaşabiliyor. Eğitilmiş checkpoint'ler üzerinde yapılan
ölçümde (tabloda "probe") kâr ödüllü SA-Attn global token'a Deep Sets'in yaklaşık yarısı kadar duyarlı çıktı ve fiyata
pratikte tepki vermedi. Bu bölümde, global bilgiyi attention modeline doğrudan vermenin farkı kapatıp kapatmadığı
test ediliyor.

**Varyantlar** (`code/v2gsafe/models/attention_policy.py`, `model.arch`):
- `attention_gin`: her şarj noktasının gömmesine global özellikler eklenir, e_i = W_c [ξ_i; ξ_0]
  (aktör ve kritik). Parametre sayısı 610.370'ten 615.234'e çıkar.
- `attention_gctx`: buna ek olarak aktör başlığı Deep Sets'teki gibi [h_i; h_0; dolu j'ler üzerinden h_j
  ortalaması] girdisini alır. Parametre sayısı 631.618 olur.

İki varyantta da Lemma N1 korunuyor (yeni testler `tests/test_models.py` dosyasında; tüm paket 102 geçti, 1 atlandı).
Ödül cezalı (w = 10); güvenlik katmanı, alt sınır, admission ve hiperparametreler `attn_safe_penrew` ile aynı
(attn_safe tuning seçimi). Eğitim 2 seed × 3M adım; her turda iki eğitim paralel çalıştı (2 × 1,9 sa). Ardından 43
yapılandırmada değerlendirme yapıldı (31 dk). Çalıştırma betiği WSL'de `~/v2g_run/run_gattn.sh` (systemd birimi
`v2g-gattn`). Önceki 81 `episodes.csv` dosyası değişmedi (md5 ile karşılaştırıldı).

| Şarj noktası ve episot başına | SA-Attn cezalı | **gin** | **gctx** | Deep Sets cezalı | Fiyat eşiği + katman |
|---|---:|---:|---:|---:|---:|
| Açık € | 0,909 | 0,954 | 0,931 | 0,843 | 0,822 |
| Kâr € | −2,016 | −1,961 | −1,974 | −1,911 | −1,980 |
| ENS cezası € | 0,106 | 0,206 | 0,170 | 0,145 | 0,054 |
| Olurlu olmayan episot % | 7,7 | 11,9 | 9,2 | 7,6 | 5,6 |
| Deşarj kWh | 7,8 | 9,0 | 8,1 | 8,7 | 6,0 |
| Seed 0 / 1 açık | 0,93 / 0,88 | 0,85 / 1,06 | 0,89 / 0,98 | 0,93 / 0,76 | — |
| Probe: global token'a duyarlılık | 0,94 | 0,84 | 0,93 | 1,11 | — |
| Probe: corr(aksiyon, fiyat) | −0,06 | **−0,16** | **−0,13** | −0,14 | — |

(Probe: 4 yapılandırma × 2 replay. Duyarlılık = global token aynı çalışmanın başka bir adımındakiyle değiştirildiğinde
|Δμ| / sd(μ). Kâr ödüllü modellerde probe değerleri: SA-Attn 0,57 / −0,03, Deep Sets 1,03 / −0,15.)

| Açık €, N'ye göre | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| SA-Attn cezalı | 1,176 | 0,854 | 0,826 | 0,850 | 0,826 | 0,854 |
| gin | 1,459 | 0,956 | 0,807 | 0,785 | 0,753 | 0,772 |
| gctx | 1,323 | 0,942 | 0,814 | 0,789 | 0,787 | 0,788 |
| Deep Sets cezalı | 1,259 | 0,818 | 0,716 | 0,715 | 0,701 | 0,715 |
| Fiyat eşiği + katman | 0,956 | 0,790 | 0,788 | 0,793 | 0,774 | 0,789 |

| Seed başına açık € | N ≤ 10 | N = 20 | N ≥ 50 | Olurlu olmayan episot % (ρ = 4) |
|---|---:|---:|---:|---:|
| SA-Attn cezalı s0 / s1 | 1,05 / 0,98 | 0,84 / 0,82 | 0,87 / 0,82 | 13,7 / 14,3 |
| gin s0 / s1 | 1,00 / 1,41 | 0,75 / 0,86 | 0,74 / 0,81 | 15,5 / 27,6 |
| gctx s0 / s1 | 0,98 / 1,28 | 0,84 / 0,79 | 0,82 / 0,76 | 14,0 / 19,9 |
| Deep Sets cezalı s0 / s1 | 1,17 / 0,91 | 0,77 / 0,67 | 0,76 / 0,66 | 14,5 / 13,4 |
| Fiyat eşiği | 0,87 | 0,79 | 0,79 | 10,3 |

Eğitim: son %10'da attention varyantları güncelleme başına 7,6–9,1 epoch çalıştı (KL erken durdurması, KL ≈ 0,010–0,014).
Deep Sets cezalı 10 epoch'un tamamını çalıştı (KL ≈ 0,004). Attention lr = 3e-4 kullanıyor, Deep Sets lr = 1e-3.

**Sonuçlar.**
1. **Bilgiye erişim hipotezi kısmen doğrulandı.** Global bilgi doğrudan verilince attention politikası fiyata Deep Sets
   düzeyinde tepki vermeye başladı (corr −0,06'dan −0,16 ve −0,13'e). Kâr iyileşti (−2,016'dan −1,961 ve −1,974'e).
   Büyük istasyonda açık 0,83–0,85'ten 0,75–0,79'a indi; bu iyileşme dört seed karşılaştırmasının dördünde de aynı
   yönde. N ≥ 50'de varyantlar fiyat eşiği kuralıyla aynı düzeyde ya da biraz önünde.
2. **Ortalama açık yine de iyileşmedi** (0,909'a karşı 0,954 ve 0,931). Neden: daha iyi fiyat kullanımı daha fazla
   deşarj getirdi (7,8'den 9,0 ve 8,1 kWh'ye). Bu da sıkı trafoda ve küçük N'de daha fazla olurlu olmayan episot
   üretti; özellikle seed 1'de ρ = 4'te %20–28. ENS cezası neredeyse ikiye katlandı. Bu, mekanizma 2'nin (kapasiteyi
   görmeyen alt sınır) bağımsız bir teyidi: politika fiyata daha iyi tepki verdikçe kapasite sıkışmasına daha çok
   giriyor.
3. **Deep Sets cezalı hâlâ önde** (N ≥ 20'de 0,70–0,72'ye karşı 0,75–0,81). Bilgiye erişim farkın bir kısmını açıklıyor,
   tamamını açıklamıyor. Kalan aday optimizasyon: attention KL erken durdurmasına daha sık takılıyor ve daha düşük lr
   ile eğitiliyor. Tuning seçimleri gürültülü olduğundan (§11) bu fark tasarımdan değil seçimden de gelebilir.
4. gctx, gin'e göre biraz daha az olurlu olmayan episot üretiyor (%9,2'ye karşı %11,9). Ancak seed farkları
   (0,85 / 1,06) bu farktan büyük; 2 seed ile ayırt edilemiyor.
5. **Sıradaki adım.** Kalan açık ve yeni varyantlardaki kötüleşme aynı yerde toplanıyor: sıkı trafo, küçük N, olurlu
   olmayan episotlar. Attention tarafındaki bilgi düzeltmesi tek başına farkı kapatmıyor; kapasiteye duyarlı alt sınır
   hem bu varyantların kazancını küçük N'de korumak hem de tüm yöntemler için gerekli görünüyor. İkinci aday, gin/gctx
   için ayrı tuning (özellikle lr ve target_kl).

## 13. Kapasiteye duyarlı ortak alt sınır ve kurala göre rezidüel aksiyon (10 Kasım 2026)

**Amaç.** §12'nin 5. sonucu: kalan açık ve yeni varyantlardaki kötüleşme sıkı trafo, küçük N ve olurlu olmayan
episotlarda toplanıyor; bunun yapısal nedeni ayrılış alt sınırının trafonun ortak kapasitesini görmemesi (mekanizma 2).
Bu bölüm iki mekanizma test ediyor: (i) kapasiteye duyarlı ortak alt sınır, (ii) aksiyonun [l, u] aralığına ve fiyat
eşiği kuralına göre parametrize edilmesi. Hedef, SA-Attn'in fiyat eşiği kuralı (0,822) ve Deep Sets cezalı (0,843)
karşısındaki açığını kapatmak.

**Mekanizma 1: ortak alt sınır** (`code/v2gsafe/safety/joint.py`, `safety.lower_bound: joint`). Her araç için ayrılışta
gereken enerji eşikleri B_i(k) üzerinden geriye doğru su doldurma (water-filling) yapılıyor: gelecekteki tahmini
kapasite Ĉ(t) her adımda araçlar arasında, ayrılışı en yakın olana öncelik verilerek paylaştırılıyor ve bugün en az ne
kadar şarj edilmesi gerektiği (l^J) bulunuyor. Katmanın alt sınırı l = min(u, max(l_dep, l^J)); toplam alt sınır C̃'yi
aşarsa A5'i koruyacak şekilde l_dep'e doğru küçültülüyor. Tahmin hatasına karşı Ĉ'de yük × 1,3, PV × 0,7 (marj 0,3).
l_dep ≤ l ≤ u olduğundan Lemma 1 ve Teorem 1′ aynen geçerli; kapasite bol olduğunda sınır per-araç sınıra indirgeniyor.
§7.1'deki karşı örnek (iki araç, 10 kWh, C = 10 kW) çözülüyor. Testler: `code/tests/test_joint.py` (8 test).

**Mekanizma 2: aksiyon parametrizasyonu** (`safety.action`). `box`: x = l + σ(a / P_ref)(u − l), her ham aksiyon
[l, u] içinde. `resid`: x = l + σ(a / P_ref + logit r)(u − l); r, fiyat eşiği kuralının (baselines `PriceLLF`, aynı
karar) [l, u] içindeki konumu, [r_min, 1 − r_min] aralığına kırpılıyor ve 13. şarj noktası özelliği olarak da
veriliyor. Ortalama başlığı sıfırla başladığından başlangıçtaki deterministik politika kuralın kendisi; PPO düzeltme
öğreniyor. İkisinde de β = 0. Ek seçenek `ppo.resid_l2 = κ`: kayba κ Σ_i μ_i² ekleniyor (rezidüel logit üzerinde
kurala merkezli Gauss önseli). Yeni anahtarların hepsi hash-nötr; mevcut yapılandırmaların hash'leri değişmedi.

**Adil karşılaştırma.** Ortak alt sınır paylaşılan güvenlik katmanının parçası, bu yüzden kural ve önceki modeller de
onunla değerlendirildi (yeniden eğitim yok, `evaluate.py --lower-bound joint`, sonuç klasörleri `*_jlb`). Yeni
varyantlar bu "joint katman" referanslarıyla karşılaştırılmalı; orijinal hedefler (0,822 ve 0,843) per-araç katmanla.

**Çalıştırmalar** (her biri 2 seed × 3M adım, cezalı ödül, attn_safe tuning overlay'i; gjoint hariç):

| Yöntem | Yapılandırma | Değişiklik (`attn_safe_penrew`'e göre) |
|---|---|---|
| `attn_joint_safe_penrew` | `train_attn_joint_safe_penrew.yaml` | ortak alt sınır |
| `attn_jbox_safe_penrew` | `train_attn_jbox_safe_penrew.yaml` | + box aksiyonu (yalnız seed 0; seed 1 öncelik nedeniyle durduruldu) |
| `attn_jres_safe_penrew` | `train_attn_jres_safe_penrew.yaml` | + kural rezidüeli, r_min = 0,02 (%25'te durduruldu, aşağıya bakın) |
| `attn_jres2_safe_penrew` | `train_attn_jres2_safe_penrew.yaml` | + kural rezidüeli, r_min = 0,15 |
| `attn_jres3_safe_penrew` | `train_attn_jres3_safe_penrew.yaml` | jres2 + önsel κ = 0,01 (yalnız seed 0; aşağıya bakın) |
| `attn_gres2_safe_penrew` | `train_attn_gres2_safe_penrew.yaml` | jres2 + gin (global özellikler her token'da) |
| `attn_gjoint_safe_penrew` | `train_attn_gjoint_safe_penrew.yaml` | ortak alt sınır + gin, Deep Sets PPO ayarları (lr 1e-3, mb 512, ent 0), overlay yok |

jres (r_min 0,02) ara kontrolde kuralda takılı kaldı: kural çoğunlukla ya hep ya hiç (r bir kırpma sınırında) ve
0,02 / 0,98'de sigmoid doymuş (eğim 0,02), keşif uygulanan gücü neredeyse değiştirmiyor. r_min 0,15'te eğim 0,13.
jres3 (κ = 0,01) seed 0'da kurala fazla yapıştı (aşağıda), bu yüzden seed 1 eğitilmedi. gres2'nin ilk denemesi 10
Kasım 23:11'de OOM ile durdu: iki eğitim, 10 işlik değerlendirme ve aynı makinedeki başka bir proje belleği aştı.
Sonrasında işler sıraya kondu ve `v2g-memguard` birimi eklendi (MemAvailable < 2 GB olunca değerlendirme süreçlerini
durduruyor). Gece yaşanan elektrik kesintisinde gres2 seed 0 700., seed 1 150. güncellemedeydi. İkisi de
`checkpoint_last.pt`'den devam ettirildi (aşağıda "İşletim notları").

**Sonuçlar** (ana ızgara, ρ 4/7, literatür yapılandırmaları hariç; € şarj noktası ve episot başına; açık = yumuşak
kahine göre; `review/joint_diagnosis.py`, çıktı `code/results/run_reports/joint/joint_diagnosis.txt`):

| Yöntem | Katman | Seed | Açık € | Kâr € | ENS cezası € | Deşarj kWh | Olurlu olmayan episot % |
|---|---|---:|---:|---:|---:|---:|---:|
| SA-Attn cezalı | per-araç | 2 | 0,909 | −2,016 | 0,106 | 7,8 | 7,7 |
| Deep Sets cezalı | per-araç | 2 | 0,843 | −1,911 | 0,145 | 8,7 | 7,6 |
| Fiyat eşiği + katman | per-araç | — | 0,822 | −1,980 | 0,054 | 6,0 | 5,6 |
| SA-Attn cezalı `_jlb` | joint | 2 | 0,811 | −2,016 | 0,007 | 7,5 | 1,5 |
| gin `_jlb` | joint | 2 | 0,764 | −1,963 | 0,014 | 8,7 | 2,9 |
| gctx `_jlb` | joint | 2 | 0,776 | −1,978 | 0,011 | 7,9 | 2,4 |
| Deep Sets cezalı `_jlb` | joint | 2 | **0,707** | −1,912 | 0,008 | 8,4 | 1,7 |
| Fiyat eşiği `_jlb` | joint | — | 0,773 | −1,982 | 0,004 | 5,9 | 1,3 |
| `attn_joint` | joint | 2 | 0,822 | −2,014 | 0,021 | 9,2 | 3,9 |
| `attn_jbox` | joint | 1 | 0,840 | −2,033 | 0,020 | 9,7 | 5,3 |
| `attn_jres2` | joint | 2 | **0,768** | −1,962 | 0,018 | 9,3 | 4,0 |
| `attn_jres3` | joint | 1 | 0,801 | −2,012 | 0,001 | 4,9 | 0,8 |
| `attn_gjoint` | joint | 2 | 0,829 | −2,019 | 0,022 | 9,7 | 4,2 |
| `attn_gres2` | joint | 2 | 0,800 | −1,989 | 0,023 | 9,0 | 4,3 |

(Trafo aşımı cezası tüm yöntemlerde 0. Yeni yöntemlerin adlarındaki `_safe_penrew` son eki kısaltıldı.)

| Eşli fark (a − b; negatif = a iyi), 95 % bootstrap CI | Fark € | CI |
|---|---:|---|
| `attn_jres2` − fiyat eşiği (per-araç, **orijinal hedef**) | **−0,054** | [−0,092, −0,021] |
| `attn_jres2` − Deep Sets cezalı (per-araç, **orijinal hedef**) | **−0,075** | [−0,136, −0,022] |
| `attn_jres2` − fiyat eşiği `_jlb` (aynı katman) | −0,005 | [−0,022, +0,013] |
| `attn_jres2` − Deep Sets cezalı `_jlb` (aynı katman) | +0,061 | [+0,052, +0,071] |
| SA-Attn cezalı `_jlb` − SA-Attn cezalı (yalnız katman, yeniden eğitim yok) | −0,098 | [−0,150, −0,058] |
| `attn_joint` − SA-Attn cezalı `_jlb` (joint katmanla yeniden eğitim) | +0,011 | (aşağıda) |
| `attn_jres2` − `attn_joint` (kural rezidüeli) | −0,054 | [−0,061, −0,048] |
| `attn_jbox` − `attn_joint` (box aksiyonu) | +0,018 | [+0,012, +0,024] |
| `attn_gjoint` − `attn_joint` (gin + Deep Sets PPO ayarları) | +0,006 | [+0,001, +0,012] |
| `attn_gres2` − `attn_jres2` (gin) | +0,032 | [+0,026, +0,038] |
| `attn_jres3` − `attn_jres2` (önsel κ = 0,01; seed 0) | +0,033 | [+0,017, +0,047] |
| `attn_gres2` − fiyat eşiği (per-araç) | −0,022 | [−0,060, +0,011] |
| `attn_jres3` − fiyat eşiği `_jlb` | +0,028 | [+0,021, +0,033] |

(`attn_joint` − `_jlb` satırı ortalamalardan: 0,822 − 0,811. Öğrenilen–öğrenilen karşılaştırmalar seed'e göre de
eşlendi; CI replay kümeleri üzerinden ve seed varyansını içermiyor.)

| Seed başına | N ≤ 10 | N = 20 | N ≥ 50 | Toplam | Olurlu olmayan episot % (ρ = 4) | Deşarj kWh |
|---|---:|---:|---:|---:|---:|---:|
| Fiyat eşiği `_jlb` | 0,763 | 0,770 | 0,785 | 0,773 | 2,5 | 5,9 |
| Deep Sets cezalı `_jlb` s0 / s1 | 0,753 / 0,662 | 0,761 / 0,659 | 0,754 / 0,658 | 0,755 / 0,660 | 3,0 / 3,2 | 8,1 / 8,6 |
| `attn_joint` s0 / s1 | 0,818 / 0,857 | 0,807 / 0,820 | 0,813 / 0,811 | 0,814 / 0,831 | 8,3 / 6,3 | 9,5 / 9,0 |
| `attn_jbox` s0 | 0,838 | 0,847 | 0,839 | 0,840 | 9,9 | 9,7 |
| `attn_jres2` s0 / s1 | **0,748** / 0,815 | **0,700** / 0,814 | **0,702** / 0,819 | **0,720** / 0,816 | 4,8 / 10,0 | 8,3 / 10,4 |
| `attn_jres3` s0 | 0,783 | 0,806 | 0,816 | 0,801 | 1,6 | 4,9 |
| `attn_gjoint` s0 / s1 | 0,833 / 0,819 | 0,839 / 0,813 | 0,839 / 0,825 | 0,837 / 0,820 | 8,7 / 6,7 | 9,7 / 9,7 |
| `attn_gres2` s0 / s1 | 0,856 / 0,772 | 0,855 / 0,745 | 0,839 / 0,733 | 0,849 / 0,751 | 10,3 / 6,0 | 9,5 / 8,5 |

N = 5'te (yalnız): `attn_jres2` 0,811, fiyat eşiği `_jlb` 0,753, Deep Sets `_jlb` 0,721. Per-araç katmanla aynı
değerler 1,176 (SA-Attn cezalı), 0,956 (fiyat eşiği), 1,259 (Deep Sets cezalı).

**Hızlı test ve rezidüel büyüklüğü.** Tam değerlendirmeden önce her checkpoint, 24 yapılandırma (N ≤ 50) × 15 replay
üzerinde joint katmanla J^pen (şarj noktası başına, yüksek = iyi) ile tarandı. Ayrıca rezidüel logitin büyüklüğü |μ|
ölçüldü (μ = 0 kuralın kendisi).

| Checkpoint | J^pen (hızlı test) | Medyan \|μ\| | Tam ızgara açığı |
|---|---:|---:|---:|
| Fiyat eşiği | −2,061 | — | 0,773 |
| Deep Sets cezalı s0 / s1 | −2,005 / −1,909 | — | 0,755 / 0,660 |
| `attn_jres2` s0 / s1 | −1,977 / −2,072 | 1,12 / 3,0 | 0,720 / 0,816 |
| `attn_jres3` s0 | −2,083 | 0,14 | 0,801 |
| `attn_gjoint` s0 | −2,086 | — | 0,837 |
| `attn_gres2` s0 / s1 | −2,105 / −1,997 | 4,27 / 2,73 | 0,849 / 0,751 |

Hızlı test, tam ızgaradaki sıralamayı seed düzeyinde doğru verdi.

**İşletim notları.** (i) Elektrik kesintisi (10–11 Kasım gecesi): gres2 iki seed'de `checkpoint_last.pt`'den devam
ettirildi. `train_log.csv` dosyalarında kesintiden kalan bozuk satırlar (null baytlar) vardı. Bunlar checkpoint
güncellemesine kadar olan geçerli satırlarla yeniden yazıldı; her güncellemenin son kaydı tutuldu, böylece devam
ettirmeden gelen yinelenen satırlar da temizlendi. Seed 0'da 1–700, seed 1'de 1–150 arası eksiksiz. Bozuk dosyalar
`train_log.csv.corrupt_poweroff` olarak WSL'de saklandı, depoya alınmadı. (ii) Devam ettirme hatası: CUDA'da
`map_location` RNG durumlarını GPU'ya taşıyordu ve `torch.set_rng_state` hata veriyordu. `ppo._load` RNG durumlarını
`.cpu()` ile yüklüyor (düzeltildi, devam ettirme doğrulandı). Devam ettirilen eğitim RNG durumunu geri yüklüyor; yine
de kesintisiz bir çalıştırmayla bit düzeyinde aynı olduğu garanti değil. (iii) jres3 değerlendirmesi bellek nedeniyle
bir kez öldürüldü ve 3 işle yeniden çalıştırıldı (tamamlanan parçalar atlandı).

**Sonuçlar.**
1. **Orijinal hedef sağlandı, ama asıl kazanç katmandan geliyor.** `attn_jres2` (ortak alt sınır + fiyat eşiği
   kuralına göre rezidüel aksiyon, r_min 0,15) ana ızgarada 0,768 açık veriyor. Bu, per-araç katmanlı fiyat eşiği
   kuralından 0,054 [−0,092, −0,021], Deep Sets cezalıdan 0,075 [−0,136, −0,022] daha iyi; ikisinde de CI sıfırı
   içermiyor. Ancak bu kazancın büyük kısmı ortak alt sınırın kendisinden geliyor: eski SA-Attn checkpoint'i yeniden
   eğitilmeden joint katmanla değerlendirildiğinde açık 0,909'dan 0,811'e iniyor (−0,098). Joint katmanla yeniden
   eğitim tek başına buna bir şey eklemiyor (`attn_joint` 0,822). Kural rezidüeli bunun üstüne −0,054 ekliyor.
2. **Adil karşılaştırmada attention kural düzeyinde, Deep Sets düzeyinde değil.** Aynı joint katmanla `attn_jres2`
   fiyat eşiği kuralıyla berabere (−0,005 [−0,022, +0,013]) ve Deep Sets cezalının 0,061 [+0,052, +0,071] gerisinde.
   Tüm yöntemler içinde en iyisi yeniden eğitilmemiş Deep Sets cezalı `_jlb` (0,707). Ortak alt sınır paylaşılan
   güvenlik katmanının parçası olduğundan "attention açığı kapandı" ifadesi yalnız per-araç referanslara göre doğru.
   Aynı katmanla kurala yetişmek bir kazanım; Deep Sets'e karşı mimari farkı kapatmıyor.
3. **Joint katmanla yeniden eğitim olurlu olmayan episotları artırıyor.** Yeniden eğitilmeden joint katmanla
   değerlendirilen modellerde olurlu olmayan episot oranı %1,5–2,9. Joint katmanla eğitilen modellerde %3,9–5,3; en
   sıkı durumda (workplace, ρ = 4) %11–14, `_jlb`'de %4–8. Yeniden eğitilen politikalar daha fazla deşarj ediyor
   (9,0–9,7 kWh'ye karşı 7,5–8,7). Yorum: politika alt sınıra kadar deşarj etmeyi öğreniyor ve 0,3 marjı (yük × 1,3,
   PV × 0,7) tahmin hatasını karşılamıyor. Marj, eğitilen politikalar için ayarlanmadı.
4. **Box aksiyonu tek başına işe yaramadı** (+0,018 [+0,012, +0,024]). Kazanç, aksiyonun [l, u]'ya göre
   parametrize edilmesinden değil, kuralın başlangıç noktası (anchor) olarak kullanılmasından geliyor.
5. **Seed varyansı baskın.** `attn_jres2` s0 0,720, s1 0,816; s1 fiyat eşiği kuralıyla (0,822) neredeyse aynı.
   gres2'de sıra ters: s0 0,849, s1 0,751. Kuraldan uzaklaşma (büyük |μ|) kısmen kötü seed'lerle örtüşüyor
   (jres2 s1 3,0, gres2 s0 4,27). Örtüşme tam değil: gres2 s1 |μ| 2,73 ile iyi. 2 seed ile `attn_jres2`'nin kurala
   karşı kazancı sağlam değil. Bootstrap CI replay'ler üzerinden ve seed varyansını içermiyor.
6. **Kurala çeken önsel (jres3) fazla güçlü.** κ = 0,01'de medyan |μ| 0,14: politika pratikte kırpılmış kuraldı.
   r_min 0,15 kırpması, çoğunlukla ya hep ya hiç davranan kuralı yumuşatıyor; bu da açığı kötüleştiriyor (fiyat
   eşiği `_jlb`'ye göre +0,028 [+0,021, +0,033]). Olumlu yan etki: en düşük olurlu olmayan episot oranı (%0,8) ve en
   az deşarj (4,9 kWh). κ ayarlanmadı; arada bir değer (ör. 0,001) denenmedi.
7. **Global girdi rezidüele katkı vermedi.** gres2, jres2'den +0,032 [+0,026, +0,038] kötü. Bu, joint katmanda
   gjoint'in joint'ten farksız olmasıyla (+0,006) tutarlı. §12'deki bilgiye erişim kazancı (gin `_jlb` 0,764)
   yeniden eğitimde korunmadı.

**Uyarılar.** jbox ve jres3 tek seed. Varyantlar sırayla, önceki sonuçlara bakılarak seçildi; bu da en iyi varyantın
(jres2) iyimser seçilmesi riskini taşıyor. r_min, κ ve joint marjı ayarlanmadı. Sonuçlar makaleye ve `analysis/`
tablolarına dahil değil.
