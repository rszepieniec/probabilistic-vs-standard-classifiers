# Projekt: warstwy probabilistyczne w sieciach neuronowych (Keras)

## Cel

Porównanie klasyfikatora **z warstwami probabilistycznymi** i **bez nich** na rzeczywistych danych medycznych z szumem pomiarowym. Chcemy znaleźć, w jakich warunkach różnice między modelami są **największe**, a w jakich **najmniejsze**.

---

## 1. Dane: PTB-XL (EKG)

- **Baza:** [PTB-XL na PhysioNet (v1.0.1)](https://physionet.org/content/ptb-xl/1.0.1/)
- **Kopia danych (Google Drive):** [pobierz](https://drive.google.com/file/d/1svOX5eObCFqY5okBq0o0mX32Rgn6MV9x/view?usp=drive_link)
- **Publikacja:** Wagner i in., *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data 2020.

### Gdzie umieścić dane

Pobierz archiwum (z Drive'a albo PhysioNet) i rozpakuj je do katalogu `data/` w głównym folderze repo. Struktura musi wyglądać dokładnie tak:

```
probabilistic-vs-standard-classifiers/
├── README.md
└── data/
    └── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
        ├── ptbxl_database.csv
        ├── scp_statements.csv
        ├── records100/
        └── records500/
```

- Nazwa folderu z danymi musi zostać bez zmian: `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`.
- Pliki `ptbxl_database.csv` i `scp_statements.csv` muszą leżeć bezpośrednio w tym folderze, a nie w dodatkowym podfolderze po rozpakowaniu.
- Dane zajmują ok. 3,2 GB i **nie są wrzucane do repo** (katalog `data/` jest w `.gitignore`).

Szybkie sprawdzenie (z głównego folderu repo):

```bash
ls data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv
```

### Co zawiera

- **21 837 zapisów EKG** od 18 885 pacjentów, każdy 10 s, **12 odprowadzeń** (I, II, III, aVR, aVL, aVF, V1–V6).
- Dwie wersje: 500 Hz (`records500/`) i **100 Hz** (`records100/`). Używamy 100 Hz, więc jeden przykład ma kształt `(1000, 12)`.
- Format WFDB (biblioteka `wfdb`), metadane w `ptbxl_database.csv`, opis kodów diagnoz w `scp_statements.csv`.

### Etykiety: 5 nadklas diagnostycznych

| Klasa | Znaczenie | Zapisów (wszystkie) | Zapisów z tylko tą klasą |
|---|---|---|---|
| **NORM** | prawidłowe EKG | 9 528 | 9 083 |
| **MI** | zawał mięśnia sercowego | 5 486 | 2 538 |
| **STTC** | zmiany odcinka ST i załamka T | 5 250 | 2 406 |
| **CD** | zaburzenia przewodzenia | 4 907 | 1 709 |
| **HYP** | przerost mięśnia sercowego | 2 655 | 536 |

Zadanie jest z natury wieloetykietowe: 5 158 zapisów ma kilka nadklas naraz, a 407 nie ma żadnej.

> **Propozycja na start:** klasyfikacja jednoetykietowa (softmax, $K = 5$) na 16 272 zapisach, które mają dokładnie jedną nadklasę. Wersja wieloetykietowa jest w propozycjach rozwijania projektu.

### Źródła niepewności zapisane w bazie

| Kolumna | Co oznacza | Zapisów |
|---|---|---|
| `scp_codes` | przy każdej diagnozie **pewność 0–100**, wyliczona przez autorów bazy ze sformułowania opisu („możliwy”, „prawdopodobny”…). To nie jest liczba wpisana przez kardiologa. Typowe wartości: 15, 35, 50, 80, 100. Wartość 0 oznacza „brak informacji” | ~7,6 tys. diagnoz (nie zapisów) z pewnością od 1 do 99 |
| `static_noise`, `burst_noise`, `baseline_drift`, `electrodes_problems` | problemy z jakością sygnału (które odprowadzenia są zaszumione) | 5 021 z co najmniej jednym problemem |
| `validated_by_human` | `False` oznacza, że opis mógł wygenerować automatycznie aparat EKG i nie potwierdził go lekarz (możliwy szum etykiet) | 5 743 z `False` |

### Podział danych

Kolumna `strat_fold` (1–10) dzieli dane **po pacjentach**: foldy 1–8 to trening, 9 to walidacja, 10 to test. Ten sam pacjent nigdy nie trafia do dwóch części.

---

## 2. Modele

> **Uwaga:** poniższy dobór warstw to **propozycja na start**. Może się zmienić w trakcie eksperymentów.

| | Model A: deterministyczny | Model B: probabilistyczny |
|---|---|---|
| Warstwy | `Conv1D` / `Dense` | `tfp.layers.Convolution1DFlipout` / `tfp.layers.DenseFlipout` |
| Wejście | `(1000, 12)` | `(1000, 12)` |
| Wyjście | softmax | rozkład predykcyjny uśredniony z T próbek (np. T = 50) |
| Architektura | taka sama w obu modelach (ta sama liczba warstw i filtrów) | |

**Uwaga techniczna:** TensorFlow Probability współpracuje z **tf-keras** (Keras 2), a nie z Keras 3. Trzeba zainstalować `tf-keras` i używać `import tf_keras as keras`.

---

## 3. Eksperymenty

Każdą kombinację powtarzamy kilka razy z różnymi seedami.

1. **Rozmiar zbioru treningowego**, np. 500 / 1 000 / 2 000 / 5 000 / pełny (foldy 1–8).
2. **Sztuczny szum** dodany do danych treningowych lub testowych:
   - szum etykiet: 0%, 10%, 20%, 30% losowo zamienionych etykiet,
   - szum na sygnale (gaussowski): różne wartości σ.
3. **Naturalna niepewność z bazy:** porównanie modeli na podgrupach zbioru testowego:
   - sygnał czysty vs. zaszumiony (kolumny jakości sygnału),
   - pewność diagnozy 100 vs. < 100,
   - opis potwierdzony przez lekarza vs. automatyczny (`validated_by_human`).

---

## 4. Metryki

Oznaczenia: $N$ to liczba przykładów testowych, $K = 5$ to liczba klas, $p_{i,k}$ to przewidziane prawdopodobieństwo klasy $k$ dla przykładu $i$, a $y_i$ to prawdziwa klasa. Dla modelu probabilistycznego $p_{i,k}$ to średnia z T przebiegów:

$$p_{i,k} = \frac{1}{T}\sum_{t=1}^{T} p^{(t)}_{i,k}$$

### 4.1 Accuracy

$$\text{Acc} = \frac{1}{N}\sum_i \mathbb{1}\left[\arg\max_k p_{i,k} = y_i\right]$$

Odsetek trafnych predykcji (0–1, ↑). Punkt odniesienia: spodziewamy się, że oba modele będą tu podobne. Accuracy nie mówi nic o tym, jak pewny był model. Przy niezbalansowanych klasach warto dodać **balanced accuracy** lub **macro-F1**.

### 4.2 NLL (log-loss)

$$\text{NLL} = -\frac{1}{N}\sum_i \log p_{i,y_i}$$

Prawdopodobieństwo przypisane prawdziwej klasie (0–∞, ↓). Surowo karze **pewne pomyłki**, więc tu model probabilistyczny powinien wygrywać przy szumie i małym zbiorze.

### 4.3 Brier score

$$\text{Brier} = \frac{1}{N}\sum_i \sum_k \left(p_{i,k} - \mathbb{1}[k = y_i]\right)^2$$

Błąd kwadratowy między prawdopodobieństwami a one-hot etykiety (0–2, ↓). Podobny do NLL, ale łagodniejszy dla skrajnych błędów.

### 4.4 ECE (Expected Calibration Error)

Predykcje dzielimy na $M$ koszyków według pewności $\hat{p}_i = \max_k p_{i,k}$ (np. $M = 15$):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

Czy pewność modelu zgadza się z trafnością (0–1, ↓). **Kluczowa metryka**: zwykłe sieci są zwykle zbyt pewne siebie, a warstwy bayesowskie powinny to poprawiać. Wizualizacja: *reliability diagram*.

### 4.5 Entropia predykcji

$$H_i = -\sum_k p_{i,k}\log p_{i,k}$$

Jak rozmyta jest predykcja: od 0 (pełna pewność) do $\log 5 \approx 1.61$ (wszystkie klasy jednakowo prawdopodobne). To miara **zachowania modelu**: chcemy, żeby rosła z szumem i była wyższa tam, gdzie baza wskazuje niższą pewność diagnozy.

Dla modelu probabilistycznego dzielimy ją na dwie części:

$$\underbrace{H\left[\bar{p}_i\right]}_{\text{całkowita}} = \underbrace{\frac{1}{T}\sum_t H\left[p^{(t)}_i\right]}_{\text{aleatoryczna (dane)}} + \underbrace{\text{MI}_i}_{\text{epistemiczna (model)}}$$

Mutual information (MI) mówi, jak bardzo kolejne próbki modelu się ze sobą nie zgadzają. Zwykły model nie ma tej informacji.

### 4.6 Accuracy vs. odrzucenie

Odrzucamy 0%, 10%, …, 50% najbardziej niepewnych przykładów i liczymy accuracy na pozostałych. To scenariusz „model odsyła trudne zapisy do kardiologa”. Podsumowanie liczbowe: AURC (↓).

### 4.7 Koszt obliczeniowy

Czas treningu, czas predykcji (model B wymaga T przebiegów) i liczba parametrów (warstwy Flipout mają ~2× więcej).

### Podsumowanie metryk

| Metryka | Lepiej | Spodziewana różnica A vs B |
|---|---|---|
| Accuracy / macro-F1 | ↑ | mała |
| NLL | ↓ | duża przy szumie i małym zbiorze |
| Brier | ↓ | średnia |
| ECE | ↓ | **duża** |
| Entropia | rośnie z trudnością danych | B reaguje wyraźniej |
| MI (epistemiczna) | rośnie z trudnością danych | tylko model B |
| AURC | ↓ | średnia–duża |
| Czas / parametry | ↓ | B droższy (~T× przy predykcji) |

**Zestaw minimum do wykresów:** Accuracy, NLL, ECE i entropia.

---

## 5. Prezentacja wyników

- Metryki **w funkcji rozmiaru zbioru treningowego** dla obu modeli na jednym wykresie.
- Metryki **w funkcji poziomu szumu** (etykiet i sygnału).
- **Heatmapa różnicy (B − A)** dla kombinacji rozmiar zbioru × poziom szumu, która pokaże, gdzie przewaga modelu B jest największa, a gdzie najmniejsza.
- **Entropia na podgrupach z bazy:** czysty vs. zaszumiony sygnał, pewność diagnozy 100 vs. < 100, opis lekarza vs. automatyczny.
- **Reliability diagrams** dla obu modeli.
- Krzywe **accuracy vs. odrzucenie**.
- Przykładowe zapisy EKG, przy których model A jest pewny i się myli, a model B zgłasza niepewność.

---

## 6. Wnioski (pytania, na które odpowiadamy)

- W jakich warunkach warstwy probabilistyczne dają największą przewagę? Spodziewamy się: mały zbiór, dużo szumu, niepewne diagnozy.
- Kiedy różnica jest pomijalna? Spodziewamy się: duży zbiór, czysty sygnał, jednoznaczne diagnozy.
- Czy niepewność modelu B zgadza się z niepewnością zapisaną w bazie (jakość sygnału, pewność diagnozy)?
- Ile kosztuje model probabilistyczny: czas treningu, czas predykcji, liczba parametrów.

---

## 7. Propozycje rozwijania projektu

- **Klasyfikacja wieloetykietowa:** wszystkie zapisy, 5 nadklas z wyjściem sigmoid zamiast softmax (metryki liczone osobno dla każdej klasy).
- **Wykrywanie danych spoza rozkładu (OOD), metryka AUROC:** np. zapisy z rozrusznikiem (`pacemaker`) albo EKG z innej bazy. Score to entropia lub MI. Sprawdza, czy model „wie, że nie wie”.
- **Wersja 500 Hz** lub bardziej szczegółowe etykiety (podklasy diagnostyczne).
- **MC Dropout** jako trzeci, tani model porównawczy.
- **Probabilistyczna warstwa wyjściowa** (`tfp.layers.OneHotCategorical` / `DistributionLambda`), żeby jawnie modelować niepewność aleatoryczną.
