# Projekt: warstwy probabilistyczne w sieciach neuronowych (Keras)

## Cel

Porównanie klasyfikatora **z warstwami probabilistycznymi** i **bez nich**. Chcemy znaleźć, w jakich warunkach różnice między modelami są **największe**, a w jakich **najmniejsze**.

---

## 1. Dane: Rotated MNIST

- Trening na cyfrach bez rotacji lub z małą rotacją.
- Test na cyfrach obróconych o coraz większy kąt (np. 0°, 15°, 30°, …, 180°).

---

## 2. Modele

> **Uwaga:** poniższy dobór warstw to **propozycja na start**. Może się zmienić w trakcie eksperymentów albo po podmianie zbioru danych.

| | Model A: deterministyczny | Model B: probabilistyczny |
|---|---|---|
| Warstwy | `Dense` / `Conv2D` | `tfp.layers.DenseFlipout` / `tfp.layers.Convolution2DFlipout` |
| Wyjście | softmax | rozkład predykcyjny uśredniony z T próbek (np. T = 50) |
| Architektura | taka sama w obu modelach (ta sama liczba warstw i neuronów) | |

**Uwaga techniczna:** TensorFlow Probability współpracuje z **tf-keras** (Keras 2), a nie z Keras 3. Trzeba zainstalować `tf-keras` i używać `import tf_keras as keras`.

---

## 3. Eksperymenty

Każdą kombinację powtarzamy kilka razy z różnymi seedami.

1. **Rozmiar zbioru treningowego**, np. 500 / 1 000 / 5 000 / 10 000 / 60 000 przykładów.
2. **Poziom zaszumienia**
   - szum etykiet: 0%, 10%, 20%, 30% losowo zamienionych etykiet,
   - szum na obrazach (gaussowski): różne wartości σ.
3. **Kąt rotacji danych testowych** (przesunięcie rozkładu).

---

## 4. Metryki

Oznaczenia: $N$ to liczba przykładów testowych, $K$ to liczba klas (dla MNIST $K = 10$), $p_{i,k}$ to przewidziane prawdopodobieństwo klasy $k$ dla przykładu $i$, a $y_i$ to prawdziwa klasa.

Dla modelu probabilistycznego $p_{i,k}$ to **średnia z T przebiegów**:

$$p_{i,k} = \frac{1}{T}\sum_{t=1}^{T} p^{(t)}_{i,k}$$

### 4.1 Accuracy

$$\text{Acc} = \frac{1}{N}\sum_i \mathbb{1}\left[\arg\max_k p_{i,k} = y_i\right]$$

- **Co mierzy:** jaki odsetek predykcji jest trafny.
- **Zakres:** 0–1, im więcej, tym lepiej.
- **Rola w projekcie:** punkt odniesienia. Spodziewamy się, że oba modele będą tu podobne. Accuracy **nie mówi nic o tym, jak pewny był model**.

### 4.2 NLL (Negative Log-Likelihood, log-loss)

$$\text{NLL} = -\frac{1}{N}\sum_i \log p_{i,y_i}$$

- **Co mierzy:** jakie prawdopodobieństwo model przypisał prawdziwej klasie.
- **Zakres:** 0–∞, im mniej, tym lepiej.
- **Rola w projekcie:** surowo karze **pewne pomyłki**. Jeśli model da 0.99 złej klasie, NLL mocno rośnie. Tu model probabilistyczny powinien wyraźnie wygrywać przy dużej rotacji i szumie.
- **Kod:** `keras.losses.sparse_categorical_crossentropy`

### 4.3 Brier score

$$\text{Brier} = \frac{1}{N}\sum_i \sum_k \left(p_{i,k} - \mathbb{1}[k = y_i]\right)^2$$

- **Co mierzy:** błąd kwadratowy między wektorem prawdopodobieństw a one-hot prawdziwej etykiety.
- **Zakres:** 0–2, im mniej, tym lepiej.
- **Rola w projekcie:** podobna do NLL, ale łagodniejsza dla skrajnych błędów i odporna na $\log 0$. Dobrze ją pokazać obok NLL.

### 4.4 ECE (Expected Calibration Error)

Predykcje dzielimy na $M$ koszyków według pewności $\hat{p}_i = \max_k p_{i,k}$ (np. $M = 15$).

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\,\left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

- **Co mierzy:** czy pewność modelu zgadza się z rzeczywistą trafnością. Jeśli model mówi „90%”, powinien mieć rację w ~90% takich przypadków.
- **Zakres:** 0–1, im mniej, tym lepiej.
- **Rola w projekcie:** **kluczowa metryka**. Zwykłe sieci są zwykle zbyt pewne siebie (*overconfident*), a warstwy bayesowskie powinny to poprawiać.
- **Wizualizacja:** *reliability diagram*, czyli słupki accuracy vs. confidence dla każdego koszyka na tle przekątnej oznaczającej idealną kalibrację.

### 4.5 Entropia predykcji (miara niepewności)

$$H_i = -\sum_k p_{i,k}\log p_{i,k}$$

- **Co mierzy:** jak rozmyta jest predykcja. Wynosi 0, gdy model jest całkowicie pewny, i $\log K \approx 2.30$, gdy wszystkie klasy są jednakowo prawdopodobne.
- **Rola w projekcie:** to nie jest metryka „lepiej/gorzej”, tylko **zachowanie modelu**. Chcemy, żeby entropia **rosła razem z kątem rotacji i szumem**. Pokazujemy średnią entropię osobno dla poprawnych i błędnych predykcji.

**Dodatkowo tylko dla modelu probabilistycznego:** rozkład entropii na dwie części.

$$\underbrace{H\left[\bar{p}_i\right]}_{\text{całkowita}} = \underbrace{\frac{1}{T}\sum_t H\left[p^{(t)}_i\right]}_{\text{aleatoryczna (dane)}} + \underbrace{\text{MI}_i}_{\text{epistemiczna (model)}}$$

Mutual information (MI) mierzy, jak bardzo kolejne próbki modelu się ze sobą nie zgadzają. Powinna rosnąć przy dużej rotacji i małym zbiorze treningowym. Zwykły model **nie ma tej informacji**.

### 4.6 Accuracy vs. odrzucenie (selective prediction)

- **Jak liczymy:** sortujemy przykłady według niepewności (entropii) i odrzucamy kolejno 0%, 10%, …, 50% najbardziej niepewnych. Na pozostałych liczymy accuracy.
- **Rola w projekcie:** praktyczny sens niepewności, czyli „model oddaje trudne przypadki człowiekowi”. Lepszy model ma krzywą, która rośnie szybciej.
- **Podsumowanie liczbowe:** pole pod krzywą ryzyko–pokrycie (AURC).

### 4.7 Koszt obliczeniowy

- Czas treningu na epokę.
- Czas predykcji, bo model probabilistyczny wymaga T przebiegów.
- Liczba parametrów. Warstwy Flipout mają ~2× więcej parametrów (średnia i wariancja dla każdej wagi).

### Podsumowanie metryk

| Metryka | Lepiej | Spodziewana różnica A vs B |
|---|---|---|
| Accuracy | ↑ | mała |
| NLL | ↓ | duża przy rotacji i szumie |
| Brier | ↓ | średnia |
| ECE | ↓ | **duża** |
| Entropia | rośnie z trudnością danych | B reaguje wyraźniej |
| MI (epistemiczna) | rośnie z trudnością danych | tylko model B |
| AURC | ↓ | średnia–duża |
| Czas / parametry | ↓ | B droższy (~T× przy predykcji) |

**Zestaw minimum do wykresów:** Accuracy, NLL, ECE i entropia.

---

## 5. Prezentacja wyników

- Accuracy, NLL, ECE i entropia **w funkcji kąta rotacji**, dla obu modeli na jednym wykresie.
- Metryki **w funkcji rozmiaru zbioru treningowego**.
- Metryki **w funkcji poziomu szumu**.
- **Heatmapa różnicy (B − A)** dla kombinacji rozmiar zbioru × poziom szumu, która pokaże, gdzie przewaga modelu B jest największa, a gdzie najmniejsza.
- **Reliability diagrams** dla obu modeli.
- Krzywe **accuracy vs. odrzucenie**.
- Przykładowe cyfry, przy których model A jest pewny i się myli, a model B zgłasza niepewność.

---

## 6. Wnioski (pytania, na które odpowiadamy)

- W jakich warunkach warstwy probabilistyczne dają największą przewagę? Spodziewamy się: mały zbiór, dużo szumu, duża rotacja.
- Kiedy różnica jest pomijalna? Spodziewamy się: duży i czysty zbiór, dane z tego samego rozkładu co trening.
- Ile kosztuje model probabilistyczny: czas treningu, czas predykcji, liczba parametrów.

---

## 7. Propozycje rozwijania projektu

- **Wykrywanie danych spoza rozkładu (OOD), metryka AUROC:**
  - łączymy dane testowe in-distribution (etykieta 0) z danymi OOD, np. Fashion-MNIST albo cyframi obróconymi o ≥ 90° (etykieta 1),
  - score to entropia lub MI, a potem liczymy AUROC (`sklearn.metrics.roc_auc_score`),
  - zakres: 0.5 to zgadywanie, 1.0 to idealne rozróżnienie,
  - sprawdza, czy model „wie, że nie wie”. Tu spodziewamy się dużej różnicy między modelami.
- **Podmiana zbioru danych**, np. na Fashion-MNIST albo CIFAR-10 / CIFAR-10-C, wraz z dostosowaniem architektury (więcej warstw konwolucyjnych, probabilistyczne warstwy konwolucyjne).
- **MC Dropout** jako trzeci, tani model porównawczy.
- **Probabilistyczna warstwa wyjściowa** (`tfp.layers.OneHotCategorical` / `DistributionLambda`), żeby jawnie modelować niepewność aleatoryczną.
