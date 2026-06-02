# Odległość supremum funkcji ciągłych

Aplikacja Streamlit służy do numerycznego wyznaczania odległości supremum
dwóch funkcji ciągłych `f, g : [a,b] -> R`:

```text
d∞(f,g) = max_{x in [a,b]} |f(x)-g(x)|
```

Interfejs pozwala wpisać przedział, wzory funkcji oraz dokładność `epsilon`.
Wynik jest prezentowany jako liczba, wykres funkcji `f(x)` i `g(x)`, wykres
`|f(x)-g(x)|` z poziomą linią maksimum oraz tabela punktów, w których maksimum
jest osiągane z dokładnością `epsilon`.

## Uruchomienie

Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

Uruchom aplikację:

```bash
streamlit run app.py
```

## Publikacja na GitHubie i Streamlit Cloud

Repozytorium powinno zawierać co najmniej trzy pliki:

```text
app.py
requirements.txt
README.md
```

Przykładowe komendy do wrzucenia projektu na GitHuba:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Add Streamlit supremum distance app"
git branch -M main
git remote add origin https://github.com/GrzegorzTarasek/odleglosc-supremum-streamlit.git
git push -u origin main
```

Jeżeli wrzucasz pliki ręcznie przez stronę GitHuba, utwórz repozytorium:

```text
https://github.com/GrzegorzTarasek/odleglosc-supremum-streamlit
```

i dodaj do niego pliki:

```text
app.py
requirements.txt
README.md
.gitignore
```

Po wypchnięciu repozytorium na GitHuba aplikację można uruchomić online w
Streamlit Community Cloud:

1. Wejdź na https://share.streamlit.io.
2. Zaloguj się kontem GitHub.
3. Kliknij `Create app`.
4. Wybierz repozytorium `GrzegorzTarasek/odleglosc-supremum-streamlit`.
5. Ustaw branch jako `main`.
6. Jako main file path wpisz `app.py`.
7. Kliknij deploy.

Streamlit automatycznie odczyta plik `requirements.txt` i zainstaluje
wymagane biblioteki.

## Użyte biblioteki

- Streamlit: interfejs aplikacji.
- NumPy: obliczenia na gęstej siatce punktów.
- SymPy: parsowanie wzorów funkcji i zamiana ich na funkcje numeryczne.
- SciPy: lokalna optymalizacja kandydatów na maksimum.
- Plotly: interaktywne wykresy.
- Pandas: tabela punktów maksimum.

## Jak wpisywać funkcje

Funkcje wpisuje się jako wzory zależne od zmiennej `x`.

Przykłady:

```text
sin(x)
x**2 / 4
exp(-x**2)
sqrt(abs(x))
log(x)
cos(pi*x)
```

Obsługiwane funkcje:

```text
sin, cos, tan, atan, arctan, exp, log, sqrt, abs
```

Obsługiwane stałe:

```text
pi, E
```

Potęgowanie należy wpisywać jako `x**2`, `x**3` itd.

## Metoda numeryczna

Program oblicza funkcję

```text
h(x) = |f(x)-g(x)|
```

Najpierw generowana jest gęsta siatka punktów na przedziale `[a,b]`.
Następnie aplikacja oblicza wartości `h(x)` na tej siatce, pomijając wartości
`NaN` oraz nieskończoności. Kandydatami na maksimum są lokalne maksima na
siatce, kilka punktów o największych wartościach oraz końce przedziału.
Dla tych kandydatów wynik jest poprawiany lokalnie funkcją
`scipy.optimize.minimize_scalar`, przez minimalizację `-h(x)`.

Jako `d∞(f,g)` przyjmowana jest największa znaleziona wartość `h(x)`.
Punkty maksimum to punkty spełniające warunek:

```text
h(x) >= d∞(f,g) - epsilon
```

Zduplikowane punkty są usuwane przez zaokrąglenie wartości `x`.

## Opis teoretyczny

Ponieważ `f` i `g` są ciągłe na przedziale domkniętym i ograniczonym `[a,b]`,
funkcja `|f-g|` również jest ciągła. Z twierdzenia Weierstrassa wynika, że
funkcja ciągła na przedziale domkniętym i ograniczonym osiąga maksimum.
Dlatego odległość `d∞(f,g)` jest dobrze określona.
