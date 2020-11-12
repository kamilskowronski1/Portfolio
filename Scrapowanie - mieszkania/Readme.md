# Wyszukiwanie danych dotyczących mieszkań na sprzedaż w Gdańsku

Napisany przeze mnie skrypt ma na celu zbieranie i zapisywanie w pliku csv danych dotyczących mieszkań na sprzedaż. Zapisywany jest tytuł ogłoszenia, lokalizacja, powierzchnia, liczba pokoi, rok budowy oraz cena sprzedaży. 

<b>Działania skryptu</b>

1. Pobranie zawartości wskazanej strony w html przy pomocy biblioteki BeatifulSoup.
2. Sprawdzenie ilości stron dla zadanych kryteriów wyszukiwania.
3. Odnosząc się do konkretnych elementów danego ogłoszenia zapiswanie potrzebnych danych w tabeli przy użyciu biblioteki Pandas
4. Wykonanie działania dla każdego ogłoszenia na stronie i powtórzenie tego zgodnie z wcześniej zadeklarowaną ilością stron.
5. Zapisanie zbiorczej tabeli do pliku csv.
