import time
import urllib.parse
import urllib3
import inspect

import requests
from bs4 import BeautifulSoup
from spellchecker import SpellChecker

def replace_umlauts(text):
    replacements = {
        "ä": "ae", "ö": "oe", "ü": "ue",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
        "ß": "ss"
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    return text


class OnlineSolver:

    def __init__(self, lang='de'):
        self.sources = [
            self._get_from_woxikon,
            # self._get_from_kreuzwort_net,
            self._get_from_wort_suchen,
        ]
        self.lang = lang

        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8"
        }

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _get_from_wort_suchen(self, hint_text:str, word_length=None) -> list[str]:

        base_url = "https://www.wort-suchen.de/kreuzwortraetsel-hilfe/loesungen/"
        query = urllib.parse.quote(hint_text.strip()) + "/" + "_" * word_length if word_length else ""
        url = base_url + query + "/"

        candidates = []

        try:
            response = requests.get(url, timeout=5, verify=False, headers=self._headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            result_rows = soup.select("tr.cw-result")

            for row in result_rows:
                size = int(row.select("td.cw-option span.cw-letter")[0].get_text(strip=True))

                # filtering with form_data seems not to work properly
                if size == word_length:
                    anw = row.select("a.cw-answer-link")[0].get_text(strip=True).encode('latin-1', 'ignore').decode(
                        'latin-1').upper().strip()
                    candidates.append(anw)

        except requests.exceptions.HTTPError:
            print(f"[{inspect.currentframe().f_code.co_name}] HTTP error ({response.status_code}): {url}")
        except requests.exceptions.Timeout:
            print(f"[{inspect.currentframe().f_code.co_name}] timeout for request: {url}")
        except requests.exceptions.RequestException as e:
            print(f"[{inspect.currentframe().f_code.co_name}] connection error: {e}")
        except Exception as e:
            print(f"[{inspect.currentframe().f_code.co_name}] unexpected error: {e}")
        finally:
            return candidates

    def _get_from_kreuzwort_net(self, hint_text:str, word_length=None) -> list[str]:

        url = "https://www.kreuzwort.net/api/question/search"

        payload = {
            "query": hint_text.strip(),
            "pattern": "_" * word_length if word_length else ""
        }

        self._headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

        candidates = []

        try:
            response = requests.post(url, json=payload, headers=self._headers, timeout=5, verify=False)
            response.raise_for_status()
            data = response.json()

            for entry in data.get("results", []):
                solution = entry.get("solution", "").strip().lower()
                if solution.isalpha():
                    candidates.append(solution)

        except requests.exceptions.HTTPError:
            print(f"[{inspect.currentframe().f_code.co_name}] HTTP error ({response.status_code}): {url}")
        except requests.exceptions.Timeout:
            print(f"[{inspect.currentframe().f_code.co_name}] timeout for request: {url}")
        except requests.exceptions.RequestException as e:
            print(f"[{inspect.currentframe().f_code.co_name}] connection error: {e}")
        except Exception as e:
            print(f"[{inspect.currentframe().f_code.co_name}] unexpected error: {e}")
        finally:
            return candidates

    def _get_from_woxikon(self, hint_text:str, word_length=None) -> list[str]:

        letters = []
        for cnt in range(word_length):
            letters.append("")

        query = replace_umlauts(hint_text.replace(",", ""))
        query = query.strip().upper().replace(" ", "-")

        url = f"https://www.woxikon.de/kreuzwortraetsel-hilfe/frage/{query}"

        form_data = {
            "answer_letters[]": letters
        }

        candidates = []

        try:
            response = requests.get(url, timeout=5, data=form_data, verify=False, headers=self._headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            result_rows = soup.select("div.result-table-content div.row.result")

            for row in result_rows:

                size = int(row.select("div.clue-answer-size")[0].get_text(strip=True))

                # filtering with form_data seems not to work properly
                if size == word_length:
                    anw = row.select("div.clue-answer")[0].get_text(strip=True).encode('latin-1', 'ignore').decode(
                        'latin-1').upper().strip()
                    candidates.append(anw)

        except requests.exceptions.HTTPError:
            print(f"[{inspect.currentframe().f_code.co_name}] HTTP error ({response.status_code}): {url}")
        except requests.exceptions.Timeout:
            print(f"[{inspect.currentframe().f_code.co_name}] timeout for request: {url}")
        except requests.exceptions.RequestException as e:
            print(f"[{inspect.currentframe().f_code.co_name}] connection error: {e}")
        except Exception as e:
            print(f"[{inspect.currentframe().f_code.co_name}] unexpected error: {e}")
        finally:
            return candidates

    def lookup_answers_online(self, hint_text: str, word_length=None, debug=False):
        all_results = []

        if debug:
         print(f"searching for: '{hint_text}' (length: {word_length})")

        for source_fn in self.sources:
            try:
                if debug:
                    print(f'using {source_fn.__name__}: ', end='')
                words = source_fn(hint_text, word_length)
                if debug:
                    print(f'{words}')
                all_results.extend(words)
                time.sleep(1)
            except Exception as e:
                print(f"error at{source_fn.__name__}: {e}")

        # Duplikate entfernen, sortieren
        return sorted(set(all_results))
