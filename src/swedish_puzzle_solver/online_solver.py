import time
import urllib.parse
from typing import Final
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List
import urllib3
import json
import io
import requests
from bs4 import BeautifulSoup
from spellchecker import SpellChecker

import logging
from logging import Logger

from swedish_puzzle_solver.util import run_functions_in_parallel

LOG:Final[Logger] = logging.getLogger(__name__)

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
        self.source_functions: List[Callable[[any, any], List[str]]] = [
            self._get_from_kreuzwort_net,
            self._get_from_wort_suchen,
            self._get_from_woxikon,
            self._get_from_buchstaben_com,
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

    def _get_from_wort_suchen(self, hint_text:str, word_length:int) -> list[str]:

        base_url = "https://www.wort-suchen.de/kreuzwortraetsel-hilfe/loesungen/"
        query = urllib.parse.quote(hint_text.strip()) + "/" + "_" * word_length
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
        except Exception as ex:
            LOG.error(ex, exc_info=True)
        finally:
            return candidates

    def _get_from_kreuzwort_net(self, hint_text:str, word_length:int) -> list[str]:

        base_url = "https://www.kreuzwort.net/"

        query = replace_umlauts(hint_text)
        query = query.strip().lower().replace(" ", "-")
        query += ".html?pattern="
        query += "_" * word_length

        url = base_url + "fragen/" + query

        self._headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*"
        })

        candidates = []

        try:
            response = requests.get(url, headers=self._headers, timeout=5, verify=False)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            answers = soup.find(id='js-question-answers')

            if not answers:
                return []

            data_json = json.loads(answers.attrs['data-answers'])

            if str(word_length) in data_json:
                answers = data_json[str(word_length)]

                for answer in answers.keys():
                    anw = answer.encode('latin-1', 'ignore').decode(
                        'latin-1').upper().strip()
                    candidates.append(anw)


        except Exception as ex:
            LOG.error(ex, exc_info=True)
        finally:
            return candidates

    def _get_from_buchstaben_com(self, hint_text:str, word_length:int) -> list[str]:

        base_url = "https://www.buchstaben.com/"

        query = replace_umlauts(hint_text)
        query = query.strip().lower().replace(" ", "-")
        query += "?pattern="
        query += "_" * word_length

        url = base_url + "raetsel/" + query

        candidates = []

        try:
            response = requests.get(url, headers=self._headers, timeout=5, verify=False)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            div = soup.select("div#app")

            data_json = json.loads(div[0].attrs['data-page'])

            if data_json["props"]["errors"]:
                return []

            answers = data_json["props"]["answers"][str(word_length)]
            candidates.extend([answer.encode('latin-1', 'ignore').decode('latin-1').upper().strip() for answer in answers.keys()])

        except Exception as ex:
            LOG.error(ex, exc_info=True)
        finally:
            return candidates

    def _get_from_woxikon(self, hint_text:str, word_length:int) -> list[str]:

        letters = []
        for cnt in range(word_length):
            letters.append("")

        query = replace_umlauts(hint_text)
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

        except Exception as ex:
            LOG.error(ex, exc_info=True)
        finally:
            return candidates

    def lookup_answers_online(self, hint_text: str, word_length:int, use_threads:bool=False):

        if not hint_text:
            return []

        all_results = []

        # clean hint text
        hint_text = "".join(c for c in hint_text if c not in ",.:")

        if use_threads:
            all_results = run_functions_in_parallel( self.source_functions,
                arg1=hint_text, arg2=word_length)
        else:
            for source_fn in self.source_functions:
                try:
                    words = source_fn(hint_text, word_length)
                    all_results.extend(words)
                except Exception as e:
                    print(f"error at{source_fn.__name__}: {e}")

        # Duplikate entfernen, sortieren
        try:
            result = sorted(set(all_results))
        except Exception as ex:
            LOG.error(ex, exc_info=True)

        return result
