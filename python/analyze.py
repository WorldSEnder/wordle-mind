from collections import defaultdict, namedtuple
from functools import reduce, partial
from itertools import combinations_with_replacement, chain, product
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import os
from math import comb
import gzip
import array
from multiprocessing import Manager

def huffcode(wordles):
    """pretty much unused, not worth it"""
    from dataclasses import dataclass, field
    from heapq import heappush, heappop, heapify
    from typing import Any

    cnts = defaultdict(int)
    for v in wordles._wordle_dict.values(): cnts[v] += 1

    @dataclass(order=True)
    class Node:
        frequency: int
        symbol: Any = field(compare=False)
        children: Any = field(compare=False)        
    nodes = [Node(c, w, None) for (w, c) in cnts.items()]
    while len(nodes) > 1:
        l = heappop(nodes); r = heappop(nodes)
        heappush(nodes, Node(l.frequency + r.frequency, None, (l, r)))
    codes = {}
    queue = [('', nodes[0])]
    while queue:
        c, n = queue.pop()
        if n.symbol: codes[n.symbol] = c
        else: queue.append(('0' + c, n.children[0])); queue.append(('1' + c, n.children[1]))

def read_dictionary(filename):
    with open(filename) as dict_h:
        words = set(l.rstrip() for l in dict_h.readlines())
        assert all(len(w) == 5 for w in words)
        return words

COMBINATIONS_TABLE = [[comb(i, j) for j in range(6)] for i in range(31)]
class WordleResult(namedtuple('WordleResult', 'number')):
    __slots__ = ()
    def encode(self):
        return self.number.to_bytes(3, byteorder='big')
    @staticmethod
    def decode(encoded):
        number = int.from_bytes(encoded[0:3], byteorder='big')
        return WordleResult(number)
    @staticmethod
    def from_readable(exact, approximate):
        # 0..<32 (5 bits)
        _exact = reduce(lambda x, i: x | (1 << i), exact, 0)
        # n = 26, m = 0..=5
        # equivalently n = 27, m = 5, combinations with replacements:
        # C(27+5-1, 5) = 169911
        # can be encoded in 0..<169911 (18 bits):

        # pad with '{' to convert multiset -> combinations with replacement
        approx = ''.join(c * i for (c, i) in sorted(approximate.items())).ljust(5, chr(ord('z') + 1))
        _approximate = sum(
            # correct for 'a' being the first character
            # add the index to convert combinations with replacement -> combinations
            # encode with https://web.archive.org/web/20170325012457/https://msdn.microsoft.com/en-us/library/aa289166.aspx
            COMBINATIONS_TABLE[ord(c) - ord('a') + i][i + 1]
            for (i, c) in enumerate(approx)
        )
        # aka ~3 bytes
        return WordleResult(_exact + _approximate << 5)
    @staticmethod
    def parse_for_guess(guess, user_input):
        correct_indices = []
        correct_chars = defaultdict(int)
        for (i, x) in enumerate(user_input):
            if x == '!': correct_indices.append(i)
            elif x == '?': correct_chars[guess[i]] += 1
        return WordleResult.from_readable(correct_indices, correct_chars)

def wordle(word, guess):
    assert len(word) == 5
    assert len(guess) == 5
    exacts = list()
    leftover_word, leftover_guess = defaultdict(int), defaultdict(int)
    for i in range(5):
        if word[i] == guess[i]:
            exacts.append(i)
        else:
            leftover_word[word[i]] += 1
            leftover_guess[guess[i]] += 1
    approximate = {}
    for letter, remaining in leftover_word.items():
        if letter in leftover_guess:
            approximate[letter] = min(remaining, leftover_guess[letter])
    return WordleResult.from_readable(exacts, approximate)

class WordleDict():
    @staticmethod
    def _at_idx(u, v, guess_list):
        return wordle(guess_list[u], guess_list[v])
    @staticmethod
    def _fill_entry(table, dict_len, guess_list, indices):
        for idx in indices:
            u, v = idx
            result = WordleDict._at_idx(u, v, guess_list)
            table[u + v * dict_len] = result.number
            if v < dict_len:
                table[v + u * dict_len] = result.number
    @staticmethod
    def _create(dictionary, allowed_list, read_table):
        dictionary = sorted(dictionary)
        allowed_list = sorted(allowed_list)
        dict_len = len(dictionary)
        allw_len = len(allowed_list)
        guess_list = dictionary + allowed_list

        word_indices = chain(
            combinations_with_replacement(range(dict_len), 2),
            product(range(dict_len), range(dict_len, dict_len + allw_len)),
        )
        total = (len(dictionary) * (len(dictionary) + 1)) // 2
        total += len(dictionary) * len(allowed_list)
        def build_table():
            table = array.array('I', [0] * (dict_len * (dict_len + allw_len)))
            fill_entry = partial(WordleDict._fill_entry, table, dict_len, guess_list)
            fill_entry(tqdm(word_indices, total=total, desc="Generating wordles"))
            #indices = list(word_indices)
            #def split(a, n):
            #    k, m = divmod(len(a), n)
            #    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
            #thread_map(fill_entry, split(indices, 8))
            return table

        table = read_table(build_table)
        return WordleDict(table, range(dict_len), dict_len, guess_list)

    @staticmethod
    def for_dictionary(dictionary, allowed_list):
        def read_table(build_table):
            return build_table()
        return WordleDict._create(dictionary, allowed_list, read_table)

    def __init__(self, table, remaining_words, dict_len, guess_list):
        self._wordle_table = table
        self._remaining_words = remaining_words
        self._dict_len = dict_len
        self._guess_list = guess_list

    @property
    def word_indices(self):
        return self._remaining_words

    @property
    def guess_indices(self):
        return range(len(self._guess_list))

    @property
    def guess_list(self):
        return self._guess_list

    def __getitem__(self, idx):
        word, guess = idx
        return self._wordle_table[guess * self._dict_len + word]

    def to_wordle_code(self, wordle_result):
        return wordle_result.number

    def refine(self, remaining):
        return WordleDict(self._wordle_table, remaining, self._dict_len, self._guess_list)

    def write_to(self, file_h):
        with gzip.GzipFile(filename='wordles', mode='wb', fileobj=file_h) as gfile_h:
            total = len(self._wordle_table) * self._wordle_table.itemsize
            with tqdm.wrapattr(gfile_h, "write", total=total, desc="Caching wordles analysis") as obs_file_h:
                self._wordle_table.tofile(obs_file_h)

    @staticmethod
    def read_from(dictionary, allowed_list, file):
        def do_read(file_h):
            ar = array.array('I')
            dict_len = len(dictionary)
            gues_len = dict_len + len(allowed_list)
            ar.fromfile(file_h, dict_len * gues_len)
            return ar
            # def at_idx(_u, _v, _words):
            #     return WordleResult.decode(file_h.read(3))
            # return build_table(at_idx)
        if isinstance(file, str):
            def read_table(_build_table):
                with gzip.open(file) as file_h:
                    return do_read(file_h)
        else:
            def read_table(_build_table):
                return do_read(file)
        return WordleDict._create(dictionary, allowed_list, read_table)

def analyze_dict(wordles):
    def analyze(guess):
        analysis = defaultdict(list)
        for word in wordles.word_indices:
            wordle_code = wordles[(word, guess)]
            analysis[wordle_code].append(word)
        return analysis
    def quick_analysis(guess):
        analysis = defaultdict(int)
        for word in wordles.word_indices:
            wordle_code = wordles[(word, guess)]
            analysis[wordle_code] += 1
        return analysis
    def rate_guess(guess):
        """Compute the largest possible remaining result set"""
        analysis = quick_analysis(guess)
        return max(v for v in analysis.values())
    best_guess_idx = min(tqdm(wordles.guess_indices, desc="Analyzing possible guesses"), key=rate_guess)
    best_guess = wordles.guess_list[best_guess_idx]
    return (best_guess, analyze(best_guess_idx))

def main_loop(wordles):
    remaining_wordles = wordles
    while len(remaining_wordles.word_indices) > 1:
        best_word, best_analysis = analyze_dict(remaining_wordles)
        response = yield best_word
        result = WordleResult.parse_for_guess(best_word, response)
        remaining = best_analysis[wordles.to_wordle_code(result)]
        remaining_wordles = remaining_wordles.refine(remaining)
    if not remaining_wordles.word_indices:
        raise Exception("Impossible wordle, is the dictionary correct or did you mess up an input?")
    solved_index = list(remaining_wordles.word_indices)[0]
    return wordles.guess_list[solved_index]

def setup():
    if not os.path.exists('wordles.gz'):
        regenerate_wordles()
    dictionary = read_dictionary('dictionary')
    allowed_list = read_dictionary('allowed_list')
    return WordleDict.read_from(dictionary, allowed_list, 'wordles.gz')

def regenerate_wordles():
    dictionary = read_dictionary('dictionary')
    allowed_list = read_dictionary('allowed_list')
    wordles = WordleDict.for_dictionary(dictionary, allowed_list)
    with open('wordles.gz', 'wb') as wordles_h:
        wordles.write_to(wordles_h)

if __name__ == "__main__":
    wordles = setup()
    it = main_loop(wordles)
    user_input = None
    while True:
        try:
            best_word = it.send(user_input)
        except StopIteration as stop:
            print(f"You win by playing '{stop.value}'")
            break
        user_input = input(f"Play the word '{best_word}' next: ").rstrip().ljust(5, ' ')
