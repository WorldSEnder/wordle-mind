# Automating wordle

To use this, switch to the python directory, then install the requirements into a virtual environment:

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

When first running, it will compute all possible wordles from the dictionary and the list of additional possible guesses.
This information is cached and will be available on future runs. Simply run

```
python analyze.py
```

The bot will propose words it wants you to play and you resond with the colors the puzzle returns for the guess. Example
session:

```
(.venv) $ python analyze.py 
Analyzing possible guesses: 100%|██████████████████████████| 12972/12972 [00:08<00:00, 1460.80it/s]
Play the word 'raise' next: ...GG
Analyzing possible guesses: 100%|████████████████████████| 12972/12972 [00:00<00:00, 120361.36it/s]
Play the word 'cloth' next: ..GYY
You win by playing 'those'
```

