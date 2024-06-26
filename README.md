# mincda2

Clone virtual environment
```
conda create -n mindca2 --file package-list.txt
```

Run Fixed Channel Allocation (FCA) with a call rate of 150 calls per hour:
```python
python main.py fca --call_rateh 150
```

Run Reduced-Sate SARSA (RS-SARSA) with a call rate of 150 calls per hour:
```python
python main.py rs_sarsa --call_rateh 150
```
