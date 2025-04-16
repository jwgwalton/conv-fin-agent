# Methodology

I've decided to formulate a solution that uses a multi-agent swarm to decompose the problem into smaller tasks & then solve those tasks

I would normally try a simpler solution & only go for the more complex


## Metrics
I investigate the data in eda.py to get a better understanding of the expected outputs.

The answers include a mix of formats. 
- Percentages (both positive & negative)
- Currency
- Decimals

Given that there are multiple units & variations of unit, i'm going to use an exact match for the evaluation metric. 
I could use different metrics for the numeric & non-numeric outputs, but for simplicity i'm going to use exact match for both.
Other options could be a F1 score (the harmonic mean of precision & recall).