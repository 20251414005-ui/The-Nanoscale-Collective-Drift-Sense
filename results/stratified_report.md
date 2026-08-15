# Stratified Accuracy Report (tolerance = 5.0px)

Built from 800 pairs in results/full_results.csv. Buckets are terciles (roughly equal-sized groups) of each dimension's actual observed range in this dataset, not fixed external cutoffs.

## By Scale

| Bucket | Range | N pairs | Classical @5.0px | DL @5.0px |
|---|---|---|---|---|
| Low | 9.00–9.61 | 266 | 43.6% | 59.8% |
| Mid | 9.61–10.33 | 267 | 33.3% | 57.7% |
| High | 10.33–11.00 | 267 | 43.1% | 57.3% |

## By Rotation (degrees)

| Bucket | Range | N pairs | Classical @5.0px | DL @5.0px |
|---|---|---|---|---|
| Low | -2.98–-0.85 | 265 | 40.4% | 57.4% |
| Mid | -0.85–1.09 | 268 | 36.6% | 59.7% |
| High | 1.09–3.00 | 267 | 43.1% | 57.7% |

## By Noise (search-image sigma)

| Bucket | Range | N pairs | Classical @5.0px | DL @5.0px |
|---|---|---|---|---|
| Low | 10.01–12.79 | 265 | 40.4% | 59.2% |
| Mid | 12.79–15.34 | 268 | 39.2% | 59.7% |
| High | 15.34–18.00 | 267 | 40.4% | 55.8% |