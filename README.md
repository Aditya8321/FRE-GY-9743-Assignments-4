# FRE-GY 9743 - Assignment 4

**Aditya Amit Shah | as22008**

The [completed notebook](hw4_ifr_yield_curve.ipynb) contains the written answer,
all computational tasks, saved results, and analytic verification checks. A
[readable HTML copy](submission/Assignment_4_as22008.html) and a
[three-page solution report](output/pdf/Assignment_4_as22008.pdf) are included.

## Written answer

Daily OTC collateral secures a future claim and normally carries interest at
the collateral remuneration rate. Standard futures variation margin settles
daily profit or loss and resets the continuing contract's value to zero. Under
the assignment's assumption that the next settlement period's discount factor
is known, discounting cancels from the zero-value equation for the futures
quote. The notebook explains the distinction with a cash example, equations,
contractual qualifications, and primary references.

## Implementation

All six assigned library TODOs and every notebook placeholder are completed:

- Piecewise-constant left-continuous interpolation, flat extrapolation,
  integration, and both analytic ordinate gradients.
- Local discount factors from integrated instantaneous forward rates.
- Recursive composition of referenced curve factors, with cycle detection.
- Creation and pricing of the five-year fixed-accrual cashflow.
- Analytic price sensitivities to all ten SOFR nodes and the funding node,
  reported per unit decimal IFR with node times and the payment fixed.

The supplied inputs are retained. Product accrual uses ACT/360; curve time uses
ACT/ACT (ISDA). Tenors use Following/USGS. Payment currency is a reserved starter
argument and does not perform currency conversion.

## Default-input results

| Quantity | Result |
| --- | ---: |
| Value date | 2026-10-07 |
| Payment date | 2031-10-07 |
| ACT/360 product accrual | 5.072222222222 |
| ACT/ACT curve time | 5.000000000000 |
| Cashflow | 507.222222222222 |
| Discount factor | 0.852155991697 |
| Price | 432.232455788284 |
| Funding-node derivative | -2161.162278941419 |

The zero funding spread leaves the complete SOFR and funding curves equal,
while its derivative remains -T times the price. Only the first five SOFR
ordinates affect this payment.

## Run and verify

Use Python 3.11 or 3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
jupyter notebook hw4_ifr_yield_curve.ipynb
```

Select that Python environment and run all cells in order. To run the
independent checks:

```sh
python -m unittest discover -s tests -v
```

All ten notebook code cells executed successfully, and all nine independent
tests passed with Python 3.12.14, NumPy 1.26.4, pandas 2.3.1 and QuantLib 1.41.
Checks cover hand-calculated interpolation weights, calendar/day-count
conventions, positive and negative funding spreads, recursive references and
cycles, flat extrapolation, invalid dates, and the complete price and risk.
Finite differences are used only in the independent tests; the notebook's
reported risk is calculated analytically.

## Submission files

Keep the full repository, including the notebook, library, static convention
files and tests, together. The PDF and HTML are readable companions to the
completed code.

Following the [course submission instructions](https://immense-chocolate-92e.notion.site/Github-Setup-HW-submission-2f1831d0b1d580eaacf8d8050867e9d9),
work is prepared on the personal branch `Aditya_Shah_Branch` in the student's
own fork. Submit the [personal branch URL](https://github.com/Aditya8321/FRE-GY-9743-Assignments-4/tree/Aditya_Shah_Branch)
through Brightspace. The written answer is included as notebook Markdown,
which the course permits; the PDF is an additional readable copy. Do not
submit only a Python file.


