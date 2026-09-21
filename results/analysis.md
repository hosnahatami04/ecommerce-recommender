# Cold-start, coverage, and popularity-bias analysis

All numbers below are read directly from committed files in `results/`:
`segment_analysis.json`, `headline_comparison.json`, `longtail.json`,
`user_illustration.json`.

## Segment sizes (test-period users, by training-history length)

| Segment | Users | % of test users |
|---|---|---|
| 0 (cold) | 129,613 | 92.3% |
| 1-2 | 6,867 | 4.9% |
| 3-10 | 2,904 | 2.1% |
| 11+ | 991 | 0.7% |

The overwhelming majority of users evaluated at test time have **zero**
training-period history. This single fact explains most of what
follows.

## Recall@10 by segment and model

| Segment | Popularity | ALS | Two-stage |
|---|---|---|---|
| 0 | 0.891% | 0.000% | 0.986% |
| 1-2 | 0.420% | 1.042% | 0.610% |
| 3-10 | 0.271% | 0.978% | 0.744% |
| 11+ | 0.005% | 1.216% | 1.074% |

## Findings

**ALS is completely blind on cold users, and this alone explains its
overall loss to the baseline.** ALS scores exactly 0% Recall@10 for
the 129,613 cold-start users -- it has no matrix row for them and
nothing to base a recommendation on. Since this segment is 92% of all
test users, it dominates the overall average reported in
`results/als.json`, making ALS look uniformly bad. It isn't: for every
segment with *any* training history (1-2, 3-10, 11+), ALS actually
**beats** the popularity baseline, in some cases by more than 2x.

**Popularity gets worse, not better, as users become more active.**
Recall@10 for the popularity baseline drops steadily from 0.89% (cold
users, where it's the only reasonable fallback) down to 0.005% for the
most active segment (11+ events). Active users have moved past generic
bestsellers; recommending the same fixed list to them is close to
useless.

**The two-stage model is the most consistent performer across every
segment.** It never collapses to zero (the popularity fallback in
candidate generation rescues cold users) while still beating the
baseline in the 0 and 11+ segments and staying competitive with plain
ALS everywhere else. This is the practical case for the two-stage
architecture: no single one of the three models is best in every
segment, but the combination degrades most gracefully.

## Cold-start reality check

For the 0-event segment, the only real options are popularity and item
metadata -- there is no user behavior to personalize from. This project
already relies on the popularity fallback inside candidate generation
(`src/models/candidates.py`) for exactly this case; no separate
content-based fallback was built, since the two-stage model's cold-user
performance (0.986% Recall@10) already slightly exceeds plain
popularity (0.891%) without one.

## Coverage and long-tail

Catalog coverage@10 (fraction of the 235,061-item catalog ever
recommended to anyone):

| Model | Coverage@10 |
|---|---|
| Popularity | 0.0077% |
| ALS | 1.045% |
| Two-stage | 2.412% |

The full popularity-rank-vs-recommendation-frequency data is in
`results/longtail.json`. Popularity concentrates almost all
recommendation volume on a handful of the most popular items by
construction. ALS and the two-stage model both spread recommendations
across a meaningfully larger slice of the catalog, with the two-stage
model reaching more than twice the coverage of ALS alone.

## Popularity bias illustration

`results/user_illustration.json` compares one heavy user (6,667
training events) against one light user (1 training event):

- **Popularity model**: returns the *identical* top-10 list for both
  users -- by construction, it cannot do otherwise.
- **ALS**: returns completely different top-10 lists for the two users
  (zero overlap), demonstrating real personalization when the model
  has any signal to work with.
- **Two-stage**: also returns different, though partially overlapping,
  top-10 lists for the two users -- one item (62549) appears in both
  lists, suggesting the reranker still leans on some shared
  popularity/recency signal even while personalizing.

## Bottom line

Averages hide failure. The single overall Recall@10 number for ALS
(from Phase 3) made it look like a uniformly weak model; segmenting by
training history shows it is a strong model for anyone with prior
behavior and a non-functional one for anyone without it. The two-stage
architecture's real value, visible only once segmented, is not that it
wins everywhere -- it doesn't always beat ALS on active users -- but
that it is the only model that performs reasonably across the entire
population, cold users included.
