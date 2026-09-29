# Literature Summary: Prediction Markets as Macro Data (Kalshi)

15.S06 project literature review. For each paper in `Literature/` this file covers (a) how it relates to our three research questions and (b) the caveats in its analysis that matter if we build on it. A cross-paper synthesis and a list of design decisions come first.

**Our research questions (from `Project Description.docx`):**
- **RQ1 – Calibration.** Do Kalshi macro prices beat SPF and option-implied distributions? Is there a favorite–longshot bias (FLB)? Compare WIRP and Kalshi implied probabilities of rate hike/hold/cut states over t ∈ [−60, −1] days, scored with Brier score and cross-entropy. *Hypothesis: Kalshi overprices the tails.*
- **RQ2 – Price discovery.** How fast do Kalshi prices absorb CPI, NFP and FOMC news? Run an event study of Kalshi jumps against high-frequency fed funds futures moves.
- **RQ3 – Forecasting.** Use Kalshi data to predict unemployment, GDP and CPI, possibly with ML or "economic transformers".

**Papers covered**

| Short name | Paper | Main RQ | One-line takeaway |
|---|---|---|---|
| DKW | Diercks, Katz & Wright (2026), *Kalshi and the Rise of Macro Markets*, FEDS 2026-010 | RQ1, RQ2 | Already shows Kalshi ≈ surveys/futures on point accuracy; no scoring rules, no FLB test, daily data only |
| BDW | Bürgi, Deng & Whelan (2026), *Makers and Takers* | RQ1 | Strong FLB in the pooled Kalshi cross-section, much weaker and statistically fragile in Economics contracts |
| BO | Bartlett & O'Hara (2026), *Adverse Selection in Prediction Markets* | RQ2, RQ1 | Microstructure toolkit (signed trades, log-odds λ); "broad-based" markets look calibrated, but that pool is ~98% crypto, equity and FX |
| WZ | Wolfers & Zitzewitz (2004), *Prediction Markets*, JEP 18(2) | Framing | Contract design → statistic mapping; precursor macro markets (Economic Derivatives) matched consensus on tiny samples |
| KKMX | Kelly, Kuznetsov, Malamud & Xu (2026), *Artificial Intelligence Asset Pricing Models* | RQ3 | Cross-sectional attention SDF for equities; only a heavily regularized linear-attention idea transfers to our sample size |

> File note: `Literature/Wolfers, Zitzewitz (2006).pdf` is actually the **2004** JEP article (vol. 18(2), pp. 107–126), which matches reference [4] in the project description. The filename year is wrong.

---

## 1. Cross-paper synthesis

### RQ1 – Calibration and the tail hypothesis

**What is already done.** DKW already answer "does Kalshi match surveys and futures on point accuracy":
- Fed funds: Kalshi is roughly equal to the NY Fed SME.
- Fed funds futures: DKW compare them visually at every horizon, and formally only on FOMC day.
- Headline CPI: Kalshi beats Bloomberg consensus, at p < 0.10.

If we pitch RQ1 as "does Kalshi beat the benchmarks", a referee will say it has been done. What remains open:
1. Proper scoring rules (Brier, log score / cross-entropy) on the *full distribution*, scored at fixed horizons over [−60, −1].
2. A formal FLB / reliability test on macro contracts specifically.
3. A scored comparison against WIRP (and SPF where horizons allow).

**The evidence on the tail hypothesis is mixed, and weaker for macro than the headlines suggest.**
- **For it:**
  - BDW's pooled Kalshi sample shows a large FLB: 1–10¢ contracts lose ≈57% after fees.
  - DKW find Kalshi puts much more mass than SPF on severe stagflation outcomes (Fig. 13).
  - DKW's PIT tests show upper-tail overweighting for headline CPI and unemployment.
  - WZ report the bias is "more pronounced on smaller-scale exchanges."
- **Against, or not established:**
  - In BDW's *Economics* category the constant is insignificant (−0.978, s.e. 0.972). In cents, the fitted mispricing is larger for *underpriced favorites* (~2.3¢ at 95¢) than for overpriced longshots (<1¢ at 5¢).
  - BO find "broad-based" markets essentially calibrated (+0.2pp), but that pool is mostly crypto, equity-index and FX.
  - DKW's fed funds densities pass PIT calibration.
- **Net:** keep the hypothesis, but treat it as genuinely open, not as the expected result.

**The same mispricing looks large or tiny depending on the metric.** BDW's own example is a 5¢ contract that wins 3% of the time. That is a 2pp probability error but a −40% return.
- The Brier score barely penalizes 2pp errors in the tails; the log score does.
- Report both, with the choice justified before looking at the results.
- Also report a reliability diagram by price bin.
- Otherwise the conclusion on the hypothesis will largely follow from which metric we picked.

**Several mechanical effects inflate apparent tail overpricing. Control for each before calling it a bias:**
1. **The 1¢ floor and tick.** A state with true probability 0.2% cannot trade below 1¢ (BDW, DKW). Treat 1–2¢ prices as censored, or report results with and without them.
2. **Bracket overround.** BDW's "Exclusive Numerical" fit is negative at every price, which suggests ladder prices sum to more than $1 (our inference). Normalize ladders to sum to one before scoring.
3. **Stale last-trade prices in the tails.** DKW carry forward untraded strikes, so the tails are the stalest part of the distribution.
4. **Bid-ask bounce.** Takers mostly buy YES (BO: 56% of volume in broad-based markets), so last trades on longshots tend to print at the ask. Use midpoints where hourly candlesticks exist, and check sensitivity.
5. **Closing-day trades.** Much of BDW's FLB sits on the closing day. Our [−60, −1] window avoids t = 0. Keep it that way.
6. **Horizon and carry.** At about 4% over 60 days, carry is ≈0.66¢ on a $1 payoff. That is below the tick, so it cannot explain tail overpricing on its own. A t = −60 vs t = −1 comparison still helps separate a horizon effect from a belief bias.

**WIRP is structurally a weak benchmark for the tails.**
- WIRP is built from fed funds futures and OIS, which give an *expected rate*. Turning that into state probabilities assumes a two-state split between adjacent outcomes, as in DKW Fig. 10. So WIRP puts essentially zero mass on non-adjacent states *by construction*.
- Comparing tail pricing against WIRP therefore tests Kalshi against a model that cannot have tails.
- **Design choices:**
  - Score Kalshi vs WIRP on a common, coarse event definition, e.g. P(cut) vs P(no cut), or the two states adjacent to the WIRP mean.
  - Test the tail hypothesis *within* Kalshi via reliability by price bin, rather than against WIRP.
  - DKW argue SOFR and fed-funds options are poor meeting-level tail benchmarks. We should still consider them for robustness.

**Kalshi and WIRP are both risk-neutral (Q-measure) prices** (WZ fn. 3; DKW). "Overpriced tails" relative to realized frequencies could be a risk premium rather than a behavioral bias, especially for recession or stagflation states. One partial fix is to compare the FLB in macro contracts with the FLB in weather contracts, where risk premia should be negligible (BO use weather as a placebo).

**Power is low.**
- **Effective N is small.** Daily snapshots over [−60, −1] are strongly autocorrelated and share one outcome, so effective N ≈ number of FOMC meetings (~30–40 since the Fed decision series began) or releases (~40–60 per series).
- **Cluster by event.** BO and DKW do not cluster by event (BDW do), and their precision is overstated as a result. Don't copy them.
- **Differences will likely be insignificant.** Expect Brier/log-score differences between Kalshi and WIRP to be insignificant at conventional levels. Say so up front and pool FOMC, CPI and NFP ladders for the FLB test.

### RQ2 – Price discovery

**The ground is mostly open.**
- DKW's event studies use daily close-to-close changes.
- BO have no announcement windows.
- BDW use daily snapshots.
- None of them compares Kalshi against high-frequency fed funds futures.
- BDW point to **Swanson, Wang & Wu (2025)** on Fed announcements and Kalshi macro-release markets. That is probably the closest prior work to RQ2 and should be read next (not in our folder).

**Design issues the literature surfaces:**
- **Settlement is not discovery.** A CPI or NFP bracket contract *settles on* the release, and the FOMC decision market settles on the announcement. Their post-event "jumps" are mechanical. The informative object is contracts the release *informs but does not settle*, e.g. the next-FOMC rate ladder after a CPI or NFP print, or later-meeting ladders after an FOMC decision. We can also measure pre-release drift in the settling contract.
- **Build prices from trades.** Kalshi's public API gives trade-level data with fine timestamps and a `taker_side` flag. Quotes exist only as hourly candlesticks, covering ≈5% of broad-based trades (BO). There is no historical order book, so intraday windows must be built from trades.
- **Measurement methods:**
  - Use log-odds price changes, Δlog(p/(1−p)), which behave better near the bounds (BO).
  - Restrict to interior prices (30–70¢) or the first 80% of a market's life to avoid mechanical convergence to settlement (BO).
  - BO's weather appendix (B.4) is the closest template: Kalshi flow and log-odds prices lined up against a high-frequency public signal with leads and lags. Replace temperature with fed funds futures.
- **Separating statement from press conference.** DKW find the FOMC statement moves the mean and the press conference lowers skewness. Daily data cannot separate the two, but intraday Kalshi trades can. This is a clean contribution if liquidity allows.
- **Data gap.** We need an intraday fed funds futures source (CME tick data or similar). WRDS alone may not cover it. Confirm early.

### RQ3 – Forecasting macro quantities

- **None of the papers builds a forecasting model on Kalshi data.** DKW set the benchmark to beat: the raw Kalshi-implied mean (headline CPI MAE ≈ 6–7bp on release day). That is a much harder bar than Bloomberg consensus.
- **KKMX is a weak justification for "economic transformers".** It is a cross-sectional equity SDF trained on thousands of stocks × 132 characteristics × decades. We have roughly 50 releases per series. Its complexity-scaling evidence does not transfer. What does transfer is a ridge-regularized *linear* attention layer with a closed-form fit, used to pool information across strikes or related markets.
- **Recommended structure:**
  - *Headline model:* a low-dimensional ridge/OLS on Kalshi-implied mean, variance and skew plus consensus, estimated on an expanding window.
  - *Extension:* linear attention over strikes or markets, reported only if it beats the headline model out of sample.
  - *Most novel target:* quantities Kalshi does not directly price at that horizon. For example, nowcast GDP or unemployment from the FOMC, CPI and jobless-claims ladders before those quantities' own markets open. There the raw target-market price is not available to dominate.
  - *Benchmarks:* SPF is quarterly and a poor benchmark for monthly prints. Use Bloomberg consensus, the Cleveland Fed nowcast and the Kalshi-implied mean. Use SPF only for GDP.

### Data and implementation facts gathered from the papers

- **Trade data.** The Kalshi public REST API returns trades (price in cents, count, timestamp, `taker_side`), market metadata, settlement results and hourly candlesticks (BO, BDW). No account IDs, no historical order book.
- **Series start dates** (DKW Table 1):

  | Series | Start |
  |---|---|
  | CPI MoM | Jun 2021 |
  | CPI YoY | Nov 2022 |
  | Core CPI MoM | Jun 2022 |
  | Unemployment | Jul 2021 |
  | Payrolls | Mar 2023 |
  | GDP | Q2 2021 |
  | Fed Decision | May 2023 |
  | Fed Target Rate | Dec 2021 |

- **Fed funds Target Rate contracts** settle on the *upper bound* of the target range (DKW).
- **Fees.**
  - The taker fee is 0.07·P(1−P) per contract, rounded up. As a share of price that is 1.75% at 50¢ and ~6.7% at 5¢ (BDW).
  - **Makers pay fees since April 2025**, so pre-2025 maker/taker results may not hold in our sample.
- **Filters used in prior work:**
  - BDW: final volume ≥ $1,000, final spread ≤ 20¢. These rely on information known only at close, so use ex-ante versions.
  - BO: 30–70¢ band, and at least 50 five-minute intervals for impact estimation.
- **DKW's code and daily distributions** are promised "subject to approval" at EconFutures.com / GitHub. Check before building our own pipeline.

### Items to verify (not from the provided papers)

- Kalshi's current policy on paying interest on cash and open positions. It affects the carry argument, which is small either way.
- Wolfers & Zitzewitz (2006), "Interpreting Prediction Market Prices as Probabilities," NBER WP 12200. It covers price vs. mean belief under heterogeneous beliefs.
- Gürkaynak & Wolfers (2005/2006), "Macroeconomic Derivatives." This is the longer-sample follow-up on the Economic Derivatives markets. Cite it rather than WZ Table 3.
- Swanson, Wang & Wu (2025), cited by BDW. Likely the closest prior work to RQ2.
- Page numbers below are printed page numbers. The PDF viewer page can be off by one.

---

## 2. Paper-by-paper

### 2.1 Diercks, Katz & Wright (2026) — Kalshi and the Rise of Macro Markets

#### A. Relevance to our project

##### What the paper does

Diercks, Katz and Wright (FEDS 2026-010, Feb 2026) turn Kalshi binary contracts into daily risk-neutral distributions for the fed funds target at each FOMC meeting, CPI (headline and core, MoM and YoY), unemployment, and annual CPI and GDP. They compare these with fed funds futures, SOFR options, OIS, the FRBNY Survey of Market Expectations (SME), Bloomberg consensus, Blue Chip and SPF. They then run daily event-study regressions of changes in the moments of the fed funds distribution on macro surprises and monetary policy shocks. The sample runs from 2022 to 2025.

##### Relevance to our research questions

**RQ1 (Calibration).** This is the RQ the paper overlaps with most, but its tools differ from ours.
- *Point accuracy.* Figure 1 plots fed funds MAE against days before FOMC, out to 160 days. Kalshi is roughly equal to the SME and slightly better than fed funds futures at about 60 days out. On FOMC day (Table 3B), the Kalshi median and mode have zero MAE. Fed funds futures have MAE 0.010, and the gap is significant at 5% by Diebold-Mariano. One meeting drives this: September 2024 (a 50bp versus 25bp cut, p. 24). A DM test on a loss differential that is almost always zero is fragile, and we should not repeat the "significant" claim uncritically.
- *Macro releases (Table 3A).* For headline CPI, MAE is 0.063 for the Kalshi median and mode versus 0.081 for Bloomberg (p<0.10). Core CPI and unemployment are statistically tied.
- *Density calibration (Figure 15).* They use PIT tests (Kolmogorov-Smirnov and Cramer-von Mises style statistics, Rossi-Sekhposyan bootstrap) at 0 and 28 days ahead. Uniformity is borderline rejected for unemployment (p = 0.03-0.05) and headline CPI (p = 0.03-0.06). It is not rejected for core CPI (0.20-0.44) or fed funds (0.25-0.62). The rejections come from low outcomes happening too often: prices overstate high inflation and high unemployment.
- *What they do not do.* No Brier score, log score or cross-entropy. No calibration curve or favorite-longshot (FLB) test by price bucket. No formal comparison with WIRP or SPF.

**Is our hypothesis supported?** Partly, and only indirectly.
- Figure 13 shows Kalshi putting "much more weight" than SPF on severe outcomes (2025 CPI ≥4%, GDP <0%). That fits tail overpricing, but they cannot separate it from risk premia or retail-versus-professional differences.
- The PIT evidence points to an asymmetric upper-tail bias, not a symmetric longshot bias.
- Against a strong version of the hypothesis: the fed funds densities are well calibrated, and the mode never missed on FOMC day.
- Our hypothesis is therefore neither pre-empted nor established. The WIRP-style comparison is partly done. Figure 10 converts fed funds futures into a two-outcome distribution for the September and October 2025 meetings and argues this gives too little uncertainty. That comparison is qualitative, though, and not scored.

**RQ2 (Price discovery).** The paper says less here than the title suggests.
- Everything is daily: the change from the end of the previous day to the end of the release day (Tables 4 and 5). There is no intraday analysis, no measure of adjustment speed, and no lead-lag test against high-frequency fed funds futures.
- Findings:
  - A 10bp CPI surprise raises the mean of the next-FOMC fed funds distribution by about 3-4bp (coefficient 0.320-0.413, N = 113, R² = 0.38 for the mean).
  - The PCE effect is small (0.012).
  - NFP is insignificant (N = 108).
  - Positive CPI surprises move the mean about 4x more than negative ones (Figure 17).
  - Variance falls on release days, most after zero surprises.
  - The FOMC statement shock (SF Fed USMPD, Acosta et al. 2025) raises the mean (0.872) and the variance. The press-conference shock lowers skewness (-2.875**, N = 94).
- Figure 6 is an anecdotal intraday narrative for July 2025 (Waller and Bowman comments, June NFP).
- The gap is wide open: intraday jump timing, pre-announcement drift, and speed relative to fed funds futures and SOFR futures.

**RQ3 (Prediction).** Close to nothing beyond RQ1.
- They show the Kalshi mean, median and mode are competitive forecasts (Figure 14, horizons up to 50 days; Table 3).
- They build no forecasting model, no combination with other predictors, and no ML.
- The one relevant result is a hard benchmark: headline CPI MAE of about 7bp on release day. Any model we build has to beat the raw Kalshi mean, not just the Bloomberg consensus.

##### What we can borrow

- **Distribution construction (Section 3, pp. 13-14).**
  - Treat the last-traded Yes price as the probability. For "above strike" series, take the probability of a bin as the difference between adjacent strikes (e.g. 0.40 - 0.22 = 18% for the 4.00-4.25 bin).
  - If a strike did not trade that day, carry forward its last price.
  - Enforce a monotone CDF by building outward from the mode. They report this beats building from either tail.
  - If two strikes have the same price, give the mass to the bin closer to the mode.
  - Bid-ask midpoints performed worse because of wide tail spreads (Appendix A). We should still test this ourselves for the tails, since that is exactly where the FLB lives.
- **Contract facts.**
  - The fed funds Target Rate series settles on the upper bound of the target range.
  - Prices are bounded between $0.01 and $0.99, so tail bins are censored. This matters for any FLB test.
  - Maximum exposure is $7M per market.
  - Series start dates (Table 1): CPI MoM Jun 2021, CPI YoY Nov 2022, Core CPI MoM Jun 2022, Unemployment Jul 2021, Payrolls Mar 2023, GDP Q2 2021, Fed Decision May 2023, Fed Target Rate Dec 2021.
  - Volume is often above 1M contracts per meeting, with close to 100M for September 2025 (Figures 2-3).
- **Data sources.**
  - Kalshi trade-level data, which they scraped. They give no tickers or API endpoints.
  - Their daily distributions and code are promised, subject to approval, at EconFutures.com and on GitHub. We should check whether they exist before building our own pipeline.
  - Bloomberg consensus for surprises (actual minus consensus).
  - SF Fed USMPD for monetary policy shocks.
- **Methods.**
  - Fed funds futures to two-state probabilities using the next month's contract (Figure 10). This is the same logic as WIRP.
  - Diebold-Mariano tests for comparing point forecasts.
  - PIT uniformity tests with the Rossi-Sekhposyan bootstrap.
  - HC3 standard errors in the event-study regressions.
- **Horizons.** Our t in [-60, -1] window sits inside their 160-day FOMC window (Figure 1) and matches their 50-day macro window (Figure 14), so our results will be directly comparable.

##### Gaps we could fill

1. **Proper scoring and FLB.** Brier and log scores by horizon and by price bucket, reliability diagrams, and a direct test of the longshot bias. We should run the test in the tails, where the $0.01 floor and stale last-trade prices mechanically inflate small probabilities. Separating that mechanical effect from a behavioral bias is a real contribution.
2. **A formal WIRP comparison.** One caveat: WIRP has only two states, so scoring it on multi-state outcomes penalizes it by construction. We need a common event definition, for example P(cut) versus P(hold or hike).
3. **SPF and options benchmarks.** They use the SME and Bloomberg consensus instead of SPF, because SPF is quarterly and asks about annual targets. Matching horizons with SPF will be awkward, which is likely why they skipped it. We could compare the CPI densities with option-implied inflation densities.
4. **Intraday price discovery.** Minute-level Kalshi trades around CPI, NFP and FOMC events against fed funds futures or SOFR futures tick data. None of this exists in the paper.
5. **Forecasting models (RQ3).** Their results show that such models have to beat the raw Kalshi mean on its own.

**Novelty verdict.** For point accuracy and PIT calibration, RQ1 is already largely done: their paper covers "does Kalshi beat surveys and futures". Scoring rules, the FLB and the tails remain novel. RQ2 at intraday frequency, and RQ3, are mostly untouched.

#### B. Caveats


**Price-to-probability conversion (Sec. 3).** Probabilities are *daily last-trade* prices, and a strike that did not trade keeps its price from an earlier day. So the cross-strike distribution on any day mixes prices from different times, and the tail strikes are the most stale. Fees are never mentioned. The $0.01/$0.99 bounds are handled by an ad hoc tie rule that assigns mass "closer to the mode" (fn. 3). The monotonization method, building outward from the mode, was chosen because it "improves the reliability" (p. 14), a choice made after seeing forecast accuracy. Bid-ask midpoints are rejected (Appendix A) only for the fed funds (FFR) market. *Acknowledged:* the Q-vs-P gap and stale tails. *Not addressed:* fees, mismatched price timing across strikes, and sensitivity to these rules.

**Liquidity is shown only for FFR** (Figs. 2-3). There is no volume, spread, or depth for CPI, unemployment, or GDP contracts, which carry the "unique distributions" claims. "Compare favorably to SOFR options" (p. 12) comes without data.

**FFR accuracy: degenerate test and internal inconsistency.** The "perfect forecast record" of the Kalshi mode (Table 3B: MAE 0.000** vs futures 0.010) is measured on FOMC day, when the decision is almost always fully priced. The authors say the gap comes from a single meeting, Sept 2024 (p. 24). With ~30 meetings and a zero loss differential at nearly all of them, a Diebold-Mariano (DM) statistic means little. The comparison also favors Kalshi by construction: a discrete mode on the target grid can score exactly zero, while a futures mean distorted by EFFR offsets and month-averaging cannot. Fig. 1 shows futures MAE near 0.05 at day 0, not 0.010, and Fig. A.1's "Trades" lines are much noisier than Fig. 1, which suggests undocumented smoothing. The Survey of Market Expectations (SME) comparison uses ±½-SD bars, not a test. Yet the Conclusion claims "statistically significant improvements over fed funds futures and professional forecasters" (p. 34), contradicting the Introduction's "very similar" (p. 3).

**Macro-release accuracy (Table 3A, Fig. 14).** Timing is not aligned. Kalshi is read just before the print, while the Bloomberg consensus is compiled days earlier. Of 12 comparisons, 3 are significant, all for headline CPI and none at 1%. There is no multiple-testing correction, and "never worse" reflects low power. Fig. 14 shows Bloomberg beating Kalshi at day 0 for core CPI (~0.062 vs ~0.07) and unemployment (~0.078 vs ~0.086), but Table 3 has core CPI tied (0.070) and unemployment at 0.109 vs 0.117. The number of releases is never reported.

**Calibration (Sec. 6.1, Fig. 15).** "Fairly well calibrated" understates the evidence. At 28 days, unemployment (K and C both 0.03) and headline CPI (K 0.03, C 0.04) reject uniformity at 5%. Outcomes are discrete (25bp steps, prints rounded to 0.1), so the plain probability integral transform (PIT) is not uniform even when the forecast is correct, and a randomized PIT is needed. This is not addressed. No proper scoring rule (Brier, log score, CRPS) appears anywhere: every accuracy comparison is MAE on point summaries. The density-forecast claim itself is never scored.

**Event studies (Sec. 7, Tables 4-5).** The regressions use close-to-close daily changes, despite the "high-frequency" framing. N = 113 for CPI and for PCE, far above the ~45 releases in the sample, so observations must be pooled across several FOMC contracts. Standard errors are HC3 and not clustered by release, so precision is overstated. In Table 5, the statement and press-conference shocks fall on the same day, so a daily Kalshi change cannot separate them. The press-conference skewness result (−2.875**) that the paper headlines is one of 12 coefficients. Higher moments of a ~5-7 bin distribution with a 1¢ floor are fragile. The "four times" CPI asymmetry (p. 29) is never tested.

**Interpretation.** Sections 4-5 rely on hand-picked episodes (e.g. April 2025 was "prescient"). The SOFR-Kalshi gap (Fig. 11) is attributed to the repo basis and hedging without a test. Fig. 13 shows far more severe-stagflation mass on Kalshi than in the SPF, attributed vaguely to "risk premiums or... retail investors" (p. 23). The 1¢ floor alone puts mass in the tails, which bears directly on our longshot hypothesis.

**Sample and replication.** The series are ones "deemed most economically relevant" (Table 1), drawn from a single 2022-25 policy regime with an unclear end date. Code and data are released only "subject to approval", so plan to rebuild everything from the API.
**Done well.** The authors are candid about Q vs P. They use formal tests (bootstrapped PIT tests, DM, HC3) and argue convincingly that futures and SOFR options are poor benchmarks for meeting-specific densities. For our WIRP comparison: Fig. 1 does plot futures MAE at every horizon (with a "slight improvement with about 60 days to go for Kalshi", p. 24), but that comparison is visual, uses MAE only, and has no bands for futures. The formal test (Table 3B) covers FOMC day only. A comparison of Kalshi vs WIRP over [−60, −1] using a proper scoring rule has therefore not been done yet.

---

### 2.2 Bürgi, Deng & Whelan (2026) — Makers and Takers: The Economics of the Kalshi Prediction Market

#### A. Relevance to our project

##### What the paper does

Bürgi, Deng and Whelan pull transaction-level data from the Kalshi API: 46,282 Yes contracts from 12,403 events, 2021 to April 2025, which gives 313,972 Yes and No prices. They take daily snapshots from the close back to 10 days before it. They test calibration with win-rate plots and Mincer-Zarnowitz regressions, and compute post-fee returns separately for Makers and Takers, using Kalshi's own flag for which side initiated each trade. A model of heterogeneous beliefs with makers and takers then shows that a small bias toward overweighting small probabilities (β ≈ 0.09) is needed to fit the favorite-longshot pattern.

##### Relevance to our research questions

**RQ1: Calibration and favorite-longshot bias (high relevance, but only partly).**
- *Direction of the bias.* Contracts priced low win less often than their price implies, and contracts priced high win more often (Fig. 3, p. 15). The 1-10c band loses about 57% after fees in Fig. 5 (p. 18); the text rounds this to "over 60%". Contracts above 70c earn small but significant positive returns. The average pre-fee return is about -20% (p. 18). This supports our hypothesis that Kalshi overprices tails, at least in the broad cross-section.
- *By category.* Table 8 (p. 25) rejects unbiasedness in every category. For **Economics** (N = 24,405 Yes observations), the slope ψ is 0.034 (SE 0.010). The constant is -0.978 (SE 0.972) and is not significant. Coefficients are in cents. Our back-of-envelope reading of that linear fit: a 5c Economics contract loses about 16% pre-fee, against about 31% for the full sample. So the bias in macro markets looks roughly half as severe, but the estimate is noisy. Politics (ψ = 0.022) and Entertainment (ψ = 0.020) are not individually significant. There is no split for FOMC, CPI or NFP.
- *By contract structure.* Bracketed macro markets fall under "Exclusive Numerical" in Table 4 (p. 21): ψ = 0.014, constant = -1.637. The fitted pre-fee profit is negative at every price up to 99c. *Our inference, not stated in the paper:* this suggests the Yes prices in bracket ladders sum to more than $1, an overround. Whatever the cause, bracket prices have to be normalized before comparing them with WIRP.
- *Makers vs takers.* After fees, Makers average -9.64% and Takers -31.46% (p. 27). In the 1-10c band (Fig. 6, p. 28), Takers lose about 72% and Makers about 35%. Makers paid **no fees** in this sample and still lose about a third of their money on longshots. So the bias is not just the fee wedge. Takers buy 56.5% of the 1-10c contracts (Table 10).
- *Fees and friction.* The fee is 0.07·P(1−P) per contract, rounded up to the cent on the whole order. As a share of the price this is 1.75% at 50c but about 6.7% at 5c. The fee therefore amplifies measured tail losses, but it does not create them.
- *Is the bias economically meaningful?* In return terms it is large. **In probability terms it is small.** The paper's own example is a 5c contract that wins 3% of the time: a 2 percentage-point miscalibration that shows up as a -40% return (p. 18). This matters for our design:
  - A Brier score barely penalizes a 2pp error at the tails. Cross-entropy penalizes it much more. The choice of metric will largely decide whether "Kalshi overprices tails" looks like a real forecasting flaw.
  - A 1c tick with a 1c floor mechanically caps how small a probability can be quoted. For FOMC states with a true probability below 1%, this truncation alone produces a longshot bias. The paper notes that prices are integer cents but never treats this as a source of bias.
- *Robustness.* The bias holds for every horizon from 0 to 10 days (Table 5), every volume quintile (Table 6) and every trade-size quintile (Table 7). It weakens in 2025, when ψ = 0.021 and is significant only at the 10% level (Table 9).
- *Accuracy over the horizon.* Mean absolute error (MAE) is about 17c at 10 days, about 11-12c at 1 day and about 6c at the close (Fig. 4, p. 16).
- *What the paper does not do.* It makes no comparison with SPF, options, fed funds futures or WIRP, and it does not compute Brier scores or log scores. The horizon stops at 10 days, while our window is [-60,-1]. Page and Clemen (2013), cited on p. 4, find a longshot bias on InTrade at long horizons that they attribute to discounting. We treat this as a hypothesis, not an expectation. Their mechanism relies on capital sitting idle without earning interest. Kalshi has paid interest on cash and open positions since 2024 (not from the paper; to verify against Kalshi's current terms), which would weaken that channel. The carry is also small in cents. At about 4% a year over 60 days it is about 0.66c on a $1 payoff, and about 0.03c on a 5c contract, so it is below the 1c tick at the tails. A horizon-dependence test (t = -60 vs t = -1) can tell a carry or discounting story apart from a belief-bias story.

**RQ2: Price discovery (low relevance).** The paper uses daily snapshots only, with no intraday or event-time analysis. The one related finding is the steep drop in MAE on the final day (Fig. 4), which fits information arriving at resolution. The authors hint at a "Yogi Berra" effect: losses on cheap contracts get worse on the closing day for both sides. The paper does point to Swanson, Wang and Wu (2025) on Fed announcements and Kalshi macro-release markets as the relevant prior work, which we should read.

**RQ3: Predicting macro quantities (essentially none).** There is nothing on forecasting realized macro data. The one transferable idea is a debiasing map. The model's mean belief is μ = π* + β(0.5 − π*), which inverts to π* = (p − 0.5β)/(1 − β). One could apply it to prices before using them as features. The caveat is that β describes beliefs in the model, not a direct mapping from prices, so it would need re-estimating on macro contracts.

##### What we can borrow

- **Calibration regression:** regress Y − P on P, test α = ψ = 0 with an F-test, and run it on Yes contracts only, since No prices are a linear transformation and would double the sample. Cluster standard errors by event and by contract.
- Add **binned calibration and return plots** in 10c bands with 95% confidence intervals. The linear test cannot capture the tail nonlinearity, which is concentrated in the 1-10c band.
- **Return formula:** R = (Y − P − C)/(P + C), with the fee C imputed on a 100-contract order and rounded up.
- **Data filters:**
  - final volume of at least $1,000
  - final bid-ask spread of no more than 20c
  - no hourly markets
  - daily snapshots taken at the same time of day as the final trade
  - a check of the transaction data against Kalshi's reported final prices (63 contracts dropped)
- The API's taker-side flag lets us split results by who initiated each trade.

##### Gaps we could fill

1. **A macro-specific tail test.** Calibration by price bin for FOMC, CPI and NFP contracts, on bracket prices normalized to sum to one, using both Brier and log scores. This would tell us whether the tail bias matters for forecasting or is mostly a return-space artifact of the tick size.
2. **Benchmarking against other forecasts.** Compare Kalshi with WIRP, SPF and option-implied probabilities over [-60,-1]. The paper has no external benchmark at all.
3. **Horizon.** Extend the horizon to 60 days and run the horizon-dependence test (t = -60 vs t = -1) to separate carry or discounting from belief bias.
4. **The post-April-2025 fee regime.** Makers now pay fees too, so the maker-taker results may not hold in our sample. The 2025 weakening in Table 9 suggests the bias may be shrinking.
5. **Intraday price discovery.** Event studies around announcements, which the paper does not attempt.

#### B. Caveats


*Page numbers are the printed page numbers in the January 2026 PDF.*

**The evidence on macro contracts is thin and does not match the headline.** "Economics" supplies only 24,405 of the 156,986 Yes prices, and the paper never splits it into CPI, payrolls or FOMC (Table 8, p.24). Its constant is −0.978 (s.e. 0.972) and its slope 0.034 (0.010), so the fitted line crosses zero near 29c. That implies a 5c contract is mispriced by under 1c while a 95c contract is about 2.3c too cheap. In other words, the rejection in macro markets comes mostly from favorites being underpriced, not from overpriced longshots. The paper gives no return-by-bin chart and no Maker/Taker split for any category. Politics (0.022, s.e. 0.021) and Entertainment (0.020, s.e. 0.012) have slopes that are not significant, so the claim that the bias holds "across a wide range of different categories" (p.3) overstates the table. The authors say the results survive dropping sports and cutting the sample at December 2024 (p.9), but neither check is reported. The 2025 sample, mostly sports, shows a weaker slope (0.021*, Table 9).

**Much of the bias may come from the closing day and the 1c floor.** Two-thirds of observations are priced below 10c or above 90c (Table 2). Final trades are often placed after the outcome is effectively known, as in same-day temperature markets. MAE falls from about 11c one day out to about 6c at close (Fig. 4, p.16), and Takers in the cheapest bin lose about 90% on closing-day prices (Fig. 7, p.29). Because prices are whole cents, a contract with a true probability of 0.2% cannot trade below 1c, so it looks overpriced by construction. The model assumes integer pricing away (p.31), and the empirical work never checks this floor. The 1–10c bin, which holds 33.8% of the data, is likely dominated by 1–2c contracts. Before reading this as evidence about beliefs, a replication should drop closing-day prices and the 1–2c/98–99c extremes and re-estimate.

**The weighting and sample filters differ from what the headlines imply.** Each observation is the *last trade* of a contract-day, not every trade and not volume-weighted. The "−20% average return" (pp.3, 17) is an equal-weighted mean across contracts. The authors admit the dollar-weighted pre-fee return is zero by construction, so the −20% mostly shows how numerous cheap contracts are. The Maker −9.64% vs Taker −31.46% gap (p.27) is partly composition too, since Takers hold 56.5% of 1–10c contracts (Table 10). The filters are set on information known only at close: final volume of at least $1,000 and a final spread of 20c or less (p.9). The resulting sample is not one a trader could form in real time, and dropping thin longshot legs means event probabilities need not sum to one. The bin edges are also inconsistent: Table 2 uses "41c-50c" and "50c-59c", while the text has "11c-19".

**Inference is careful in the regressions but not in the figures.** The Mincer-Zarnowitz regressions use Yes contracts only, which avoids doubling the sample with mirror-image No trades, and they cluster by both event and contract (p.19). That is done well. The cluster counts are never reported, though, which matters for Economics, where the number of events is likely small. The confidence intervals in Figs. 3, 5 and 6 are described as "standard" (p.14). They appear to treat up to 11 same-outcome daily observations of a contract as independent, so they are probably too narrow. Nothing accounts for common shocks such as many city-temperature markets on the same day or linked macro releases. No proper scoring rule is used either. MAE rewards the median rather than calibrated probabilities, and it falls toward close partly because prices mechanically converge to 0 or 1. It is not a benchmark for our Brier and log-loss comparisons.

**Fees, prices and roles are approximated.** The Taker fee is imputed on a hypothetical 100-contract order (p.17). The median trade is $35, about 700 contracts at 5c, and rounding matters most at low prices, so post-fee returns in the low bins are the least reliable. A single γ = 0.07 is applied throughout. The paper does not discuss whether some series or programmes had different fee terms. Last trades happen at the bid or the ask, with spreads of up to 20c allowed, and the paper never compares them with midpoints. The API's taker-side flag is genuinely better than Lee-Ready (p.26). However, "Maker" labels a role in one trade, not a type of trader, and Maker returns are measured only on orders that were filled, which hides adverse selection on stale quotes.

**The model's claims go further than it shows.** Three parameters are fitted to 20 bin means, with θ = 0.6 simply assumed and σ trading off against θ (correlation 0.80), and there is no formal test of fit (pp.35–37). β is "tightly identified" (0.06–0.12) only given normal beliefs, a constant σ and a 0.5 anchor. Calling behavioral bias "necessary" (p.41) rests on comparison with a β = 0 version of the *same* model. The paper does not test tick-size effects, a taste for skewness or informed trading as alternatives. The argument about why the bias is not competed away (p.40) is speculative. The +2.6% Maker return uses only filled orders and ignores price impact. Makers have also paid fees since April 2025 (p.6), which the authors acknowledge as the reason for their sample cutoff.

---

### 2.3 Bartlett & O'Hara (2026) — Adverse Selection in Prediction Markets: Evidence from Kalshi

#### A. Relevance to our project

##### What the paper does

Bartlett and O'Hara use Kalshi's full trade history (July 2021 to March 11, 2026; 41.6M trades on 478,167 markets) to measure adverse selection. They compare "single-name" markets (earnings-call mentions, celebrity attendance, company KPIs) with "broad-based" markets (Fed decisions, CPI, jobs, GDP, equity indices, crypto). Single-name markets show more informed price impact: log-odds Kyle's λ, t = 10.94, Table 2. Yet makers earn more there (1.91¢ vs 0.81¢ per contract, Table 5), because optimistic YES takers overpay on markets that mostly settle NO. The paper's contribution is a microstructure one. It does not evaluate forecasts, and it runs no event studies around announcements.

##### Relevance to our research questions

**RQ1 (calibration, favorite-longshot bias, WIRP vs Kalshi).** This is the paper's most useful input for us, and it cuts against our hypothesis. Their calibration estimator (Figure 6; Figure B.1) uses the volume-weighted YES price over the first 80% of each market's life, with each market weighted equally and SEs clustered by market. On it, broad-based markets are "essentially calibrated": the per-market gap is +0.2pp (SE 0.11, 114,808 markets). There is only a "mild signature" of the textbook longshot pattern (pp. 35-36). The large miscalibration (+4.5pp, and up to 8.7pp in the 40-cent bucket) is a YES-optimism across the whole price range in single-name markets, not overpriced tails. Two caveats keep this from refuting our hypothesis:

1. The broad-based pool is dominated by crypto (303k markets, 2.56B contracts) and equity-index ranges (86k markets). Fed funds has only 1,198 markets (658M contracts), CPI 2,392 and GDP 295 (Table A.2). Results are never broken out by macro subcategory, so the +0.2pp tells us little about FOMC or CPI markets specifically.
2. Their window averages over a market's life. It does not score at fixed horizons like our t ∈ [-60, -1], and it uses no Brier or log score.

Net: the paper is weak evidence against "Kalshi overprices tails" for macro contracts, and it leaves the macro-specific, horizon-specific test open. Their adverse-selection story also gives a competing explanation for any tail mispricing we find. The spread cap min(P, 100 − P) and loss of max(P, 100 − P) (pp. 30-31) mean makers are most exposed at extreme prices. That fits Shin-style shading, not just behavioral bias. One fact matters for our WIRP comparison: takers buy YES 56.3% of the time (volume-weighted) in broad-based markets, and 60.9% of these markets settle NO (Table 1). A YES tilt is plausible even here, so we should check both directions of mispricing.

**RQ2 (price discovery around CPI/NFP/FOMC).** The paper is indirectly relevant. Their evidence that Kalshi flow is informative and permanent fits what we need to assume. Glosten-Harris transitory components are near zero and significant in only 7-8% of markets. Correct-side λ exceeds wrong-side λ even in broad-based markets (t = 18.46, Table B.3 Panel C). But they find that flow direction does not predict settlement in broad-based markets (t = 0.06): one-sided flow follows a moving public underlying. That is exactly the setting of a macro release. Their weather appendix (B.4) is the closest template to our design. It matches Kalshi signed flow and prices to an external high-frequency public signal (station temperature) in an hourly panel. Flow and price move with the signal at the same time (t = 7.07 and 9.37) with no lag. Swap temperature for fed funds futures and this is RQ2.

A warning they don't make explicitly: a CPI or NFP bracket market settles on the release itself. The "post-announcement jump" in that contract is settlement, not discovery. They show that approaching settlement makes flow mechanically one-sided and inflates impact measures. Their fix is to restrict to 30-70¢ interior prices or the first 80% of a market's life. Our event study should look at *other* contracts that the release informs: the next FOMC market after a CPI print, or the rate path after Fed speeches and minutes.

**RQ3 (predicting macro quantities).** The paper says essentially nothing here. At most it suggests that raw broad-based prices need little bias correction. Signed taker flow and trailing VPIN are candidate features, but VPIN did *not* predict outcomes in broad-based markets (OLS t = 0.52, Table 10), so we shouldn't expect order-flow features to add much over price.

##### What we can borrow

- **Trade signing is free.** Every Kalshi trade record carries `taker_side` (YES or NO), plus the YES price in cents and a contract count. No Lee-Ready is needed. Maker P&L per trade is (V − P) × m, and it needs no account IDs (Kalshi exposes none; eq. 6).
- **Log-odds price changes**, Δlog(p/(1−p)), for any impact or jump regression near the bounds (eq. 1). We should also use log-odds changes when comparing Kalshi moves with fed funds futures moves.
- **Kyle's λ on 5-minute intervals** of signed order flow, with a minimum of 50 intervals per market, and the directional (correct-side vs wrong-side) λ split on the regressor, not the price change (B.3).
- **Controls for settlement convergence:** restrict to 30-70¢ interior prices, or to the first 80% of life, defined as the first trade until cumulative volume reaches 99%.
- **Calibration estimator** with equal market weights and SEs clustered by market. Each market's outcome is one draw, which matters for our Brier SEs because the strikes within one FOMC event share a single outcome.
- **Data realities:**
  - Trades come via the REST API with timestamps fine enough for their 30-second VWAPs (Figure 4).
  - Quotes exist only as *hourly candlesticks*. Valid bid/ask covers about 7% of trades overall and just 5.2% of broad-based trades. Kalshi does not keep candlestick history for many short-lived markets.
  - There is no historical order book in their data. Hourly midpoints are too stale for Huang-Stoll at 5-30 minutes: 91% of trades have the same midpoint 5 minutes later.
  - For intraday event windows, build prices from trades, not quotes.
  - Fees are ignored and vary over time (0.07-1.75¢ taker).

##### Gaps we could fill

1. **Break out macro subcategories.** Give calibration and Brier/log scores for Fed, CPI, NFP and GDP separately, at fixed horizons before the event, against WIRP and SPF. The paper pools these with crypto.
2. **An event-study version of their weather test.** Relate Kalshi signed flow and log-odds jumps to high-frequency fed funds futures surprises around CPI, NFP and FOMC, with explicit leads and lags. They have no announcement windows at all.
3. **Test whether macro tail mispricing, if present, is behavioral or adverse selection.** Do tail contracts show higher correct-side λ, or just higher wrong-side λ, as weather does?
4. **Pre-announcement informed flow.** FOMC leaks or the drift in the rate path before an FOMC meeting are a real informed-trading question in "broad-based" markets that they assume away.

#### B. Caveats


**1. "Broad-based" is mostly crypto and equity-index brackets, not macro.** Per Table A.2 (p. 71), crypto (303,336 markets; 2.56B contracts), equity indices (86,516) and FX (61,940) make up about 98% of the 462,651 broad-based markets and about 78% of broad-based volume. Fed, CPI, jobs, GDP, PCE and recession markets together are only about 5,750 markets (about 1.2% of the count, about 21% of volume). Every broad-based result is pooled, and the paper never reports a macro-only split. That covers near-perfect calibration (gap +0.2pp, Fig. 6), low λ, 0.81¢ maker P&L, and the VPIN null. The authors explain the VPIN null with a Bitcoin example ("continuously moving underlying", p. 46), which does not describe a scheduled CPI or FOMC release. The line "absence of concentrated private information in macro and index markets" (p. 48) goes further than the evidence. The authors do not address this. Also note that "press conference mentions" (1,441 markets, Table A.1) and Trump speech markets count as *single-name*, so Fed-communication-style contracts sit on the other side of the split.

**2. The standard errors ignore that markets within an event are dependent.** Cross-sectional regressions use market-level HC1 (Tables 2–4, B.2–B.3). Calibration and VPIN errors are clustered by market "since each market's settlement outcome is a single draw" (p. 35). That reasoning fails for bracket ladders: in a mutually exclusive strike ladder, exactly one market settles YES. Mention markets from one earnings call also share one realized transcript. The single-name sample is 15,516 markets but only 351 series, and the independent events number far fewer than the 114,808 "markets" in the broad-based calibration. The authors do not cluster by event or series, so the t-statistics are likely overstated. The headline flow-direction statistic (t = 12.25, abstract and p. 46) appears to be bucket-level. The market-clustered logit in fn. 35 gives z = 2.04, p = 0.04, which is much weaker and still not event-clustered.

**3. The key test does not condition on price, so "flow forecasts settlement beyond the price" is untested.** Section 3.2.1 says only this test identifies private information. The Q5 comparison (35.9% vs. 22.2% YES) sorts on flow sign but does not control for the prevailing price within the 30–70¢ band. YES-directed buckets probably trade at higher prices. The appendix correct-side λ is positive in *every* category, weather included (Table B.3, Panel C), so the λ-based measures do not discriminate on their own. The authors acknowledge this and show weather as a false positive (B.4).

**4. Several claims have no inference behind them, and single-name profit is concentrated.** Table 5 reports no standard error for "makers earn twice as much" (1.91¢ vs. 0.81¢). Median P&L per market is $4.50 vs. −$0.01. The Wahlberg market alone earned makers $1.18M (p. 22), about 26% of all single-name maker P&L ($4.60M). The paper has no leave-one-event-out or series-concentration check. Equation (7) is an accounting identity against an arbitrary 50/50 benchmark, not independent evidence. The "anti-informedness gap" (Fig. 7) ignores price levels. Calibrated takers who buy cheap YES longshots would also produce a positive gap. Figure 7 has no confidence bands.

**5. Fees are ignored.** Fn. 22 (p. 32) omits taker fees (0.07–1.75¢) and, from April 2025, maker fees (0.02–0.44¢). The authors claim the comparison across market types is unaffected "unless fee structures differ systematically", yet they say fees vary "by market type". The 0.81¢ broad-based margin is the same order as the fees. Rebates are called "negligible" without numbers (fn. 4). This is acknowledged but not resolved.

**6. Quote data are thin and stale, so the spread results are fragile.** Midpoints come from hourly candlestick opens and cover about 7% of trades: 35.5% for single-name but only 5.2% for broad-based, because Kalshi does not keep candles for most short-lived markets (fn. 14). Huang–Stoll is infeasible (p. 24). The single-name spread premium flips sign with the staleness window: +0.74 at τ ≤ 5 min, −0.16 overall (Table 4). The authors acknowledge this. λ and GH use only the 14,809 markets (about 3%) with at least 50 five-minute intervals.

**7. The comparison is not random, and some choices were tuned.** Single-name and broad-based markets differ in duration, late-volume clustering (40% vs. 15% in the final tenth, p. 16), clientele, and contract-wording risk. The Netflix example is literacy in resolution rules, not private information. The controls are only volume, price, price², and √p(1−p) (R² ≈ 0.04), with no series fixed effects. The $500 bucket and 30–70¢ band were picked for "discriminating power", and the bucket-size sweep is not shown (fns. 31, 38). Most of the single-name effect comes after late 2024 (Fig. 7), which is thin support for calling the equilibrium "durable". Evidence on maker holding comes from *unreported* Polymarket results (pp. 17, 33), because Kalshi has no account IDs.

**What they do well / replication notes.** Everything comes from the public Kalshi REST API: trades with `taker_side`, settlement results, and candlesticks. We can therefore replicate it without proprietary maker/taker or account data. The limits are the same ones they face: no account IDs and no historical order book. The robustness checks against settlement convergence (first 80% of life, interior prices) are sensible. For our RQ1, their only broad-based evidence shows a *mild* favorite–longshot bias. That weakens our tail-overpricing hypothesis, but it does not test it for Fed or CPI contracts. We should rebuild calibration on macro-only contracts at fixed horizons, cluster by event, and handle fees explicitly.

---

### 2.4 Wolfers & Zitzewitz (2004) — Prediction Markets (JEP 18(2))

#### A. Relevance to our project

##### What the paper does

Wolfers and Zitzewitz (JEP 18(2), 2004, pp. 107–126) survey the early evidence on prediction markets. They classify contract designs by the statistic each one reveals, review accuracy against polls, experts and surveys, and describe known departures from efficiency: favorite–longshot bias, bubbles and manipulation. They close with practical advice on market design and on using prices in later analysis. The paper is conceptual and descriptive. Its only formal macro evidence is a 16-observation table.

##### Relevance to our research questions

**Contract types (framing for all three RQs).** Table 1 (p. 110; text pp. 109–110) sets out three designs. A winner-take-all contract prices a probability. An index contract prices the mean. A spread bet at even money prices the median. Kalshi's binary and bracket contracts are winner-take-all. A ladder of brackets is the paper's "family of winner-take-all contracts," which "reveal[s] almost the entire probability distribution" (p. 109). From that ladder we get the mean, the median and the tail mass by construction. We do not observe a mean directly, because Kalshi lists no index contracts. For "above X" strike ladders, the strike priced at 50 is the counterpart of the spread-bet median, and the strike priced at 80 marks a quantile (p. 109). Footnote 3 (p. 109) warns that these prices are *risk-neutral* probabilities (state prices). This matters for RQ1: WIRP is also built from fed funds futures prices, so any Kalshi-vs-WIRP gap may partly be a difference in risk premia rather than in beliefs.

**RQ1 (calibration, favorite–longshot bias).** The closest precursor is the Economic Derivatives analysis (Goldman Sachs/Deutsche Bank; pp. 114–115, Table 3). It covers nonfarm payrolls (NFP), retail sales ex-autos and ISM manufacturing. The market-implied means correlate 0.91–0.95 with the Briefing.com consensus of about 50 forecasters. The authors find "no statistically (or economically) meaningful differences" in accuracy: mean absolute error (MAE) is 72.2 vs 71.1 for NFP and 1.07 vs 1.10 for ISM. The samples are tiny, n = 11–16 (p. 114). A likely misreading needs flagging. The "Market (implied expectation)" row (65.7 for NFP) is the market's *own predicted* forecast error, a measure of uncertainty. It is not better accuracy. The authors use it to argue that the market's uncertainty is about right, and that it may be overconfident relative to Census and BLS sampling error (p. 115, fn. 4). On favorite–longshot bias, pp. 117–118 and Table 4 show TradeSports S&P 500 range contracts overpricing extreme outcomes relative to CME option-implied prices. For "under 600," TradeSports quotes 5–8 against about 1 from options, and a small arbitrage gap persisted. The authors say the bias is "more pronounced on smaller-scale exchanges" (p. 118). This supports our tail-overpricing hypothesis. Footnote 5 (p. 117) cuts the other way: the options benchmark may itself be miscalibrated (the volatility smile).

**RQ2 (price discovery).** The evidence here is anecdotal. After the Poindexter news, the price moved from 40 to about 80 "within minutes," hours before the story spread widely (pp. 107–108). Figure 1 (p. 112) shows forecast error shrinking as the election nears. That is a horizon profile, not an announcement event study. The most relevant sentence for us is on p. 116: the Saddam Security "responded to news about Iraq with a slight lag relative to deeper financial markets." That is exactly the Kalshi-vs-fed-funds-futures lead–lag question, and no estimate is given. The Economic Derivatives markets were pari-mutuel auctions (p. 120), so they yield no continuous path to study around a release.

**RQ3 (forecasting macro quantities).** The paper says little here. "Making Inferences" (p. 122) treats prices as regressors, for example the war risk-to-oil price link from the Saddam Security. It covers conditional contracts (pp. 122–124) and warns about causality and selection. It says nothing about machine learning, about combining market prices with other predictors, or about forecasting realized macro series beyond the Table 3 comparison.

##### What we can borrow

- A clear map from contract design to statistic: bracket ladder → distribution → mean, median and tail probabilities (pp. 109–110).
- The Table 3 layout: correlation with the consensus, correlation with actuals, MAE, and the market's implied uncertainty compared with its realized error. This is a ready template for Kalshi vs SPF and the consensus.
- The Table 4 design: bracket prices compared with option-implied bracket prices on the same date, to test tail pricing (p. 118).
- Conditions for good performance (p. 121): widely discussed events with ambiguous public information work well. Markets where insiders hold concentrated information fail, as with the Supreme Court and papacy contracts. Markets also fail when public information is misleading (pp. 121–122). FOMC decisions are heavily signalled by the Fed, which makes them a borderline case under these criteria.
- Caveats about thin markets, for manipulation (p. 119) and for the failed inflation futures market (p. 121, fn. 7). These justify liquidity filters.

##### Gaps we could fill

- Table 3 rests on 11–16 releases. Kalshi gives roughly 40–60 releases per series since 2021–23 (each with a full strike ladder), so the comparison is better powered, though still far from large; effective N is the number of releases, not strikes or days.
- No proper scoring rules are used; the paper reports only MAE and correlations. Brier and log scores on the full distribution are new here.
- The bias evidence is one snapshot of the S&P 500 on 23 July 2003. We can test for bias systematically across events and horizons t ∈ [−60, −1].
- The "slight lag" claim (p. 116) has never been quantified with high-frequency data. RQ2 does that directly.
- There is no evidence on forecasting with machine learning. RQ3 goes beyond anything this paper covers.

#### B. Caveats


The authors call their own synthesis "a rough and fairly optimistic description of what we have learned from early experiments" (p. 108). We should read it that way: it is a 2004 survey built mostly from anecdotes, single snapshots, and small samples, and it was written before any CFTC-regulated retail event-contract exchange existed.

**Prices as probabilities (hedged, but the hedge matters most for us).** Footnote 3 (p. 109) says the price is a probability only "assuming risk neutrality." The authors defend this by noting that stakes are small, then concede that "if the event in question is correlated with investors' marginal utility of wealth, then probabilities and state prices can differ," and set the issue aside. For elections this is a footnote. For our contracts it is not: CPI surprises, payroll misses, and Fed cuts all co-move with marginal utility, so Kalshi prices on these events are risk-neutral probabilities in the same sense as WIRP or option-implied densities. That actually helps us compare Kalshi with WIRP and options like for like, but it means neither side is a physical-probability benchmark. The authors also flag the aggregation problem: the market's median is not the median participant's belief (p. 109). This paper does not cite their later work, Wolfers and Zitzewitz (2006, NBER WP 12200), which formalizes the point: price equals mean belief only under specific utility and wealth assumptions, and the gap grows when beliefs are dispersed and when prices sit near 0 or 1.

**Markets beat polls and experts (partly hedged, thin evidence).** Figure 1 compares the IEM with Gallup over four elections (1.5 vs. 2.1 points of error, p. 112). The authors themselves say this "may not provide a particularly compelling comparison" (p. 112). The HP and Siemens results come from markets with "20–60 people" and subsidized trading (pp. 113–114). The Saddam Security only "moved in lockstep" with expert opinion (p. 113), which shows correlation with experts, not better accuracy than them.

**Economic Derivatives (hedged, and weaker than the text implies).** This is the most relevant evidence for our project, and it rests on n = 16, 12, and 11 releases (Table 3, p. 115). The authors concede "precise conclusions are difficult to draw" (p. 114). The market is highly correlated with consensus (0.91–0.95), and for payrolls Corr(Market, Actual) is only 0.22. The claim that the market's implied uncertainty is "of about the right magnitude" (p. 115) is generous. Implied mean absolute error (MAE) is 0.34 against a realized 0.46 for retail sales, which suggests overconfidence, and 1.58 against 1.07 for ISM, which suggests underconfidence. Structurally, these were pari-mutuel auctions run by Goldman Sachs and Deutsche Bank shortly before each release (pp. 114, 120), with institutional money in the "hundreds of millions" (Table 2). Kalshi runs a continuous limit order book that is retail-heavy and trades for weeks before a release. The follow-up with a longer sample is Gürkaynak and Wolfers (2005/2006), and we should cite it rather than Table 3.

**Favorite-longshot bias (acknowledged, supports our hypothesis).** The evidence is a single-day snapshot (July 23, 2003) of Tradesports S&P 500 ranges with 3–5 point bid-ask spreads (Table 4, p. 118). Footnote 5 concedes that option-smile miscalibration is "less clear cut." Still, "the long-shot bias being more pronounced on smaller-scale exchanges" (p. 118) is a direct prior for Kalshi's tail buckets.

**Arbitrage, bubbles, manipulation (largely unhedged, and dated).** The claim that bubbles are limited because there are "no restrictions on short selling" and informed traders are not "capital constrained" (p. 118) sits uneasily with the authors' own finding that the S&P mispricing "persisted for most of summer 2003" (p. 118). On Kalshi, contracts are fully collateralized, fees fall hardest on low-priced contracts, and capital is locked up until resolution, all of which are real limits to arbitrage. The manipulation-resistance claim ("the profit motive has usually proven sufficient," p. 119) rests on anecdotes: candidates betting on themselves, Strumpf's random $500 IEM bets, and Camerer's racetrack cancellations. The authors hedge that it "depends ... on how thin the markets are" (p. 119), and footnote 8 (p. 124) warns that manipulation incentives rise once prices are used for decisions. That is exactly what happens when Kalshi prices get quoted as macro data. The "marginal trader is rational" argument (pp. 108, 118) is asserted, not tested.

**What does not transfer.** None of the venues studied was a CFTC-designated contract market with retail access, maker-taker fees, or on-chain settlement. Their participants were academics (IEM caps positions at $500), offshore bettors, or bank clients. The finding that insider-dominated topics "generated very little trade" (p. 121) and the adverse-selection logic behind it are only raised in passing; Bartlett and O'Hara (2026) address them directly.

---

### 2.5 Kelly, Kuznetsov, Malamud & Xu (2026) — Artificial Intelligence Asset Pricing Models

#### A. Relevance to our project

##### What the paper does

KKMX put a transformer inside the stochastic discount factor. Each stock's portfolio weight depends on its own characteristics and also, through attention, on the characteristics of every other stock that month (Sections 2-3). On monthly US stocks from 1968 to 2022 with 132 JKP characteristics, the nonlinear transformer reaches an out-of-sample Sharpe ratio of 4.57, against 4.31 for an equally deep own-asset MLP and 3.60 for linear BSV (Table 1, p. 23).

**Architecture in implementable terms.** Each month is one sample. The input is an N_t x D matrix X_t: stocks are the "tokens" and the columns are characteristics rank-standardized to [-0.5, 0.5] (p. 18). The target is next-month returns R_{t+1}. There is no positional encoding and no attention over time, so the model is a permutation-equivariant function of the cross-section.
- **Linear version (eq. 2, p. 7):** w_t = (X_t W X_t') X_t λ. It is linear in D^3 triple interactions (eq. 6), so ridge gives a closed-form estimate (eq. 12, p. 12).
- **Nonlinear version (eqs. 13-20, Fig. 1):** K blocks of softmax attention across stocks, σ(Y W_h Y') Y V_h, each followed by a residual, a per-stock ReLU feed-forward layer (d_f = 256) and another residual. A final λ maps the output to weights. They use H = 1 and K ≤ 10, about 1M parameters.
- **Loss:** MSRR, E[(1 - w'R)^2] plus a penalty (eq. 9, p. 11). This is a pricing objective, not a forecast loss. Appendix B.2 (p. 61) lists an MSE variant, which is the prediction-style analogue.
- **Training:** 60-month rolling windows, Adam, results averaged over 10 random seeds (pp. 21-22).

##### Relevance to our research questions

**RQ3 (primary, but indirect).** The paper does no macro forecasting and no time-series attention; it is not an "economic transformer" for sequences. What it offers is a template for predicting across a set of related contracts, where each element borrows information from the others.

**RQ1 (partly relevant).** The paper's no-attention benchmarks (DKKM/MLP, eq. 24, p. 17) apply a nonlinear map to each asset's own features. On Kalshi, that is a per-strike recalibration, i.e. a favorite-longshot correction. Adding attention tests whether a strike's miscalibration depends on the rest of the ladder, for example whether tails are overpriced more when the distribution is wide.

**RQ2 (not relevant).** The paper has nothing on high-frequency price discovery.

##### What we can borrow

1. **Strikes as tokens.** For release r at snapshot t, build X_{r,t} with one row per strike. Features: strike as a z-score against consensus, mid price, spread, volume and open interest, days to release, and 1- and 5-day price changes. Related markets (Fed decision, core CPI, fed funds futures path) can enter as extra tokens (fn. 6, p. 9).
2. **Output and loss.** Replace λ with one logit per outcome bin and apply a softmax across bins to get a density. Train on the log score of the realized bin, which matches RQ1's cross-entropy metric; use CRPS as a check. For a point forecast, score the density mean by MSE against the print.
3. **Linear attention before softmax.** With D ≈ 5-8 features, the saturated linear model has about 125-500 parameters and a closed-form ridge fit. A deep softmax transformer is not estimable on our data.
4. **Standardization within each cross-section** (p. 18). This is what allows pooling across releases and series (CPI, core CPI, NFP, unemployment, jobless claims).
5. **The benchmarking ladder** (Section 4.4.5): no attention, then own-strike nonlinearity, then attention. Beating the own-strike baseline is the honest test.

##### Gaps we could fill / how to adapt

**Sample size is the binding constraint.** KKMX have about 60 months x thousands of stocks per training window. Kalshi CPI and NFP markets offer roughly 40-60 resolved releases per series. Snapshots within a release (t in [-60, -1]) are autocorrelated and share one outcome, so they add little independent information. At this scale the paper's evidence that bigger models perform better (Figs. 4-5) does not transfer, and the paper itself warns about "limits to learning" (p. 30).

**The headline model should be low-dimensional.** With about 50 releases per series, the main RQ3 specification should be a ridge or OLS regression, fitted on an expanding window, of the realized print on four inputs: the Kalshi-implied mean, variance and skew, plus consensus. Linear attention over strikes is an extension or robustness check, not the main model. If it does not beat that regression out of sample, it should not be reported as the main result.

**The most novel target is one Kalshi does not trade at that horizon.** Forecasting CPI from the CPI ladder is almost certainly dominated by the raw Kalshi price. A more interesting exercise is using the FOMC, CPI and jobless-claims ladders to nowcast GDP or unemployment before those quantities' own markets open. This is the natural home for cross-market attention: the tokens are contracts from different series, and the raw price of the target market is not yet available to compete against.

**Baselines to beat.** All should be out of sample, with standard errors clustered by release and Diebold-Mariano tests.
- The normalized, monotone Kalshi-implied density. This is the key baseline for same-series targets.
- Isotonic or Platt recalibration of each strike's price.
- Consensus with a Gaussian spread fitted to historical consensus errors.
- The Cleveland Fed nowcast for CPI.

SPF is quarterly, so it is a weak benchmark for monthly prints; use it only for GDP.

**Not transferable.** The Sharpe and HJD metrics, the SDF interpretation and the observable-linkages analysis (Section 4.5.4) do not carry over. The paper also offers no guidance on attention over time, which needs a different reference.

#### B. Caveats


**Bottom line for RQ3.** Very little of the empirical machinery carries over to forecasting macro releases from Kalshi prices. What survives is one idea: pool information across related contracts through a learned similarity (attention) weight. The paper's gains depend on a huge cross-section, a long monthly panel, and a portfolio objective. A Kalshi macro dataset has none of these.

**1. Data scale.** The evidence uses US stocks from 1963 to 2022, with 132 JKP characteristics per stock-month and thousands of stocks each month (Section 4.1, p. 18). Models are retrained on 60-month rolling windows, giving 659 out-of-sample months (Section 4.3, p. 21). Attention runs across the cross-section of stocks, so the effective sample is stock-months, not months. Kalshi macro markets go back only to about 2021. That is roughly 50-60 CPI or payrolls releases and about 40 FOMC meetings, so one 60-month training window would use up the whole history. There are also only tens of contemporaneous contracts and no analogue of 132 standardized characteristics.

**2. Wrong objective.** Training minimizes the maximum-Sharpe-ratio loss (1 - w'R)^2 (Eq. 9, p. 11). Evaluation uses the SDF's Sharpe ratio and HJ-distance pricing errors on 132 factors (Section 4.2, p. 19). Forecasting a scalar macro release is a point or density problem, scored by RMSE, Brier score or log score. Algorithm 2 (p. 61) allows an MSE loss, but no MSE results are reported.

**3. Fragile "virtue of complexity" evidence.** Sharpe ratio rises from about 3.8 to 4.6 as depth goes from 1 to 10 blocks. That is a single seed-averaged curve with no confidence band (Figure 5, p. 30). The claim that million-parameter MLP and DKKM models do not catch up is asserted in the text but not tabulated (p. 29). Gains from linear attention level off after about 20 heads (Figure 4). The MLP keeps a significant alpha against the transformer (1.2%, t = 4.9; Table 2), which the authors attribute to "limits to learning" (pp. 29-30). With about 50 observations, nothing in the paper supports benign overfitting.

**4. Look-ahead and tuning.** The headline results use characteristics that were not yet public. With point-in-time data, the transformer's Sharpe ratio falls from 4.57 to 3.79, and its edge over the MLP nearly disappears (3.79 vs 3.68; Table 11, p. 46). The set of characteristics is chosen by full-sample missingness (p. 18). The BSV benchmark's ridge penalty is chosen on the full sample (p. 22). No validation protocol is described for the transformer's learning rate, epochs, d_f or H (fn. 11). A short Kalshi history makes both risks worse: tuning on the test period, and using features that did not exist at forecast time.

**5. No implementability test.** The paper reports no transaction costs, turnover, shorting constraints or capacity, so Sharpe ratios above 4 are not attainable returns. Performance rises steadily toward micro caps: the transformer's Sharpe ratio is 1.84 among mega caps and 4.42 among micro caps (Table 7, p. 37). It also decays after 2002 (Table 1 Panel B), with 2017-2022 weakest (fn. 21). Kalshi spreads and thin books on tail strikes would widen this gap, not narrow it.

**6. Opaque attention; heavy compute.** Observable linkages explain 18% of the variation in attention, led by shared size and liquidity groups (Table 9, p. 41). Training needed a national supercomputing grant, and the authors stopped at 10 blocks because of cost (pp. 1, 30). Two of the four authors have AQR ties: Kelly is at AQR and Malamud is a consultant to AQR (p. 1).

**What transfers and what breaks.** What plausibly transfers is a heavily ridge-regularized linear-attention layer, which has a closed form (Eq. 12, p. 12), used to pool signals across related contracts. For example, a CPI forecast could draw on the Fed, payrolls and yield markets. It should be scored with MSE or log score in an expanding window, against the Kalshi-implied mean, the Bloomberg consensus and the Cleveland Fed nowcast. What breaks:
- the deep nonlinear transformer, because the sample is too small;
- the MSRR and HJD framework, because it targets the wrong quantity;
- the complexity-scaling argument, because there is no data to support it;
- significance claims, since about 50 events cannot produce the t-statistics of 10-30 in Tables 2-3.

Citing this paper to motivate "economic transformers" for RQ3 overstates what it shows. It is a proof of concept for cross-sectional equity SDFs, not for small-sample macro forecasting.

---
