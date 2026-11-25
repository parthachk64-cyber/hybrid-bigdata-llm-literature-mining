"""Generate the ORIGINAL, SYNTHETIC demo corpus. Nothing here is a real publication, DOI, author or result.

DOIs use the 10.5555 prefix with the literal token 'litmine.syn' and are not registered anywhere.
Run:  python scripts/make_demo_corpus.py   (writes src/litmine/demo/*)
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "litmine" / "demo"


def doi(n, v=None):
    return f"10.5555/litmine.syn.{n:03d}" + (f"v{v}" if v else "")


A, B = "0000-0000-0000-0001", "0000-0000-0000-0002"
H, M, G, L, T, X, E = "hydrology", "manufacturing", "genomics", "linguistics", "transport", "materials", "energy"

# (n, year, subjects, title, abstract, body, refs(list of n), authors)
RAW = [
    (1, 2008, [H], "Mann-Kendall trend testing of annual streamflow records",
     "We apply the Mann-Kendall test to annual streamflow records from forty river gauges. Slow monotonic trends are found at a third of the gauges, but abrupt shifts are not tested.",
     "The Mann-Kendall statistic assumes no serial correlation. Abrupt discharge shifts after land-use change remain undetected by this trend test.", [], ["R. Alder", "K. Mbeki"]),
    (2, 2010, [H], "Wavelet decomposition of rainfall regimes in monsoon catchments",
     "Wavelet decomposition separates seasonal and interannual rainfall variability in monsoon catchments. We locate regime changes at scales between two and eight years.",
     "Wavelet coefficients are thresholded to isolate regime change in the rainfall record. The method needs long records and is unsuitable for short sensor streams.", [], ["K. Mbeki", "S. Ortega"]),
    (3, 2011, [H], "Cumulative sum detection of discharge shifts after dam construction",
     "A CUSUM chart detects abrupt shifts in river discharge sensor streams after dam construction. Detection delay is two days on average for shifts larger than one standard deviation.",
     "The CUSUM reference value is set from pre-dam discharge variance. A fixed decision threshold causes false alarms during flood season.", [], ["S. Ortega", "L. Haddad"]),
    (4, 2012, [M], "EWMA control charts for tool wear drift in milling",
     "An EWMA control chart tracks tool wear drift in milling spindle current streams. The chart flags gradual drift earlier than a Shewhart chart at equal false alarm rate.",
     "The smoothing constant trades detection speed against false alarms. Abrupt shifts from tool breakage are detected more slowly than gradual wear.", [], ["D. Weiss", "P. Nair"]),
    (5, 2012, [H], "Bayesian online changepoint detection for river gauge streams",
     "We apply Bayesian online changepoint detection to river gauge sensor streams and recover abrupt shifts with a run-length posterior. The method needs no fixed threshold.",
     "We follow the cumulative sum design of the earlier discharge study [1] but replace its fixed threshold with a run-length posterior. Posterior mass on short run lengths signals an abrupt shift.", [3], ["L. Haddad", "R. Alder"]),
    (6, 2014, [M], "Cumulative sum monitoring of spindle vibration signals",
     "CUSUM monitoring of spindle vibration signals detects abrupt shifts in bearing condition on a milling line. The detector runs on streaming sensor data with constant memory.",
     "The reference value is tuned on healthy vibration windows. Abrupt shifts of one standard deviation are flagged within forty samples.", [], ["P. Nair", "D. Weiss"]),
    (7, 2015, [H], "Kalman filter residual monitoring of groundwater level sensors",
     "A Kalman filter residual test monitors groundwater level sensors and flags abrupt sensor faults. Innovations outside three standard deviations trigger an alarm.",
     "The state model assumes slowly varying aquifer storage. Seasonal pumping violates the assumption and raises the false alarm rate.", [], ["S. Ortega", "A. Quinn"]),
    (8, 2014, [M], "Kalman filter residual test for process sensor faults",
     "A Kalman filter residual test detects abrupt sensor faults in a chemical process line. The test outperforms fixed limit checks on step faults.",
     "Residuals are standardised by the innovation covariance. Drift faults are detected later than step faults.", [], [{"name": "J. Park", "orcid": A}, "D. Weiss"]),
    (9, 2016, [H], "Regime shifts in lake temperature sensor networks",
     "We apply Bayesian online changepoint detection and wavelet decomposition to lake temperature sensor networks. Abrupt regime shifts coincide with ice-off dates across nine lakes.",
     "The Bayesian method of earlier gauge work [1] transfers to lake temperature streams with a new hazard rate. Wavelet scales confirm the detected regime shifts.", [5], ["A. Quinn", "L. Haddad"]),
    (10, 2016, [M], "Isolation forest for anomaly screening in assembly line vibration",
     "Isolation forest screens anomalies in assembly line vibration streams without labelled faults. It ranks sensor windows by isolation depth.",
     "Contamination is set from historical downtime logs. The forest does not model temporal order and misses slow drift.", [], ["P. Nair", "C. Vega"]),
    (11, 2018, [H], "Seasonal confounding in streamflow shift detection",
     "Seasonal cycles inflate false alarms when abrupt shifts are detected in streamflow. We show that deseasonalising first removes most false alarms.",
     "Unlike simple trend testing [1], shift detectors fail when seasonality is unmodelled. Mann-Kendall results alone do not separate trend from cycle.", [1], ["R. Alder", "S. Ortega"]),
    (12, 2017, [M], "Adaptive Page-Hinkley thresholds for sensor drift",
     "Adaptive thresholds make the Page-Hinkley test robust to sensor drift in production lines. The adaptive rule halves false alarms on injection molding streams.",
     "The threshold adapts to a running variance estimate. Abrupt shifts are still detected within thirty samples.", [], ["C. Vega", "D. Weiss"]),
    (13, 2017, [M], "Kernel two-sample tests for drift in production sensor distributions",
     "A kernel two-sample test based on maximum mean discrepancy detects distribution drift in production sensor streams. It finds abrupt shifts that mean-based charts miss.",
     "Windows of sensor readings are compared with a Gaussian kernel. The test builds on cumulative sum monitoring [1] but compares whole distributions rather than means.", [6], ["C. Vega", "P. Nair"]),
    (14, 2018, [M], "Page-Hinkley detection of sudden quality shifts in injection molding",
     "The Page-Hinkley test detects sudden quality shifts in injection molding pressure streams. Detection delay is under fifteen cycles for large shifts.",
     "The cumulative deviation statistic is reset after every alarm. We use a conservative tolerance to limit false alarms.", [], ["D. Weiss", "C. Vega"]),
    (15, 2014, [G], "Changepoint detection in copy number profiles", "",
     "", [], ["M. Ito", "F. Duarte"]),
    (16, 2013, [G], "Circular binary segmentation of genomic copy number shifts",
     "Circular binary segmentation detects copy number shifts along genomic profiles. We detect shifts in tumour samples with segment-level false discovery control.",
     "The statistic is evaluated at every candidate breakpoint of the profile. Detected shifts are validated against array data.", [], ["M. Ito", "F. Duarte"]),
    (17, 2016, [G], "Bayesian changepoint models for chromatin accessibility",
     "A Bayesian changepoint model segments chromatin accessibility tracks. Posterior breakpoints align with regulatory element boundaries.",
     "The prior on segment length controls the number of detected shifts. Accessibility counts are modelled as overdispersed.", [], ["F. Duarte", "N. Okafor"]),
    (18, 2014, [L], "Negation scope detection in clinical notes",
     "Negation scope detection finds the words affected by a negation cue in clinical notes. A conditional random field reaches high scope accuracy on discharge summaries.",
     "Negation cues such as no and without are tagged first. Scope ends at the next punctuation mark in most notes.", [], ["N. Okafor", "H. Lindqvist"]),
    (19, 2018, [H], "Stochastic rainfall generators for flood design",
     "Stochastic rainfall generators produce synthetic storms for flood design. We compare two generators on extreme quantiles.",
     "Both generators reproduce mean intensity but differ in tail behaviour. No sensor stream monitoring is involved.", [], [{"name": "J. Park", "orcid": B}, "K. Mbeki"]),
    (20, 2016, [L], "Sentence embeddings for semantic textual similarity",
     "Sentence embeddings trained with a contrastive objective improve semantic textual similarity. The encoder maps sentences to dense vectors compared by cosine similarity.",
     "Training pairs are mined from paraphrase corpora. Cosine similarity of the embeddings predicts human similarity ratings.", [], ["H. Lindqvist", "N. Okafor"]),
    (21, 2022, [L], "Retrieval-augmented question answering over scientific abstracts",
     "Retrieval-augmented generation answers questions over scientific abstracts by quoting retrieved passages. Answers cite the passages that support each claim.",
     "A dense retriever fetches passages and a generator writes the answer. Unsupported questions should be refused rather than answered.", [], ["H. Lindqvist", "R. Sato"]),
    (22, 2019, [H, M], "Transferring changepoint methods between river gauges and assembly lines",
     "We transfer Bayesian online changepoint detection from river gauges and cumulative sum monitoring from spindle vibration to the other domain. Both abrupt shift detectors work in hydrology and manufacturing sensor streams.",
     "We apply the Bayesian run-length method from river gauge monitoring [1] and the cumulative sum rule from spindle vibration monitoring [2] to both data types. The transfer needs only a retuned hazard rate and a retuned reference value.", [5, 6], ["A. Quinn", "P. Nair"]),
    (23, 2015, [T], "Graph neural networks for road traffic flow forecasting",
     "Graph neural networks forecast road traffic flow on a sensor graph of loop detectors. Spatial dependencies between roads improve the forecast.",
     "Each detector is a node and roads define edges. The forecast horizon is up to one hour ahead.", [], ["I. Costa", "R. Sato"]),
    (24, 2017, [T], "Incident detection from loop detector streams using CUSUM",
     "A CUSUM chart detects traffic incidents as abrupt shifts in loop detector occupancy streams. Detection takes under three minutes.",
     "The reference value is set per time of day. Highway sensors only; no hydrology or manufacturing data are used.", [], ["I. Costa", "M. Ito"]),
    (25, 2012, [X], "Phase shift detection in X-ray diffraction patterns",
     "Phase shift between diffraction patterns is detected by cross-correlation. We detect shifts of fractions of a degree in alloy samples.",
     "The shift in angle is estimated per peak. This is a spatial shift in a pattern, not a change in time.", [], ["T. Berg", "V. Rao"]),
    (26, 2020, [X], "Spectral shift detection in polymer films",
     "Spectral shift detection tracks absorption peak position in polymer films. Peak shifts reveal curing progress.",
     "Peak position is fitted with a Lorentzian. The method needs one spectrum per film.", [], ["J. Park", "T. Berg"]),
    (27, 2021, [E], "Detecting abrupt capacity shifts in battery cycling data",
     "Abrupt capacity shifts in battery cycling data are detected with a likelihood ratio test. The detector flags knee points in capacity fade.",
     "The test uses a piecewise linear model. Cycling data come from laboratory cells, not field sensor streams.", [], ["V. Rao", "T. Berg"]),
    (28, 2020, [G], "Detecting abrupt shifts in single-cell expression trajectories",
     "Abrupt shifts in single-cell expression trajectories mark cell fate decisions. A changepoint detector on pseudotime finds the shifts.",
     "Expression is smoothed along pseudotime. Detected shifts agree with known marker genes.", [], ["N. Okafor", "F. Duarte"]),
    (29, 2020, [L], "Citation intent classification for scientific text",
     "Citation intent classification labels why a paper cites another, for example background, method use or contrast. A fine-tuned language model labels intent from the citing sentence.",
     "Intent depends on the citing sentence and its neighbours. Sentences without a cue should be labelled unknown rather than guessed.", [], ["R. Sato", "H. Lindqvist"]),
    (30, 2023, [M], "Drift monitoring for automated assembly lines using transferred hydrological methods",
     "We monitor drift on an automated assembly line with Bayesian online changepoint detection and a kernel two-sample test. The monitor reuses methods transferred from hydrology.",
     "Unlike fixed-threshold charts, our monitor extends the transferred method of [1]. Kernel tests [2] were described for production sensors and are used here as the second detector.", [22, 13], ["P. Nair", "C. Vega"]),
    (31, 2022, [H], "Hydrological changepoint toolboxes revisited",
     "We review changepoint toolboxes used in hydrology for abrupt shifts in streamflow sensor records. The review compares Bayesian online changepoint detection and CUSUM implementations.",
     "The transfer study [1] first described how manufacturing monitors apply to river gauges. Lake network results [2] are included for completeness.", [22, 9], ["A. Quinn", "R. Alder"]),
    (32, 2021, [T], "Traffic demand shifts and transformer forecasting",
     "Transformer models forecast traffic flow under demand shifts. Attention over sensor graphs improves peak hour accuracy.",
     "Demand shifts after events are handled by fine-tuning. Graph neural networks [1] remain a strong baseline for traffic flow.", [23], ["I. Costa", "R. Sato"]),
    (33, 2021, [M], "Online EWMA with adaptive thresholds for milling drift",
     "An online EWMA chart with adaptive thresholds tracks tool wear drift in milling. It reduces false alarms compared with the fixed-limit chart.",
     "We build on the earlier control chart for tool wear [1] and add a variance-adaptive limit. Abrupt breakage is still flagged by a secondary CUSUM rule.", [4], ["D. Weiss", "C. Vega"]),
    (34, 2019, [H], "Flood frequency nonstationarity under regime change",
     "Flood frequency analysis is extended to nonstationary records using Mann-Kendall screening and wavelet decomposition. Regime change shifts the flood quantiles.",
     "Mann-Kendall screening [1] removes stationary gauges before wavelet analysis. Wavelet scales identify regime change in the remaining records.", [1], ["R. Alder", "K. Mbeki"]),
    (35, 2024, [M], "Isolation forest ensembles for streaming vibration",
     "Ensembles of isolation forest models screen anomalies in streaming vibration data from assembly lines. A sliding window keeps the model current.",
     "The ensemble extends the single-forest screening of earlier line monitoring [1]. Window size trades latency against stability.", [10], ["C. Vega", "P. Nair"]),
]


def author(a):
    return a if isinstance(a, dict) else {"name": a}


def record(n, year, subj, title, abstract, body, refs, authors, version=None, extra=None):
    r = {"source_id": f"S{n:02d}" + (f"v{version}" if version else ""), "doi": doi(n, version), "version": version or 1,
         "title": title, "authors": [author(a) for a in authors], "date": f"{year}-06-01", "abstract": abstract, "body": body,
         "references": [{"doi": doi(t)} for t in refs], "license": "CC0-1.0 (synthetic)", "subjects": subj}
    if extra:
        r.update(extra)
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [record(*row) for row in RAW]
    # duplicate older version of S12 (different abstract, lower version): must be dropped by dedup
    n, year, subj, title, ab, body, refs, au = RAW[11]
    recs.append(record(n, year, subj, title, ab.replace("halves", "reduces"), body, refs, au, version=1,
                       extra={"source_id": "S12v1", "doi": "10.5555/litmine.syn.012v1", "version": 1}))
    recs[11]["source_id"], recs[11]["doi"], recs[11]["version"] = "S12v2", "10.5555/litmine.syn.012v2", 2
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    def ids(*ns):
        return [doi(n) for n in ns]

    Q = {
        "q01": ("Which methods have hydrology and manufacturing used to detect abrupt shifts in sensor streams?",
                {2: [3, 5, 6, 13, 14, 22], 1: [4, 7, 8, 9, 10, 12, 30, 31, 33, 35]}),
        "q02": ("Bayesian online changepoint detection for streamflow gauge records", {2: [5], 1: [9, 22, 31]}),
        "q03": ("control chart for tool wear drift in milling", {2: [4], 1: [33, 6]}),
        "q04": ("kernel two-sample test for drift in production sensor distributions", {2: [13], 1: [30]}),
        "q05": ("wavelet decomposition of rainfall regimes", {2: [2], 1: [9, 34]}),
        "q06": ("isolation forest anomaly screening of assembly line vibration", {2: [10], 1: [35, 6]}),
        "q07": ("copy number changepoint segmentation in genomic profiles", {2: [16, 15], 1: [17]}),
        "q08": ("negation scope detection in clinical notes", {2: [18]}),
        "q09": ("traffic flow forecasting with graph neural networks", {2: [23], 1: [32]}),
        "q10": ("effect of lunar gravity on yeast fermentation", {}),
        "q11": ("citation intent classification", {2: [29], 1: [21]}),
    }
    with open(OUT / "queries.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for qid, (text, _) in Q.items():
            fh.write(json.dumps({"qid": qid, "text": text}) + "\n")
    with open(OUT / "qrels.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for qid, (_, rel) in Q.items():
            for g, ns in rel.items():
                for n in ns:
                    fh.write(f"{qid}\tdoi:{doi(n)}\t{g}\n")
    split = {"train": ["q01", "q03", "q04", "q07", "q09", "q11"], "test": ["q02", "q05", "q06", "q08", "q10"]}
    (OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    sessions = {"user_a": [["q01", "q03"], ["q04", "q07"]], "user_b": [["q09", "q11"]],
                "note": "SIMULATED sessions: sequences of training queries. These are NOT real user clicks."}
    (OUT / "feedback_sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    print(f"wrote {len(recs)} records, {len(Q)} queries")


if __name__ == "__main__":
    main()
