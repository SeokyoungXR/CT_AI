# VirtualME Spatial Evidence Layer

## 1. Project concept

VirtualME converts fragmented personal records into a versioned, user-correctable model. CT_AI is used as a **spatial evidence layer**, not as a personality or emotion classifier.

The layer converts temporal 3D scene-graph traces into evidence-addressable records. It can describe which objects and spatial relations were observed, how dense the scene was, and how the observed scene changed over time. It must not claim why the person acted or what the person felt.

### Product idea: Spatial Resonance Room

A user reviews a time-based spatial memory built from their own photos or scene traces.

- **Object Field**: observed objects accumulate on a time axis.
- **Relation Threads**: relations such as `screen -> on -> desk` are shown as explicit edges.
- **Evidence Gaps**: periods without records remain gaps; they are never interpolated.
- **User Reading**: the user may accept, reject, or reinterpret a candidate pattern.
- **Wave Creature input**: approved aggregate features can drive procedural form, while uncertainty controls visual sharpness.

The first useful product slice is:

> Input 10-30 scene frames, generate spatial evidence records, show repeated objects and relations, and let the user choose the interpretation.

## 2. Data contract

`build_spatial_records()` in `scripts/spatial_evidence.py` produces one `SpatialEvidenceRecord` per temporal frame.

| Field | Meaning |
|---|---|
| `record_id` | Permanent source record identifier |
| `episode_id` | Current frame-level grouping identifier; later episode clustering may replace this |
| `objects` | Object label, 3D position, class id, and covariance-derived uncertainty |
| `relations` | Subject, predicate, object, and relation score |
| `features` | Object count, object diversity, relation count/density, spatial change rate |
| `evidence_class` | Always `contextual` in this adapter; it does not prove intent |
| `timestamp` | Optional timestamp derived from an explicit start time and FPS |

The output is JSON serializable through `SpatialEvidenceRecord.to_dict()`. No image pixels, names, aliases, or raw personal content are emitted.

## 3. Feature interpretation boundaries

Allowed statements are observational:

- "This object combination appears in 12 records."
- "The relation `screen on desk` was detected with an average score of 0.86."
- "Spatial change was higher in this interval."

Forbidden automatic statements:

- "The user was productive."
- "The user was lonely."
- "The user was depressed."
- "The object proves the user's intention."

A scene feature may become evidence for a VirtualME dimension only after cross-source analysis and user review. The result must retain record ids, episode ids, confidence, and provenance.

## 4. Baseline algorithm

1. Load and validate the complete temporal trace.
2. Decode object classes from ids or one-hot/logit arrays.
3. Convert covariance matrices into scalar positional uncertainty using mean variance.
4. Convert compact or dense relation tensors into scored edges.
5. Calculate per-frame density, diversity, relation density, and displacement.
6. Preserve empty relations and missing timestamps as valid states.
7. Pass records to an episode clustering and review layer.

This is intentionally a deterministic baseline. Future models may add tracking, change-point detection, or graph embeddings, but they must preserve the output contract and compare against this baseline.

## 5. Privacy and safety

- Process original images locally whenever possible.
- Store aggregate features instead of raw personal content.
- Keep third-party identities out of the scene record.
- Never infer a psychological diagnosis or causal explanation.
- Keep rejected readings as append-only review history.
- Do not share data with the upstream repository automatically. This project uses the user's fork remote only.

## 6. Validation

Run the focused adapter tests with:

```bash
python3 -m unittest tests.test_trace_io -v
```

The test suite verifies trace loading, frame integrity, object feature preservation, and invalid FPS rejection.

## 7. Next implementation stages

1. Add explicit episode clustering using timestamp and coarse location.
2. Add a JSONL/JSON export command for VirtualME ingestion.
3. Add repeated object/relation aggregation with confidence intervals.
4. Add change-point detection and gap visualization.
5. Add a browser-safe renderer that maps uncertainty to sharpness and never interpolates across gaps.
6. Evaluate scene features against user-reviewed readings without treating agreement as ground truth.
