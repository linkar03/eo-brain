# EO Brain: text-to-GIS with tool-calling and text-to-SQL agents

[![CI](https://github.com/linkar03/eo-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/linkar03/eo-brain/actions/workflows/ci.yml)

Ask for a satellite analysis in plain language, then ask questions about the result.
Two LLM agents do the talking; **code, not the model, decides what runs.**

![The chat asks for spatial outliers in vegetation around Fiesole. The map shows the result, and a follow-up question is answered with the SQL and the rows behind it.](docs/images/eo-brain-chat-map.webp)

*A real session, 2026-11-27: the request, the analysis it launched with its coverage, and a
follow-up answered by the data agent with its SQL and rows.*

- **A tool-calling orchestrator.** You type or dictate a request in Italian or English, for
  example *"allagamenti a Valencia negli ultimi 3 mesi"* or *"vegetation stress near Pisa"*.
  The agent geocodes the place and launches a Sentinel-2 analysis over a public STAC catalogue.
  The result is drawn on the map as a raster layer plus vectorised alert polygons, and it states
  how much of the area it covers.
- **A separate, read-only data agent.** It answers questions about analyses that already ran,
  for example *"how many hectares were at alert level in the latest flood analysis?"* or *"how
  much green is there in the latest NDVI analysis?"*. It writes SQL over curated PostGIS views,
  and the chat shows the SQL and the rows it returned.
- **A deterministic boundary around both.** Every tool call goes through validators before
  anything runs. Every query goes through an allow-list gate on the parsed SQL, then runs as a
  database role that can only read those views, in a read-only transaction with a timeout.
  See [The deterministic boundary](#the-deterministic-boundary).
- **Evals, offline and live.** The same cases replay in CI without a model and run live against
  one. The harness tells a failed case from a case that never reached the model.

## Results at a glance

| Measured live on 2026-11-26 | `gemini-3.6-flash` | `gemini-3.5-flash-lite` |
|---|---|---|
| Tool-calling cases passed | 39/39 | 39/39 |
| Requests that must be refused, refused | 12/12 | 12/12 |
| Adversarial SQL requests blocked | 8/8 | 8/8 |
| SQL answers with the right values | 9/10 | 8/10 |

"Right values" is column-subset accuracy. The stricter comparison, the method and what the
misses are: [Model-quality results](#model-quality-results). The suites have grown since that
run, to 61 tool cases and 28 SQL cases. CI runs over 850 backend tests and over 110 frontend
tests on every push.

To run it: [Quickstart](#quickstart-docker-compose). What it does not do yet:
[Limitations](#limitations).

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Scenes and coverage](#scenes-and-coverage)
- [Radiometry: from pixel values to reflectance](#radiometry-from-pixel-values-to-reflectance)
- [Index values: how much green, how much water](#index-values-how-much-green-how-much-water)
- [Voice input: the gesture](#voice-input-the-gesture)
- [The deterministic boundary](#the-deterministic-boundary)
- [The data agent and its isolation model](#the-data-agent-and-its-isolation-model)
- [Quickstart (docker compose)](#quickstart-docker-compose)
- [Configuration](#configuration)
- [Third-party services and data](#third-party-services-and-data)
- [Tests](#tests)
- [Evals](#evals)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [License](#license)

## What it does

| Algorithm | What it shows |
|---|---|
| `ndvi`  | Vegetation health composite |
| `ndwi`  | Water detection (green / NIR) |
| `mndwi` | Water detection (green / SWIR) |
| `flood` | Flood extent with severity classes (watch / suspect / alert) |
| `lisa`  | Spatial outliers in vegetation (Local Moran's I, HH / LL / HL / LH quadrants). Only vegetated pixels (NDVI ≥ `bio_gate`, default 0.3) enter the statistics and the neighbourhood. |

- **Data sources:** STAC providers `element84-s2` (default, and the only one the agent uses), `element84-landsat`, `copernicus-s2` and `planetary-s2`. `element84-landsat` and `copernicus-s2` do not work as configured (see [Limitations](#limitations)).
- **Processing:** lazy Dask/xarray pipelines, so large areas do not load into memory at once. Scenes are chosen by what their pixel mask shows over the area, not by their tile-level cloud cover, and every result states how much of the area it covers (see [Scenes and coverage](#scenes-and-coverage)). Pixel values become surface reflectance scene by scene, with a rule declared per provider and recorded in the result (see [Radiometry](#radiometry-from-pixel-values-to-reflectance)).
- **Index values:** for every completed `ndvi`, `ndwi` and `mndwi` analysis, how its values are spread over the covered area is stored in intervals 0.01 wide, so the data agent answers "how much green" or "how much water" as a sum over a threshold (see [Index values](#index-values-how-much-green-how-much-water)).
- **Outputs:** Cloud-Optimised GeoTIFFs served as XYZ tiles and a preview image (`/tiles/...`), and alert polygons stored in PostGIS and served as vector tiles or GeoJSON (`/vectors/...`). A tile or preview requested without `colormap` is drawn with the colormap of the algorithm that produced the raster, the same one the job's `tile_url` names (`ndvi`; `water` for NDWI and MNDWI; `flood`; `severity` for LISA).
- **Frontend:** React, MapLibre and a chat panel. Basemaps: OpenStreetMap (default) and Esri World Imagery (opt-in, see [Third-party services and data](#third-party-services-and-data)).
- **Voice input (optional):** a microphone button next to the chat box dictates the request with the browser's own speech recognition (Web Speech API, `frontend/src/hooks/useSpeechInput.ts`). It appears only in browsers that have the API.
  - **The words go into the text box, never straight to the agent.** Interim words show while you speak; you read the text, correct it if needed, and send it with the Send button or Enter. This is deliberate. The agent only ever receives text that you could see and edit, and that then goes through the same deterministic code as typed text (see [The deterministic boundary](#the-deterministic-boundary)). A speech recogniser is one more model, and a mis-heard place name must not launch an analysis by itself.
  - Language: an explicit IT / EN choice next to the microphone (`it-IT`, `en-US`). The default is English when the browser's language is English, and Italian otherwise, as in the orchestrator.
  - **Push-to-talk is the main gesture.** Hold the microphone (mouse, touch, or Space / Enter while the button has focus): it listens for as long as you hold it, pauses included, and stops when you let go, keeping every word. Dictation is added after the text already in the box, separated by a space.
  - **A short press is hands-free.** Released within 300 ms, the press starts listening that goes on until you press the microphone again. It does not stop on a pause, but it stops by itself after 10 seconds without a recognised word, and after 2 minutes in any case, keeping the words.
  - The line under the box names the mode in use and how to end it: "Push-to-talk (Italian): listening while you hold. Release to stop." or "Hands-free (Italian): listening until you press the microphone again; it stops by itself after 10 seconds of silence." A safety stop says why it stopped.
  - Browsers end a recognition session by themselves (after silence, or after some time). While you still hold, or hands-free is on, a new session starts at once and its words are appended. Nothing is written twice. If the browser keeps ending sessions as soon as they start, listening gives up with a message.
  - Typing in the box while it listens stops listening at once and drops what was not yet transcribed, so the box keeps exactly what you see and the recogniser never overwrites your typing. Sending, closing the chat or changing the language also stops it.
  - Listening shows a stop icon on the pressed button. Each failure the browser reports gets a short message under the box: microphone permission denied, no microphone, network error, language not supported, interrupted.
  - How the edge cases are handled (the pointer leaving the button, the window losing focus, a cancelled touch, the permission prompt on first use): [Voice input: the gesture](#voice-input-the-gesture).
  - Where the audio goes and which browsers support it: [Speech recognition](#speech-recognition-in-the-browser-voice-input-only).

## Architecture

```mermaid
flowchart LR
    UI["Frontend<br/>React + MapLibre<br/>AgentChat"] -->|POST /api/chat| API["FastAPI<br/>eo_brain/api/main.py"]
    API --> ORCH["Orchestrator agent<br/>eo_brain/agent/brain.py"]
    ORCH <-->|LLMProvider seam| LLM[("Gemini<br/>or scripted replay")]

    ORCH -->|launch_analysis<br/>rerun_with_different_algorithm| VAL["Validators + geocoder<br/>(deterministic)"]
    VAL --> JOBS[("PostGIS<br/>jobs, alert_features,<br/>index_value_bins")]
    API -->|worker thread| PIPE["STAC search + Dask pipeline<br/>COG + vectorised polygons"]
    PIPE --> JOBS
    ORCH -->|get_job_status| JOBS

    ORCH -->|"query_database(question)"| DA["Data agent<br/>eo_brain/agent/data_agent.py"]
    DA <-->|own prompt + own context| LLM
    DA --> GATE["SQL gate<br/>sql_gate.py (sqlglot AST)"]
    GATE --> EXEC["Read-only executor<br/>sql_exec.py"]
    EXEC -->|role agent_reader| VIEWS[("schema agent<br/>curated views")]
    VIEWS -. reads .-> JOBS
```

The orchestrator is a manual function-calling loop. The model sees five tools:

1. `launch_analysis`
2. `get_job_status`
3. `list_capabilities`
4. `rerun_with_different_algorithm`
5. `query_database`

Code executes every call and returns a structured result. Each request is capped at 5 tool rounds.

The model sits behind a small provider interface (`eo_brain/agent/llm.py`) with two implementations:

- `GeminiProvider` (`providers/gemini.py`, google-genai SDK, automatic function calling disabled);
- `ScriptedProvider` (`providers/scripted.py`), a deterministic replay used by tests and offline evals.

The model id and its fallback chain come from configuration (see [Configuration](#configuration)).

## Scenes and coverage

### Why this changed

The pipeline used to ask the catalogue only for scenes whose tile-level cloud cover (`eo:cloud_cover`, one number for a 110 km Sentinel-2 tile) was under 30%. In a cloudy month that can leave one scene for an area that other scenes saw clearly. An NDVI request for the Casentino valley (box `11.572, 43.509, 11.987, 43.878`, 60 m, 2026-10-28 to 2026-11-27) found 27 scenes, kept one, and drew a result over 7.3% of the area. The job reported plain success.

### How scenes are chosen (`eo_brain/core/scenes.py`)

The per-pixel cloud mask (Sentinel-2 SCL) is the real filter; the scene-level cloud cover is only a ranking and a loose bound.

1. Search the window with no cloud filter.
2. Drop scenes whose tile-level cloud cover is 95% or more. Keep at most the 16 clearest scenes of each tile.
3. Screen the rest: read only the mask band of each, at a coarse resolution (at most 256 pixels on the longer side), over the requested box. This gives, per scene, where it has data and where it is clear.
4. Group the scenes by acquisition day, as the loader does (one time slice per day). Within a day, drop a scene that adds data over less than 0.5% of the area: overlapping tiles of one pass hold the same pixels.
5. Pick days greedily. Each step takes the day that adds the most clear pixels where the area has no clear view yet, then where it has fewer than three. It stops when no day adds at least 0.5% of the area, or when the next day would exceed the budget. When several days are within 10% of the best gain, the one farthest in time from the days already picked wins, so a clear month is sampled across the window.
6. `flood` is different: it keeps each pixel's maximum water index over time, so the one day a flood was visible matters, not three views of each pixel. For `flood` (and any request with `composite_method: "max"`), every clear view counts the same with no target, and every day that adds clear pixels is kept, within the same budget (`selection_for`).

The work is bounded: the picked scenes together hold data for at most 10 times the requested area, and at most 40 scenes are opened. Only the picked scenes are loaded at full resolution, and the per-pixel mask and the composite (median, or maximum for `flood`) apply as before. If the provider has no mask band, or the screening fails, the clearest scenes by tile-level cloud cover are taken instead (at most 10). The loader now also uses odc-stac's settings for cloud-hosted files (`configure_rio(cloud_defaults=True)`: no directory listing on open, HTTP retries), which alone cut the Casentino run from 46 s to about 20 s.

Measured on 2026-11-27 with the API image (`docker run`, real Earth Search catalogue, `element84-s2`, 60 m; the pipeline function called directly, no API or database):

| Case | Before | After |
|---|---|---|
| Casentino NDVI, 2026-10-28 to 2026-11-27 (cloudy) | 1 scene of 27; 7.3% of the area; 4 to 5 s | 8 scenes on 8 days; 84.4% of the area (the remaining 15.6% was cloud-masked on every day); 17 to 20 s |
| Valencia NDVI, box `-0.6, 39.3, -0.2, 39.6`, July 2026 (clear) | 24 scenes on 7 days; 100%; 23 to 26 s | 4 scenes on 4 days; 99.9%; 13 to 16 s |
| Casentino `flood`, same month | 1 scene; 7.2%; 5 s | the same 8 scenes on 8 days; 84.4%; 19.5 s |
| Valencia `flood`, same month | 24 scenes on 7 days; 99.9%; 26 s | 9 scenes on 9 days; 100%; 22 s |

- Peak memory (NDVI): Casentino about 285 MB before and 545 MB after; Valencia about 490 MB before and 480 MB after.
- On the clear case the NDVI composite now uses 4 of the 7 clear days. Mean NDVI moved from 0.231 to 0.233; the mean absolute per-pixel difference from the 7-day composite is 0.026 (95th percentile 0.115).
- Valencia `flood`: alert pixels 11,924 before and 11,863 after; flagged area 355.3 km² before and 355.4 km² after. Run with the NDVI rule (4 days), the same request flagged only 9,611 alert pixels, which is why `flood` keeps every day.
- Casentino over 60 days (to 2026-11-27) reaches 99.9% with 3 scenes in 15.5 s. Over the 7 days from 2026-11-10 to 2026-11-17 it reaches 5.3% and is flagged as low coverage.

### Coverage is part of every result (`eo_brain/core/coverage.py`)

- **What is measured.** The share of the requested box (in WGS84, as asked) whose pixels have a value in the result: at least one cloud-free observation passed the mask. For `flood` and `lisa`, whose class layers write 0 for both "none" and "no data", it is read from the index they are computed from. It is rounded down to 0.1%.
- **Why the rest is blank.** The uncovered part is split in two: pixels the loaded scenes saw but the mask removed on every day (cloud, shadow, snow), and pixels outside the footprint of every loaded scene.
- **Where it is stored.** The job's `result.coverage` (returned by `GET /jobs/{job_id}`) holds `coverage_pct`, `low_coverage`, `uncovered_masked_pct`, `uncovered_no_data_pct`, the scene counts (`scenes_found`, `scenes_too_cloudy`, `scenes_checked`, `scenes_with_clear_pixels`, `scenes_used`), the acquisition days (`acquisition_days_with_data`, `first_date_with_data`, `last_date_with_data`) and the selection details (`selection`: method, picked days, scene ids).
- **What the user sees.** The job's message always states the coverage, for example *"Completed: cloud-free data covers 84.4% of the requested area (8 acquisition days, 2026-10-29 to 2026-11-20)."*
- **Low coverage is not a silent success.** Below 80%, the job still completes, but `low_coverage` is true, the chat badge reads "Map updated: partial coverage" in amber, and the message gives the reason and what to try. Replaying the old filter on the Casentino case gives: *"Completed with partial coverage: cloud-free data covers only 7.2% of the requested area (1 acquisition day, 2026-10-29). Reason: 0.3% of the area was masked out (cloud, shadow, snow) on every day loaded; 92.4% was outside the footprint of the scenes loaded. A longer time window may cover more of it."* A job with no covered pixel at all fails, with the scene counts in its message.
- **Why 80%.** Below it, a fifth or more of the area is blank: the map shows large holes, and every area figure (hectares at alert, mean index) describes a smaller area than the one asked for. Above it, the gaps are mostly what the mask removes even in a good month: cloud remnants and their shadows, snow, and water or shadowed slopes it is unsure about. A higher threshold would fire on most mountain analyses and stop meaning anything. The constant is `LOW_COVERAGE_PCT`; the view comments state the same number, and a DB test checks that they agree.
- **The agents can read it.** The data agent's view `agent.analyses` has the columns `coverage_pct`, `low_coverage`, `uncovered_masked_pct`, `uncovered_no_data_pct`, `scenes_found`, `scenes_too_cloudy`, `scenes_checked`, `scenes_with_clear_pixels`, `scenes_used`, `acquisition_days`, `first_acquisition` and `last_acquisition`, each documented in its column comment. They are NULL for analyses that ran before coverage was recorded. The orchestrator's `get_job_status` returns the same block, with the place and the time window, and defaults to the latest analysis of the conversation.
- **The orchestrator must not invent causes.** Asked what an analysis shows or why it looks the way it does, it is told to answer only from `get_job_status` or `query_database` results, and to say that the data does not tell when they do not explain it. It must not offer causes the results do not state, such as satellite orbits, terrain, sensor faults or processing errors. This is a prompt rule (see [The deterministic boundary](#the-deterministic-boundary)).

## Radiometry: from pixel values to reflectance

Every index (NDVI, NDWI, MNDWI, and through them `flood` and `lisa`) is computed from surface reflectance, a number from 0 to 1. The catalogues deliver integers (digital numbers, DN). How a DN becomes reflectance depends on the product, on its processing baseline and on what the provider did to the files. So the rule is decided per scene, never per band or per provider (`eo_brain/core/radiometry.py`).

### Why this changed

The algorithms used to divide integer bands by 10000 and to pass float bands through unchanged. After the cloud mask the bands are float, so in practice nothing was scaled. NDVI and the water indices are ratios, so dividing both bands by 10000 does not change them, and on Earth Search the results were right. They were right because Earth Search had already removed ESA's offset, and nothing checked that. On Planetary Computer, whose DNs still carry the offset, the same code gave a Casentino NDVI median of 0.490 instead of 0.847 (same scene, same pixels; table below).

### The rules and their sources

reflectance = DN × scale + offset, where DN is not nodata. DN 0 is nodata for every rule.

| Provider | Scenes | Reflectance | Checked on pixels |
|---|---|---|---|
| `element84-s2` (Earth Search `sentinel-2-l2a`) | processing baseline before 04.00 | DN / 10000 | yes (2021 product, below) |
| | baseline 04.00 or later, `earthsearch:boa_offset_applied` true | DN / 10000: Earth Search removed the offset | yes |
| | baseline 04.00 or later, flag false or missing | **not loaded** (below) | — |
| `planetary-s2` | baseline before 04.00 | DN / 10000 | no |
| | baseline 04.00 or later | (DN − 1000) / 10000 | yes |
| `copernicus-s2` | as `planetary-s2`, baseline from `processing:version` | | **no**: the assets need authentication |
| `element84-landsat` | Collection 2 Level-2 (`landsat:collection_number` 02, `landsat:correction` L2SP/L2SR) | DN × 0.0000275 − 0.2 | on Planetary Computer's copy of the product, not on Earth Search's (Requester Pays) |

- **Sentinel-2 L2A (ESA).** ESA's [Sentinel-2 product page](https://sentiwiki.copernicus.eu/web/s2-products) gives `L2A_SR = (L2A_DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE` from processing baseline 04.00 (25 January 2022), with "DN=0 remains the NO_DATA value". The offset "will allow encoding negative surface reflectances that may occur over very dark surfaces" ([Sentinel Online, 2021](https://sentinels.copernicus.eu/-/copernicus-sentinel-2-major-products-upgrade-upcoming)). The value is −1000 and fixed at L2A ([ESA's Sentinel-2 technical manager on the STEP forum](https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431)). Planetary Computer serves ESA's DNs as they are, offset included ([its baseline-change notebook](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb)).
- **The baseline, not the date.** The baseline is read from the item: `s2:processing_baseline`, `processing:version` (Copernicus Data Space), or the `Nxxyy` field of the product name. Reprocessed products of old dates have baseline 05.xx. Earth Search has two products of tile 32TPP for 2021-07-07: `_0_` with baseline 03.01 and flag false, `_1_` with 05.00 and flag true. Their red medians are 865 and 878 DN, so the 05.00 copy is offset-free too.
- **Earth Search.** Earth Search removes the offset when it writes its COGs and says so per item with `earthsearch:boa_offset_applied` ([earth-search #9](https://github.com/Element84/earth-search/issues/9)). The flag is not reliable when false. [#66](https://github.com/Element84/earth-search/issues/66) shows two 2022 items with baseline 04.00 and the flag false: one still carries the offset, the other does not. #66 and [#71](https://github.com/Element84/earth-search/issues/71) report more flag-false items without the offset, from 2024 and 2025. So a false or missing flag does not say which rule applies.
- **The assets' `raster:bands` scale and offset are never used.** On Earth Search every asset says `scale 0.0001, offset -0.1`, whether or not the offset was removed. Applying it to scene `S2B_32TPP_20261029_0_L2A` (baseline 05.13, flag true) subtracts the offset twice: 99.6% of its clear red pixels are at or below DN 1000, and the NDVI median over a 10 m window moves from 0.807 to 0.999. Copernicus Data Space declares `raster:offset -0.1` on assets that do carry the offset; the rule gives the same result there, but from the baseline, not from that field.
- **Landsat.** [USGS](https://www.usgs.gov/faqs/how-do-i-use-scale-factor-landsat-level-2-science-products): Collection 2 Level-2 surface reflectance is DN × 0.0000275 − 0.2, fill 0, valid range 7273 to 43636.

### Where the conversion happens

1. `STACProvider.radiometry(item)` gives each scene's rule; the API leaves out the scenes whose rule cannot be determined, before scene selection.
2. `DataCube.load_items` groups the scenes by (solar day, rule). Scenes of one day are pasted into one time slice before any conversion, so they must share a rule. When they do, which is the normal case, the slices are exactly the old one-per-day slices; a day that mixes rules becomes one slice per rule. The load sets each band's data type and nodata explicitly, also where the catalogue declares none (Planetary Computer). Each slice's rule is attached to the data as coordinates along `time` (`radiometry_rule`, `_scale`, `_offset`, `_nodata`).
3. `Algorithm._to_reflectance`, used by every index, converts each acquisition before the composite. Nodata and cloud-masked pixels become NaN first, then (DN + offset / scale) × scale is computed in float32. The offset is applied in DN units (−1000), so the same scene gives the same reflectance whether its DNs carry the offset or not. Integer and float input are converted the same way. A band with no declared rule is refused, not scaled by a guess.

### Decisions

- **A scene whose rule is undetermined is refused, not loaded with a warning.** A wrong offset moves every reflectance by 0.1, which moves an NDVI median from 0.81 to 1.00. A median composite would carry that into the map with no visible trace, and a warning in the result would not show on the map. Refusing costs coverage, and coverage is measured and reported. The job's message says how many scenes were left out and why. If no scene is left, the job fails with the reason.
- **Non-positive reflectance is raised to one DN step (0.0001), not masked.** With the offset, DNs under 1000 give reflectance ≤ 0 over very dark surfaces such as water and deep shadow. Earth Search does the same when it removes the offset: where Planetary Computer has DN < 1000, Earth Search has DN 1. Masking these pixels would remove dark water from the water indices and from `flood`. Reflectance above 1 (cloud remnants, snow) is kept as it is.

### Recorded with every job

`result.radiometry` (returned by `GET /jobs/{job_id}`, next to `coverage`):

- `provider` and `quantity` (`surface reflectance (0-1)`);
- `rules`: for each rule used, its `scale`, `offset`, `nodata`, `description`, and how many `scenes` and `acquisition_slices` fell under it;
- `non_positive`: the policy above;
- `asset_scale_offset_used`: always `false`;
- `scenes_refused`, and `refused`: the first 20 refused scenes, with the reason.

The Casentino job below records one rule, `s2-l2a-offset-removed`, for 8 scenes in 8 slices, and no refused scene.

### Checked on pixels

Measured on 2026-11-27 with the API image (`docker run`, real catalogues; Planetary Computer asset URLs signed by the `planetary-computer` package). The `planetary-s2` mask keeps SCL classes 4 and 5 and `element84-s2` also keeps 6 (water), so the comparison uses pixels with class 4 or 5 in both.

| Check | Result |
|---|---|
| Earth Search vs Planetary Computer, scene S2B 32TPP 2026-10-29 (baseline 05.13), Casentino box, 10 m, through `DataCube.load_items` and the algorithms' conversion | 1,287,837 clear pixels on both, identical SCL. Planetary Computer DN = Earth Search DN + 1000 for 99.2% of red pixels; the rest are at or below 1000 at ESA and 1 at Earth Search. NDVI, NDWI and MNDWI are identical on 100% of the pixels (largest difference 0). NDVI median 0.847 on both; the old code gave 0.490 on Planetary Computer |
| The same at 60 m | means and medians agree within 0.0004, but single pixels differ: mean absolute NDVI difference 0.0067, 99th percentile 0.042. This is the providers' overviews, not radiometry (see "Resampling at 60 m") |
| Landsat 9 (LC09 192/030) vs Sentinel-2C (32TPP), both 2026-11-03, 13 minutes apart, Planetary Computer, 30 m, 40,300 pixels clear in both | median red 0.0353 vs 0.0346, NIR (Landsat band 5 vs Sentinel-2 B8A) 0.2936 vs 0.2940, NDVI 0.795 vs 0.794. Read as DN / 10000, the Landsat red median would be 0.856. Pixel-to-pixel agreement is poor (correlation 0.31 red, 0.10 NIR, 0.78 NDVI): the Sentinel-2 scene was 47% cloudy, and clouds and shadows move in 13 minutes. This confirms the scale and offset, not per-pixel agreement |

### The default did not move

The Casentino job (box `11.572, 43.509, 11.987, 43.878`, 2026-10-28 to 2026-11-27, 60 m, `element84-s2`) was run before and after the change on the same 27 catalogue items, through the pipeline function, without the database:

- NDVI, NDWI and MNDWI: the same 343,616 valid pixels; largest difference 1.2 × 10⁻⁷ (one float32 step), no pixel differs by more than 10⁻⁶;
- `flood` and `lisa`: identical, pixel for pixel;
- the coverage block and the job message: identical.

For `lisa`, one more fix was applied to the old code too before comparing. Without it, `lisa` failed on this job: computed in one `dask.compute` call with the coverage counts, its graph raised `Missing dependency ('truediv-…')` (Dask 2026.11). The pipeline now computes the result first, then the coverage counts, which read the mask band a second time.

### Resampling at 60 m

At 60 m, odc-stac reads each band from the coarsest overview that is still finer than the request, and then picks the nearest of its pixels. Its default resampling is nearest, so the 60 m grid takes one 40 m overview pixel per cell:

- for the 10 m bands, the 4× overview (40 m); on Earth Search it is the AVERAGE of 4 × 4 pixels;
- for the 20 m bands, the 2× overview (40 m);
- for SCL, its 40 m overview, built by MODE.

Each 60 m pixel therefore stands for 44% of its area. On Earth Search scene S2B 32TPP 2026-10-29, compared with the mean of the 10 m NDVI inside each 60 m cell (36,089 cells):

- there is no bias that matters: the mean difference is −0.0014, the mean NDVI 0.7944 at 60 m vs 0.7955 at 10 m, and the area with NDVI > 0.3, > 0.6 and > 0.8 is within 1 percentage point;
- single pixels are noisy: the mean absolute difference is 0.020 (95th percentile 0.070, 99th 0.137);
- the masks disagree on about 0.3% of cells each way. 111 cells pass at 60 m although less than half of their 10 m pixels were clear, and 124 are masked although more than half were.

The result also depends on how each provider built its overviews. Planetary Computer's are neither a block mean nor a single pixel: they are 9.5 DN from the 2 × 2 mean at the 2× level. That is why the two providers agree exactly at 10 m but not at 60 m.

A fix, not implemented: load at 20 m, where Earth Search's 2× overview is an exact 2 × 2 mean and SCL is native; mask and compute the index per acquisition there; then average 3 × 3 blocks to 60 m, keeping a cell only when most of it was clear. The cost is about 4 times more data read than at 60 m and 9 times more pixels in memory, with the area statistics unchanged within the figures above.

## Index values: how much green, how much water

An `ndvi`, `ndwi` or `mndwi` result is a continuous value per pixel, with no alert polygons. To make "how much of the area is green" answerable from stored rows, every completed continuous-index analysis also records how its values are spread (`eo_brain/core/distribution.py`).

- **What is stored.** The covered pixels are counted per value interval 0.01 wide over [-1, 1]: interval *k* holds `value_from <= value < value_to`, the last one (0.99 to 1.00) also holds 1, and a value outside [-1, 1] is counted in the first or last interval. Only intervals holding at least one pixel are stored, in the table `index_value_bins` (`job_id`, `bin_index`, `value_from`, `value_to`, `pixel_count`, `area_m2`), deleted with their job. A value within 5·10⁻⁷ of an edge counts from that edge, so a float32 value stored as 0.65 (0.64999998) is in the 0.65 interval.
- **The same pixels as the coverage.** The pixels counted are those `result.coverage` counts as covered: inside the requested box and with a value. Their number must equal `covered_px`; a mismatch is refused rather than stored. The pixel area comes from the result grid's own spacing, in metres (a grid whose CRS is not projected in metres is refused).
- **Mean, lowest and highest value.** `result.value_distribution` holds the pixel count, the pixel and total area, the interval width, and the minimum, maximum and mean value over the same pixels; `agent.analyses` reads them as `value_min`, `value_max` and `value_mean`. They do not come from `result.summary`, which is computed from the COG over the whole result grid, the margin outside the requested box included (see [Limitations](#limitations)).
- **Written with the completion, never at its expense.** The intervals are computed with numpy on the result layer the pipeline has already computed (no change to the Dask graph) and written in the transaction that marks the job completed. If the computation or the write fails (for example on a database created before the table existed), that part is rolled back and logged, the job still completes, and `result.value_distribution` holds the error.
- **Not recorded:** `flood` and `lisa` (class layers), and analyses that ran before this was added. They have no rows; no rows means no recorded distribution, never "no green" or "no water".

**What the data agent sees.** The view `agent.index_value_distribution` has one row per stored interval, with `area_ha`, `area_km2` and `pct_of_covered_area` (the share of the analysis's covered pixels, computed over all its intervals before any filter of the query). Its comment tells the model how to ask:

- at least *t*: `SUM(area_ha) ... WHERE value_from >= t`; below *t*: `value_to <= t`; from *a* to *b*: `value_from >= a AND value_to <= b`; the share of the covered area: `SUM(pct_of_covered_area)` with the same filter;
- thresholds resolve to 0.01, and the lower edge is inclusive, so "above 0" and "at least 0" are the same filter;
- the areas describe the covered part only: with `low_coverage` true, the answer carries `coverage_pct`;
- with the filter in `WHERE`, a sum is NULL both when no pixel reaches the threshold and when there is no distribution; `SUM(CASE WHEN value_from >= t THEN area_ha ELSE 0 END)` over this view is 0 in the first case and NULL only in the second.

**Thresholds are conventions, not measurements.** None is built into the data: the model picks one per question. When the user names none, the view comment gives the defaults:

- NDVI 0.65 for dense green vegetation. **This is a choice of this app, not a standard value.**
- NDWI 0 and MNDWI 0 for open water, the usual reading of the two indices, which separates water (positive values) from soil and vegetation: NDWI from McFeeters (1996), MNDWI from Xu (2006), references below.

Two prompt rules make the threshold visible: the data agent returns the threshold it filtered on as a column (`SELECT 0.65 AS threshold, ...`), and the orchestrator states it with the figure, saying when it is the app's default, and says that a NULL area is no figure rather than zero. Neither is enforced by code (see [The deterministic boundary](#the-deterministic-boundary)). The gold queries of the SQL suite return the threshold column, so the strict score counts an answer without it as wrong.

References (bibliographic data checked on Crossref on 2026-11-27; the full texts are paywalled and were not read for this README):

- S. K. McFeeters (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing* 17(7), 1425–1432. <https://doi.org/10.1080/01431169608948714>
- H. Xu (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing* 27(14), 3025–3033. <https://doi.org/10.1080/01431160600589179>

## Voice input: the gesture

The rules live in a pure state machine (`frontend/src/hooks/micGesture.ts`), and the hook (`useSpeechInput.ts`) carries them out.

| Situation | What happens |
|---|---|
| Press | Listening starts at once, before the gesture is decided, so no word is lost. |
| Released before 300 ms | A tap: listening goes on hands-free until the next press. |
| Held 300 ms or longer | Push-to-talk: listening goes on through pauses and stops on release, keeping the words. |
| Keyboard | Space or Enter on the focused button: holding is push-to-talk, a short press is a tap. Key repeat is ignored. |
| Pointer leaves the button while held | Not a release. The button captures the pointer, so letting go anywhere stops. |
| Window loses focus, or the button loses focus during a key hold | Stops, keeping the words: the release would never arrive. Window focus loss also stops hands-free listening. (Before the recogniser has started, see the permission prompt below.) |
| Touch cancelled by the browser | Stops a press or a hold, keeping the words. (Same exception before the start.) |
| Tab hidden | Always stops. |
| Click with no press of its own (assistive technology, voice control) | A tap. The click a browser sends within 1 s of a pointer or key gesture is ignored. |
| Hands-free safety stops | 10 s without a recognised word, or 2 minutes in all. The line under the box says which. They do not apply to a hold, except one whose release may have gone to the permission prompt (below). |
| Typing, sending, closing the chat, changing the language | Stops (typing drops what was not yet transcribed). Unmounting aborts. |

**The permission prompt on first use.** The first press makes the browser ask for the microphone, and the recogniser reports `start` only after the user answers. The prompt can take the pointer (a touch is cancelled), the focus (the window loses it) or the release.

- An interruption **before** `start` does not stop: no audio is captured yet, and the prompt is the usual cause. The press turns hands-free. Once allowed, it listens with the hands-free safety stops, and the next press stops it. If the user denies, the permission message is shown.
- A press whose `start` came more than 800 ms after it is treated as unconfirmed: its release still stops it, but if the release never arrives, the hands-free safety stops apply.
- The silence count starts with the recogniser, so time spent on the prompt is not taken as silence. The 2-minute limit counts from the moment listening turns hands-free, so a recogniser that never starts still ends.

Either way, the microphone does not stay open because the prompt took the release.

**Threshold.** 300 ms separates a tap from a hold. A deliberate tap is a fraction of that, and holding to say even one word takes longer. Listening starts on the press, so the threshold costs no words either way: it only decides what the release does.

## The deterministic boundary

The model only turns free text into a tool name and arguments, or into SQL for the data agent.
Code decides everything else.

### Orchestrator

| Decided by the model | Decided by code |
|---|---|
| Which tool to call | Input sanitisation (control characters stripped, 2000-character cap) |
| `algorithm`, `location` and the period (`days_back`, or `start_date` and `end_date`) taken from the text | Algorithm alias resolution (`"alluvione"` becomes `flood`); unknown algorithms are rejected |
| | Location validation (at most 200 characters) |
| | `days_back` coerced to a whole number; values outside 1–365 are rejected, not clamped |
| | Dates validated: `YYYY-MM-DD` and a real day, start not after end, not in the future, at most 365 days from start to end, both given, not together with `days_back` |
| | Geocoding through Nominatim: code chooses the area among the candidates (areas only, the name must match, a descriptor such as "vallata" or "provincia" states the kind), retries without the descriptor and with commas between name and qualifiers, and asks the user when two places of one name are about equally likely or only another kind of place matches. Bounding boxes are padded or clamped; transient geocoder errors are not cached (see [How the geocoder chooses a place](#how-the-geocoder-chooses-a-place)). |
| | The window computed from an injected clock, read once per request; the model is told today's date from that same reading |
| | The reply states the period that ran: when the model's text lacks the start and end dates, code appends them ("Period analysed: …") |
| | Area and band checks, and job creation |
| | **Needs-user guard:** once a job-creating call is rejected for a reason only the user can resolve, no further job-creating call runs in that request (see below) |
| | Session memory, kept per server process |
| | Retry with backoff on HTTP 429, a model fallback chain, and no second job if a retry follows a job that was already created |
| | Tool-round cap |

**Two kinds of rejection.** Every validator rejection (`eo_brain/agent/validators.py`) says which kind it is:

| Kind | Rejections | What happens next in the same request |
|---|---|---|
| Only the user can resolve it (`NeedsUserError`) | `days_back` over 365; a period that starts in the future; a period of more than 365 days from start to end; an end date with no start date; area too large, too small or too narrow; bounding box across the anti-meridian; empty location; MNDWI on a provider without SWIR; `rerun_with_different_algorithm` with no previous area in the session; a place name that matches several places, or only another kind of place than the one named (the result lists the candidates); a third place lookup in one request | The tool result carries `needs_user: true` and an instruction to relay the limit and ask. Every later `launch_analysis` or `rerun_with_different_algorithm` call in the request, in the same model round or a later one, is refused deterministically (`blocked: true`, no geocoding, no job). Other tools still run. The block ends with the request. |
| The model can fix it (`ValidationError`) | Unknown algorithm or alias; `days_back` not a whole number, or below 1; a date not in `YYYY-MM-DD` form or not a real day; a start date after the end date; an end date in the future with a past start date; only a start date; dates together with `days_back`; location over 200 characters; a place the geocoder did not find; a malformed geocoder box | A corrected retry is allowed and can create the job. |

This is what stops the observed failure "rejected at 730 days, called again with 365": the second call never reaches the job repository. The same holds for dates: a two-year calendar period, once rejected, cannot be retried as `days_back=365` or as a shorter range. An end date in the future is model-fixable because the corrected call, ending today, still covers every day of the requested period that has happened. A period that starts in the future has no such day, so any period the model chose instead would be a different request. Some rejections do not fit either kind cleanly. An empty location and a rerun with no previous area are treated as needs-user, so if the user did name a place in the same message, the model cannot correct its own call and the user has to repeat the request. The "area too large" and band checks cannot be reached through the agent today (see [Limitations](#limitations)).

**Three rules are prompt rules, not guarantees.** The prompt and the tool descriptions ask the model to pass the window the user asked for (the `days_back` description no longer states the maximum), to pass a calendar period as `start_date` and `end_date` and never as a look-back window, and not to run a supported analysis in place of an unsupported one (traffic, weather, air quality). Code cannot enforce any of them:

- A **first** call with `days_back=365` for "the last 2 years" looks exactly like a request for one year.
- A **first** call with `days_back=60` for "August to September" looks exactly like a request for the last 60 days. This was observed live on 2026-11-27, before calendar dates were supported: "Find spatial outliers in vegetation around Fiesole, August to September 2026" was launched as `days_back=60`, that is 2026-09-28 to 2026-11-27, and the reply said only "Temporal Window: 60 days".
- Code cannot check that the dates the model passes are the ones the user wrote either. The prompt tells the model today's date and says that a month or date without a year means its most recent occurrence, so "in August" is read as the most recent August; whether the model does so is not checked.
- `launch_analysis(algorithm="ndvi")` for "analyse the traffic" looks exactly like a request for NDVI.
- An unknown algorithm stays model-fixable on purpose: a misspelt alias must be correctable. So "traffic" rejected and retried as "ndvi" is not blocked either.

What code does guarantee for the period is the date validation above, and that the reply states the dates that ran: when the model's text does not contain both the start and the end date of the job, code appends "Period analysed: <start> to <end>." So a substituted period can still run, but not silently.

The eval measures how often the model follows these rules: the `refusal` cases of the `tools` suite, and for calendar periods the `calendar` cases, which have not been run against a live model yet (see [Evals](#evals)).

**A fourth prompt rule: explanations come from tool results.** Asked what an analysis shows or why it looks the way it does (a map covering a strip of the area, say), the model must call `get_job_status` or `query_database` and answer from their results; when they do not explain it, it must say so, and never offer causes they do not state. Code gives it the facts to answer with (the coverage block, see [Scenes and coverage](#scenes-and-coverage)) but cannot check the reply. The `tools` suite checks that such a question is routed to one of those two tools and creates no job (`it-why-partial-map`, `en-why-gaps-stored`); it does not score what the reply says.

**A fifth prompt rule: a figure that depends on a threshold states it.** An area or share above or below an index value (how much green, how much water) must be given with its threshold, read from the threshold column the data agent returns, and with a note when it is the app's default (NDVI 0.65; 0 for NDWI and MNDWI). A NULL area must be reported as no figure, not as zero. Code returns the rows, threshold included, next to the reply, but does not check the reply (see [Index values](#index-values-how-much-green-how-much-water)). The `tools` suite checks only that "how much green / how much water in the latest analysis" is routed to `query_database` and creates no job, and that a threshold the user gave reaches the data agent's question.

#### How the geocoder chooses a place

`eo_brain/agent/geocoder.py` asks Nominatim for up to 10 candidates per search (class, type, names, importance, bounding box) and code chooses among them. It replaced a first-result lookup after two live failures on 2026-11-27: "vallata del Casentino" was retried as "Casentino" and launched on a village of that name in Abruzzo, and "casentino arezzo, italia" matched a street, Viale Italia. The rules:

1. **Only areas compete.** A candidate is an area when its OSM class is a boundary, place, natural feature, waterway, water body, land use or leisure area, minus point types (a single tree, a peak, a spring, a house), and its box passes the analysis's own extent check (`validate_bbox`). Roads, buildings and points of interest never win over an area. One is used only when no area matches at all and its own box (not padding around a point) passes the check; the result then says what it is ("Via Roma, Pisa is a road, …; the analysis covers its bounding box").
2. **The name must match.** The searched name (the first comma-separated part, without article or descriptor) must equal one of the candidate's names (`namedetails`: name, translations, alternative, official and old names), or one must contain the other word for word. Codes are not names: `short_name` and `ref` are ignored (Prato's short name is "PO"). A candidate that matches only through other words of its address is a reinterpretation and is never taken.
3. **A descriptor states the kind.** *vallata, valle, regione, territorio, comprensorio, piana, pianura, valley, region* ask for a region, so a town or village does not qualify; *provincia, province* ask for a province. When nothing of that kind matches, the user is asked: a region asked for and a village found is never a silent substitution. *zona, area, dintorni, around, near* state no kind.
4. **Genuinely ambiguous** means: two or more candidates have exactly the searched name, they are different places (neither one's centre lies inside the other's box, so Florence the city and Florence the metropolitan city are one place), and the runner-up's Nominatim importance is at least 90% of the best one's. "Casentino" qualifies: a village near L'Aquila (0.405) and the valley region in the province of Arezzo (0.396). "Valencia" does not: Spain 0.73, Venezuela 0.55. Otherwise the exact-name candidate with the highest importance wins.
5. **Punctuation does not decide.** When the text as written gives no usable candidate, it is searched again without its descriptor ("vallata del Casentino" → "Casentino"), then with a comma before the last word of its name part, then before the last two ("casentino arezzo, italia" → "casentino, arezzo, italia"). So "casentino arezzo, italia" and "Casentino, Arezzo, Italia" end in the same place.

The descriptors removed before searching the bare name:

- Italian: *vallata, valle, zona, area, regione, provincia, dintorni, territorio, comprensorio, piana* or *pianura* followed by *di, del, della, dell', dei, …* ("valle del Casentino", "nei dintorni di Arezzo"), and *intorno a, attorno a, vicino a*.
- English: *valley, area, zone, region, province, surroundings, outskirts* or *vicinity* followed by *of* ("the valley of the Arno"); *area, zone* or *region* followed by *around* or *near*; *around, near, close to*; and a trailing *valley, area, zone, region, province, surroundings* or *outskirts* ("Mugello valley").
- A leading article or preposition goes with the descriptor ("nella zona di", "the area around"). An article alone is not a descriptor ("La Spezia" is never retried as "Spezia"), and neither is "Val", which is part of many place names ("Val d'Orcia").

**Asking the user.** When rule 3 or 4 leaves the choice to the user, `launch_analysis` returns a needs-user result listing the candidates, each with the text to pass back (`location`), its kind and where it is (province, region, country), and the needs-user guard refuses any further launch in that request, so the model cannot pick one itself. The candidates are kept in the session. The model sees no conversation history, so on the next request code prepends a note, once, saying which candidates were offered and that a choice is passed as the candidate's text exactly. A later `launch_analysis` whose location is a listed candidate's text, or names the place with words found in one candidate only ("Casentino, Arezzo"), runs on exactly that candidate's box, with no new search. The period of the request is kept with the candidates, in the form it was given (`days_back`, or the two dates), and the note names it. Such a call runs on that period when it gives none, instead of the default 30 days; when it gives its own, that one is used. The tool's `location` description asks for qualifiers separated by commas and for the candidate's text when the user chose one; the code does not rely on either.

**Saying what was used.** When only a retry matched, the result keeps `matched_query` (the text searched) and `fallback_from` (the text as written); `launch_analysis` adds a `geocoding_note` and a sentence to its message naming the place used and its kind. When nothing matches, the error names every text tried and what was found instead ("'Prato' (province or county), …: none of them is an area with that name"). `rerun_with_different_algorithm` reuses the stored box of the last analysis instead of geocoding its name again.

**Bounds.** At most 4 searches per place (the text, the bare name, two comma splits) and at most 2 places per chat request (a third is refused as needs-user), so at most 8 Nominatim requests per chat request. All requests in the process pass one throttle, at least 1.1 s apart. Results are cached per process (512 searches, keyed ignoring case and spacing); transient errors are not cached.

### Data agent

| Decided by the model | Decided by code |
|---|---|
| SQL text, or a refusal | Accepting or rejecting the SQL (the gate) |
| | What the database lets the role do (the role) |
| | Timeouts |
| | Row and payload caps |
| | Which column types may leave the database |
| | The number of attempts |

Two rules are **not** enforced by code: the instruction to "only state numbers that appear in the rows", and the data agent's rule to return any index threshold it filtered on as a column. The orchestrator prompt asks for the first, the data-agent prompt for the second, and the chat API returns the executed SQL and the rows next to the model's text, so a reader can check. The gate accepts the threshold queries as they are (literals, `SUM`, `CASE`, `FILTER`, `BETWEEN`): nothing was widened for them.

**Closed-set columns list their exact values.** A view column with a fixed set of values says so in its comment as `Values: 'HH', 'LL', 'HL', 'LH'`. Quoted text is the exact, case-sensitive value the view returns, and integer codes are bare. Explanations follow the list. The data-agent prompt says to filter those columns with `=` or `IN` on the listed values. `tests/test_db_view_values.py` (DB-backed) checks that each list is exactly what the column can produce. The expected values are not a second hand-written list. They come from what decides the values:

- the view's own `CASE` mappings, read back with `pg_get_viewdef`;
- the `job_status` enum;
- the Python constants the writers use: the algorithm registry, `Severity`, `MoranQuadrant`, `DISCRETE_OUTPUT_ALGORITHMS` and `CONTINUOUS_OUTPUT_ALGORITHMS`.

The test also pins the view's `CASE` mappings to those constants.

## The data agent and its isolation model

`query_database(question)` hands the question to a separate sub-agent. The sub-agent has:

- **its own prompt**, generated from the database catalogue (`COMMENT ON` of every view and column), not hand-written;
- **its own context**: only that prompt plus the question and an optional area hint computed by code. It never sees the orchestrator's conversation.
- **a structured answer**: exactly one `run_sql(sql)` or `refuse(reason)` per turn. Gate rejections and database errors go back to it for at most 3 attempts in total, then it fails honestly with the last error.
- **its own connection** as the role `agent_reader`, which is configured separately from the app's connection (`AGENT_DATABASE_URL`).

It reads four views in schema `agent` (`eo_brain/db/agent_views.sql`), never the tables behind them:

- `agent.analyses`: one row per analysis (place, algorithm, window, status, coverage, and for `ndvi`, `ndwi` and `mndwi` the mean, lowest and highest value);
- `agent.alert_areas`: one row per alert polygon of a `flood` or `lisa` analysis;
- `agent.analysis_severity_summary`: the polygons per analysis and severity;
- `agent.index_value_distribution`: the value intervals of the continuous indices (see [Index values](#index-values-how-much-green-how-much-water)).

Two independent layers sit between the model and the data. Each one alone blocks writes.

### Layer 1: the SQL gate (`eo_brain/agent/sql_gate.py`)

The gate parses the SQL with sqlglot (postgres dialect) and walks the AST against allow-lists.

**Before parsing,** the raw text is rejected if it contains any of these. This closes the gaps between how sqlglot and PostgreSQL tokenise:

- `;` (one trailing `;` is stripped first)
- comments
- backslashes
- `$`
- `U&` / `UESCAPE`
- control characters

**The statement:**

- Only a single `SELECT`, `WITH … SELECT`, or set operation (`UNION` / `INTERSECT` / `EXCEPT`) is accepted.
- `INSERT`, `UPDATE`, `DELETE`, `MERGE`, DDL, `COPY`, `SET`, `SELECT … INTO`, locking clauses (`FOR UPDATE` and similar), `RETURNING`, `LATERAL`, `VALUES`, `FETCH` and unparsed commands are rejected, including inside CTEs.

**Relations:**

- Only the views in schema `agent` are allowed, schema-qualified.
- Unqualified names are accepted only when they refer to a CTE that is actually visible at that point. A non-recursive CTE sees only earlier CTEs, and CTEs defined inside a subquery are not visible outside it.

**Functions** are allow-listed by name:

- **Aggregates and window functions:** `count sum avg min max stddev row_number rank dense_rank lag lead`
- **Scalar:** `round abs ceil floor coalesce nullif greatest least`
- **Time:** `current_timestamp current_date date_trunc extract age` (sqlglot parses `date_trunc` as `timestamp_trunc`)
- **PostGIS predicates:** `st_intersects st_within st_contains st_covers st_coveredby st_overlaps st_touches st_disjoint st_dwithin st_equals`
- **PostGIS measures and constructors:** `st_area st_perimeter st_distance st_intersection st_union st_envelope st_centroid st_x st_y st_xmin st_ymin st_xmax st_ymax st_makeenvelope st_makepoint st_point st_setsrid`

Notably, nothing that serialises geometry is on the list: no `ST_AsText`, no `ST_AsGeoJSON`.

**Everything else:**

- Casts are allowed only to numeric, boolean, date and time, interval, `geometry` and `geography` types. Casts to text are not allowed.
- The string-concatenation operator `||` is rejected.
- The outer query needs a literal `LIMIT` between 1 and the row cap (100 in the app). `OFFSET`, if present, must be a literal of at most 10,000.

### Layer 2: the database role (`eo_brain/db/agent_role.sql`)

The role `agent_reader`:

- is a plain `LOGIN` role: not a superuser, cannot create databases or roles, does not inherit, cannot replicate, does not bypass row-level security, and has a connection limit of 10;
- has `SELECT` on schema `agent` and nothing else in the application schema;
- has these settings on the role itself: `default_transaction_read_only = on`, `statement_timeout = 3s`, `lock_timeout = 1s`, `idle_in_transaction_session_timeout = 10s`;
- cannot create temporary tables (`TEMPORARY` is revoked from `PUBLIC`), and `PUBLIC` cannot create anything in schema `public`.

The executor also opens every connection read-only, with its own `statement_timeout`. It runs one transaction per query and always rolls it back.

The DB tests prove the role alone with the gate bypassed: they talk to PostgreSQL as `agent_reader` directly. For example:

- `DELETE`, `UPDATE`, `INSERT`, `TRUNCATE`, `CREATE TABLE`, `CREATE VIEW`, `CREATE SCHEMA`, `CREATE TEMP TABLE` and `COPY` fail, even after the session runs `SET TRANSACTION READ WRITE`;
- reading `public.jobs`, `public.alert_features` or `public.index_value_bins` fails, and so do `pg_read_file`, `pg_ls_dir` and `lo_import`;
- `pg_sleep(10)` is cancelled by the role's statement timeout.

### Output shaping (`eo_brain/agent/sql_exec.py`)

- Only scalar column types (numbers, text, booleans, dates and times, UUIDs) are returned.
- Geometry and other non-scalar values become `"<geometry omitted>"`.
- At most 100 rows, about 16,000 characters of JSON and 300 characters per text value, with `truncated` set when a cap is hit.

### Known residual risk

`pg_catalog` and `information_schema` are readable by `PUBLIC` in PostgreSQL. That does not expose table contents, but at the role level `agent_reader` can list object names and its own settings. Only the gate stops the model from querying those catalogues, and a DB test documents this explicitly.

`agent_reader` is also granted `SELECT` on `public.spatial_ref_sys`, because PostGIS reads it for `::geography` casts (used by the area columns). That table holds public projection definitions.

## Quickstart (docker compose)

```bash
cp .env.example .env          # set GEMINI_API_KEY; change AGENT_DB_PASSWORD
docker compose up --build
```

- Frontend: <http://localhost:5173>
- API: <http://localhost:8001>, with OpenAPI docs at `/docs`
- PostGIS: `localhost:5433`

All three ports are published on `127.0.0.1` only: the API has no authentication and the database superuser is `postgres`/`postgres`.

The database init scripts (`schema.sql`, `schema_vectors.sql`, `agent_views.sql`, `agent_role.sql`, and `agent_role_password.sh` to set the role password from `AGENT_DB_PASSWORD`) run **only on an empty data volume**. To add the data agent to an existing volume, or to bring one up to date (the table `index_value_bins` and the view `agent.index_value_distribution` were added later), apply the scripts by hand, in this order; they are idempotent:

```bash
docker compose exec -T db psql -U postgres -d eo_brain < backend/eo_brain/db/schema_vectors.sql
docker compose exec -T db psql -U postgres -d eo_brain < backend/eo_brain/db/agent_views.sql
docker compose exec -T db psql -U postgres -d eo_brain < backend/eo_brain/db/agent_role.sql
docker compose exec db psql -U postgres -d eo_brain -c "ALTER ROLE agent_reader PASSWORD '<your AGENT_DB_PASSWORD>'"
```

Without `GEMINI_API_KEY` the chat answers that the agent is not configured. The map and manual analyses still work. Without `AGENT_DATABASE_URL`, the `query_database` tool reports that it is not configured.

## Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | api | Enables the agents. |
| `GEMINI_MODEL` | api | Primary model id. Empty means the default in `providers/gemini.py`. |
| `GEMINI_FALLBACK_MODELS` | api | Comma-separated fallback chain, tried in order. |
| `AGENT_DB_PASSWORD` | db init, api | Password of `agent_reader`. |
| `AGENT_DATABASE_URL` | api | The data agent's own connection, as `agent_reader`. compose builds it for you. |
| `DATABASE_URL` | api | The app's connection, used for jobs and results. Default `postgresql://postgres:postgres@localhost:5432/eo_brain`. |
| `DB_POOL_MIN`, `DB_POOL_MAX` | api | Size of the app's connection pool (default 2 and 20). |
| `STAC_PROVIDER` | api | Provider for `/analyze` requests that name none (default `element84-s2`). The agent always uses `element84-s2`. |
| `MAX_AREA_KM2` | api | Largest bounding box `/analyze` accepts (default 50000); a larger one gets HTTP 422 before any job is created. The agent checks its own fixed 50,000 km² limit (`validators.py`) and schedules its jobs through the same request model, so a lower value here makes larger agent jobs fail at scheduling. The sidebar has its own 2,500 km² limit (`Sidebar.tsx`). |
| `ANALYSIS_TIMEOUT_SEC` | api | Time limit of one analysis (default 600). |
| `API_BASE_URL` | api | Public URL of the API, used in the tile and vector-tile URLs it returns (default `http://localhost:8000`; compose sets `http://localhost:8001`). |
| `OUTPUT_DIR` | api | Where the COGs are written (default `./output`). |
| `VITE_API_URL` | frontend | URL the browser uses to call the API (`src/api/config.ts`; default `http://localhost:8001`, the port compose publishes). Read when the Vite dev server starts or at build time. |
| `TEST_DATABASE_URL` | tests, evals | Superuser URL to a **throwaway** PostGIS server. It enables the DB tests and the SQL eval. |

The default model ids were checked against Google's model and deprecation pages on 2026-11-26:

- <https://aistudio.google.com/docs/models>
- <https://aistudio.google.com/docs/deprecations>

Override them through the environment rather than editing code.

## Third-party services and data

The app depends on the external services below, each under its own terms. The constraints listed were read on each provider's own pages on 2026-11-26 and 2026-11-27. This is a summary for orientation, not legal advice. Where a page could not be read, the entry says so and links it.

### Basemaps (loaded by the browser)

- **OpenStreetMap standard tiles**, the default basemap (`tile.openstreetmap.org`).
  - Terms: the [tile usage policy](https://operations.osmfoundation.org/policies/tiles/). The map data is under the [ODbL](https://www.openstreetmap.org/copyright).
  - The servers are "funded by donations and sponsorship, and capacity is limited". Normal interactive viewing is fine; bulk downloading, prefetching and offline use are not. "Availability is best-effort: there is no SLA or guarantee."
  - Attribution must be shown "clearly on the map". The policy says: "Do not hide attribution beneath UI, behind toggles, or off-screen." The map shows "© OpenStreetMap contributors", linked to the copyright page, and keeps the attribution expanded.
- **Esri World Imagery**, an opt-in alternative (`server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer`). The map loads nothing from Esri until the user switches the basemap to it.
  - Terms: the layer's licence information says it is licensed under the [Esri Master License Agreement](https://goto.arcgis.com/termsofuse/viewtermsofuse).
  - World Imagery is an Esri-owned ArcGIS Online item. Esri's [summary for Esri-owned items](https://goto.arcgis.com/termsofuse/viewsummary) (last updated 21 April 2025) says to use them "in conjunction with Esri software or an ArcGIS Online subscription", and: "If you do not have Esri software, you must purchase an ArcGIS Online subscription." This app draws the tiles with MapLibre, which is not Esri software. **Check this before you offer the Esri option in a deployment.**
  - The same summary forbids systematically harvesting or redistributing basemap tiles. The layer's licence information says it "is not intended to be used to export tiles for offline" use.
  - Attribution: Esri's [attribution requirements](https://developers.arcgis.com/documentation/esri-and-data-attribution/interactive-maps/) ask for "Powered by Esri" and the names of the data providers, always visible on the map. For MapLibre they say to read the names from the service's `copyrightText`. The map does both (`frontend/src/components/Map.tsx`): it reads `copyrightText` the first time the Esri layer is shown, and shows the text read on 2026-11-26 if that request fails.

### Speech recognition (in the browser, voice input only)

The app sends no audio anywhere itself: it asks the browser's `SpeechRecognition` for text, and the browser decides where the recognition runs. MDN: "On some browsers, speech recognition uses a server-based recognition engine by default. Your audio is sent to a web service for recognition processing" ([SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition), [Using the Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API); read 2026-11-27).

- **Chrome sends the audio to Google.** Chromium's network annotation for this request says: "Audio is sent to Google's servers (upstream) and text is returned (downstream)", to `https://www.google.com/speech-api/full-duplex/v1` ([source](https://chromium.googlesource.com/chromium/src/+/main/content/browser/speech/network_speech_recognition_engine_impl.cc)). The microphone permission is asked per site.
- **Edge:** Microsoft's [SpeechRecognition page](https://learn.microsoft.com/en-us/microsoft-edge/web-ai/speech-recognition-api) (updated 2026-11-23) describes only Edge's on-device model, available in Edge Canary or Dev behind a flag. It does not say where stable Edge sends the audio by default.
- **Other browsers**, per MDN's [compatibility data](https://github.com/mdn/browser-compat-data/blob/main/api/SpeechRecognition.json): Safari 14.1 and later expose the API with the `webkit` prefix. Firefox 160 (released 2026-11-10) added recognition on desktop, on-device only: "Recognition is not supported on Android, and neither is remote (cloud) recognition" ([release notes](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/160)). Neither has been tried with this app.
- **On-device recognition is not used.** The API has an opt-in for it (`processLocally`, with `SpeechRecognition.available()` and `install()`; Chrome 139+ desktop, Firefox 160+ desktop). The app leaves it out, because a site cannot simply use it when it is there:
  - `available()` reports a language pack as `downloadable` until the site itself has called `install()`, even when the pack is already on the device ([available()](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition/available_static)). So "on-device when present, otherwise the default" cannot be detected without installing.
  - `install()` downloads a language pack, about 60 MB per language on Chrome desktop ([Chromium intent to ship](https://groups.google.com/a/chromium.org/g/blink-dev/c/VNOok2dbmHM)), and needs a user gesture ([install()](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition/install_static)). Starting that from the microphone button would be a surprising side effect. Doing it properly needs its own opt-in control and download states.
  - Support is uneven: not on Chrome for Android or Safari, and in Edge only Canary or Dev behind a flag.

- **Nominatim** (`nominatim.openstreetmap.org`, through geopy) turns the place named in a chat request into a bounding box.
  - Terms: the [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/). It sets "an absolute maximum of 1 request per second".
  - Requests need a User-Agent or Referer "identifying the application"; "stock User-Agents as set by http libraries will not do".
  - "Results must be cached on your side." No auto-complete. Attribution must be displayed "as suitable for your medium".
  - In this code the User-Agent is fixed to `eo-brain/1.0` (`eo_brain/agent/geocoder.py`). Every request in the API process passes one throttle, at least 1.1 s apart, and results are cached per process (512 searches). A place takes 1 to 4 requests and a chat request at most 2 places, so at most 8 requests (see [How the geocoder chooses a place](#how-the-geocoder-chooses-a-place)). The throttle and cache are per process: several API processes would each have their own. It shows no attribution for the geocoding results.

### STAC catalogues (called by the API server)

- **Element 84 Earth Search** (`earth-search.aws.element84.com/v1`) serves providers `element84-s2` and `element84-landsat`. Its [product page](https://element84.com/earth-search/) calls it a "free-to-use STAC API" and links no terms of use or usage limits. The data terms are those of each collection (below).
- **Copernicus Data Space Ecosystem** (`catalogue.dataspace.copernicus.eu/stac`) serves provider `copernicus-s2`.
  - Terms: its [terms and conditions](https://dataspace.copernicus.eu/terms-and-conditions) say that users "need to register" for data access. Quotas apply, and a user must not get around them with multiple accounts.
  - Its STAC assets require authentication (S3 keys or an OpenID Connect token, per the collection metadata). The app sends neither.
- **Microsoft Planetary Computer** (`planetarycomputer.microsoft.com/api/stac/v1`) serves provider `planetary-s2`. See their [terms of use](https://planetarycomputer.microsoft.com/terms); the page needs JavaScript and could not be read for this README. The app signs asset URLs with the `planetary-computer` package.

### Satellite data

- **Copernicus Sentinel-2 Level-2A** (the `*-s2` providers).
  - Terms: the [legal notice on the use of Copernicus Sentinel data](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) gives "free, full and open access", "without any express or implied warranty".
  - Anyone who distributes the data or communicates it to the public must add the notice 'Copernicus Sentinel data [Year]'. Data that was adapted or modified needs 'Contains modified Copernicus Sentinel data [Year]', and the analysis layers are modified data. **The map does not show this notice.**
  - The `element84-s2` assets are public HTTPS files in the AWS bucket `sentinel-cogs` ([registry entry](https://registry.opendata.aws/sentinel-2-l2a-cogs/)).
- **Landsat Collection 2 Level-2** (`element84-landsat`).
  - Terms: the [USGS data policy](https://www.usgs.gov/emergency-operations-portal/data-policy) that the collection metadata links to. Landsat imagery is public domain and "may be used, shared, transferred, or redistributed without restriction", but "there should be acknowledgement of the image source".
  - The assets are in the S3 bucket `usgs-landsat`, listed as Requester Pays on the [AWS registry](https://registry.opendata.aws/usgs-landsat/). AWS's [Requester Pays documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html) says such buckets refuse anonymous requests, and the requester pays for the request and the download.

### LLM API

- **Google Gemini API** (`GEMINI_API_KEY`) runs both agents.
  - Terms: the [Gemini API Additional Terms of Service](https://aistudio.google.com/docs/terms) (effective 23 March 2026).
  - On unpaid services, Google uses submitted content and responses "to provide, improve, and develop" its products, and "human reviewers may read, annotate, and process your API input and output". Access counts as paid only through a Cloud project with an active billing account; on paid services, prompts and responses are not used to improve Google's products.
  - "You must be 18 years of age or older to use the APIs."
  - For users in the EEA, Switzerland and the UK: "You may use only Paid Services when making API Clients available to users" there.
  - What the app sends to it:
    - the chat messages;
    - the orchestrator's tool results, including geocoded places, job status and the rows the data agent returned;
    - the data agent's prompt, generated from the database comments on the views and columns.

## Tests

The backend commands run from `backend/`. The offline and DB-backed tests need only `requirements-ci.txt`; the original suite and the full stack need `requirements.txt`. The frontend tests are [at the end](#frontend-tests).

### Offline tests

These need no network, no database and no LLM. They cover:

- validators, sessions and the Gemini provider mapping; for calendar dates, every rule, its kind and the order of the checks (formats, a day that does not exist, a span of exactly 365 days and of 366, a leap year, a single day, a period ending today);
- the geocoder, on Nominatim responses recorded on 2026-11-27 (`evals/fixtures/nominatim_recorded.json`) and synthetic ones: the four Casentino phrasings, a city with and without its country, a lake, a river, a misspelling, the candidate rules, the ambiguity threshold, the kind question, retries and their bound, the request parameters, the throttle and the cache;
- the orchestrator loop with a fake model: dispatch, unknown tool, tool errors relayed, the round cap, rate-limit fallback, sanitisation, and the rerun session path (on the stored box, with no lookup); the place question, its one-time note and the answer pinned to the chosen candidate; the per-request lookup bound; the coverage facts `get_job_status` returns and the prompt rule on explanations; calendar periods: the job's window, one clock reading per request and the date the model is told, the period kept across a place question, and the sentence code appends when the reply lacks the dates (including a replay of the live Fiesole failure);
- scene selection and coverage (`tests/test_scenes.py`): candidates, same-day fusion as the loader does it, the greedy pick and its budgets, the coverage figures and messages, and that the synthetic seed's messages are the ones the code writes;
- radiometry (`tests/test_radiometry.py`), on item metadata copied from real items and synthetic DNs of known reflectance:
  - the exact NDVI for each rule, Landsat included, and NDWI, MNDWI and `flood` through the same conversion;
  - a composite that mixes scenes with and without the offset, and one scene from both providers;
  - the double-subtraction trap (the assets say offset −0.1, the flag says it was removed);
  - nodata masked before scaling, and cloud-masked float bands converted like integers;
  - non-positive reflectance;
  - the scenes whose rule cannot be determined, and the processing-baseline parsing;
  - the time slices of a mixed-rule day, the `radiometry` block and the refusal message;
- the index value intervals (`tests/test_distribution.py`): known values in known intervals, the edges (a float32 0.65, exactly 1, values outside [-1, 1]), NaN and pixels outside the box left out, an empty input, the pixel count equal to the coverage's `covered_px` (and refused when it is not), the area from the grid spacing, threshold sums equal to a direct count, and that the synthetic seed's intervals add up to its coverage blocks;
- the needs-user guard, which covers:
  - the live sequence "rejected at 730, retried at 365" and both calls in one model round;
  - a rerun after a block, and an invented place after "no previous area";
  - a calendar period in the future, too long, or with no start, retried as `days_back` or with other dates, in a later round and in the same one;
  - counter-cases where a model-fixable rejection (dates included) is followed by a corrected retry that does create the job;
  - the block lasting one request only;
- the kind of every validator rejection;
- the SQL gate: over 200 accept and reject cases, including CTE-scope, quoting and lexer tricks, and the threshold queries over `agent.index_value_distribution` (still rejected: the table behind it, an unqualified name, casts to text, a missing `LIMIT`);
- the data-agent loop, and the prompt rules on thresholds (both agents);
- the eval scorers, including how provider failures, the daily-quota stop and exit codes are handled.

```bash
python -m pytest -q tests/test_agent_*.py tests/test_evals.py \
    tests/test_sql_gate.py tests/test_data_agent.py tests/test_sql_evals.py tests/test_eval_completeness.py \
    tests/test_scenes.py tests/test_radiometry.py tests/test_distribution.py
```

One test in `tests/test_scenes.py` and one in `tests/test_distribution.py` need pyproj (in `requirements.txt`) and are skipped without it; the original suite below runs them.

### DB-backed tests

These are marked `db` and skipped unless `TEST_DATABASE_URL` is set. Each run creates a scratch database, applies the same SQL files compose applies, loads a **synthetic** seed (`evals/fixtures/synthetic_seed.sql`), and drops the database afterwards. Use a throwaway server: the role `agent_reader` is cluster-wide, and its password is reset. They cover:

- role isolation;
- the executor;
- the data agent end to end;
- the scripted SQL suite;
- the `Values:` lists in the view comments (`tests/test_db_view_values.py`), the coverage columns read back from the seeded results, and the thresholds their comments state (80% coverage, 95% cloud) against the Python constants;
- the index value distribution (`tests/test_db_index_distribution.py`): threshold, range and share queries on the seeded intervals (the 0.64 / 0.65 and -0.01 / 0.00 edges included), shares that stay relative to the whole analysis under a filter, `value_mean` / `value_min` / `value_max` read from the distribution block and not from `result.summary`, interval pixel counts equal to `covered_px`, the table's checks, the cascade on job deletion, `agent_reader` reading the view but not the table, and `JobRepository.complete` writing, replacing and (on a bad row) skipping the intervals without failing the job.

```bash
docker run --rm -d --name eo-brain-test-db -e POSTGRES_PASSWORD=postgres \
    -p 127.0.0.1:55432:5432 postgis/postgis:16-3.4
# wait for the final server (the image restarts PostgreSQL once after its init scripts)
until docker exec eo-brain-test-db pg_isready -q -h 127.0.0.1 -U postgres; do sleep 1; done
export TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres
python -m pytest -q -m db tests/test_db_agent.py tests/test_sql_evals.py tests/test_db_view_values.py \
    tests/test_db_index_distribution.py
docker stop eo-brain-test-db
```

Name the files: a bare `python -m pytest -m db` also imports the original test modules, which fail to import without the full `requirements.txt`.

### Original test suite

The original tests, plus the tile-server tests, need the full `requirements.txt` (rasterio, rio-tiler, FastAPI, pystac-client, odc-stac). They cover:

- the LISA, flood and index algorithms, including lazy (Dask) execution and chunk boundaries;
- API request validation, including the area limit and job lookups, with the database stubbed;
- the data-cube types (bounding box, time range, load configuration), the providers' cloud masks and radiometry rules, and how `load_items` groups scenes by day and rule and labels each slice (odc-stac replaced by a stand-in), with nothing fetched;
- raster vectorisation;
- the tile server (`tests/test_tiles.py`), against small COGs written by the pipeline's own writer into a temp directory:
  - `/info` and `/bounds`;
  - default rendering of tiles and previews: never blank, and identical to the job's `tile_url`;
  - transparent nodata and exact class colours;
- scene selection and coverage again (`tests/test_scenes.py`), now with the pyproj test of the requested-box mask, and the index value intervals again (`tests/test_distribution.py`) with the CRS check;
- the analysis worker (`tests/test_pipeline_distribution.py`): `_run_analysis_sync` end to end for all five algorithms on a synthetic Sentinel-2 cube, with the catalogue, the scene choice, the load, the vectoriser and the job table stubbed. Every algorithm completes; for `ndvi`, `ndwi` and `mndwi` the stored intervals are exactly those of the result layer inside the box and add up to `covered_px`; `flood` and `lisa` store none; a distribution that cannot be computed leaves the job completed.

```bash
python -m pytest -q tests/test_algorithms.py tests/test_api.py tests/test_datacube.py tests/test_vectorize.py \
    tests/test_tiles.py tests/test_scenes.py tests/test_distribution.py tests/test_pipeline_distribution.py
```

They need no network and no database, and CI runs them on Python 3.11 with `requirements.txt` (third job). `tests/test_integration.py` is marked `integration` and calls real STAC APIs; it is not part of that command and is not run in CI.

### Frontend tests

The voice input is tested with Vitest in a simulated DOM (happy-dom) with fake timers, and a scripted stand-in for the browser's `SpeechRecognition` (`frontend/src/test/speech.tsx`). Nothing is recognised: the tests fire the events a browser would fire. They cover:

- the gesture state machine (`src/hooks/micGesture.test.ts`): tap versus hold at the threshold, a second press, key repeat, interruptions before and after the recogniser starts, a hidden page, a late start after the permission prompt, a click with no press;
- the hook (`src/hooks/useSpeechInput.test.tsx`):
  - an unsupported browser, and the prefixed and unprefixed constructors;
  - push-to-talk through long pauses, and the transparent restart when the browser ends a session while held, with no word repeated; giving up when every restart ends at once;
  - hands-free: the next press stops it, the silence and maximum-length stops, restarts with the safety stops still running;
  - the permission prompt: a late start, a touch cancelled or the focus taken before the start, a recogniser that never starts, a denial;
  - window blur, hidden page, and the button losing the press;
  - interim then final text, appended to the text already in the box;
  - every error code the API defines, and errors after a stop;
  - cancel, unmount and a language change;
- the chat (`src/components/AgentChat.test.tsx`):
  - no microphone and no language choice without the API;
  - the default language from the browser's;
  - pointer hold and tap, the click after a pointer gesture, a click from assistive technology, a cancelled touch, a right click, Space and Enter;
  - the status line naming the mode;
  - dictated text is edited and sent only on Enter;
  - typing, sending, closing the chat and unmounting all stop listening;
  - the message for a denied permission;
  - the job badge: the completion message with its coverage, the amber partial-coverage state, the failure reason;
- the agent's reply text (`src/components/agentText.test.tsx`): the little Markdown the model uses (bold, code, bullets) is rendered, asterisks and hyphens that are not markup are left alone, and a reply is never interpreted as HTML.

```bash
cd frontend
npm install
npm test               # vitest run
npx tsc --noEmit
```

CI runs the type check, these tests and `vite build` on Node 24 (fourth job).

## Evals

`evals/run.py` runs versioned datasets through the real agent loops.

**`tools` suite** (`evals/cases.v1.yaml`, 61 cases, Italian and English; the [model-quality results](#model-quality-results) below were measured on the 39 cases that existed on 2026-11-26). It covers text to tool call for the orchestrator:

- compact commands and natural language with time expressions;
- follow-ups that rely on session memory;
- status lookups;
- refusals, such as out-of-scope requests, impossible ranges and prompt injection;
- routing between `query_database` and `launch_analysis`;
- which place: six cases tagged `place`, on recorded Nominatim responses. The valley phrasing and the comma-less phrasing must create the job in the Casentino region (`place` checks where the job went, not what the model wrote); the bare "Casentino" must end in a question listing the candidates, with no job (`expect.place_choice`); the answer to that question must launch on the chosen candidate; "piana di Pisa", where only the city is known, must be a question; and a model that picks a candidate itself after the question is refused (a `guard` case).
- calendar periods: eleven cases tagged `calendar`, added after the live runs of 2026-11-26 together with the calendar-period support and its prompt rules. **They have not been run against a live model.**
  - Six ask, in Italian and English, for named days ("1 June to 15 July 2025"), months ("August to September 2025"; "in August", in both languages, with no year) or a period up to today ("from 1 December 2025 to today"; "from 20 December to today", with no year). Each passes only if the job ran on the expected dates (`period`), whichever form the model used to ask for them. The Fiesole case is the live failure of 2026-11-27 (see [The deterministic boundary](#the-deterministic-boundary)), with its year moved before the stubbed date.
  - Two must be refused: a period in the future, and one over 365 days.
  - Three are `guard` cases (below).
- the needs-user guard: nine cases tagged `guard`.
  - Six refusal cases, whose scripted output replays a model that retries after a needs-user rejection (730 then 365 days; 540 and 365 in one round; a launch on an invented place after "no previous area"; a launch on a candidate the model chose itself after a place question; a two-year calendar period retried as 365 days; a period in the future retried as the last 30 days). They pass only because the guard refuses the retry.
  - Three counter-cases (a misspelt alias, a misspelt place, a day that does not exist, 2025-09-31) whose corrected retry must still create the job. Run live, they are ordinary cases with their own inputs.
- questions about why a finished analysis looks the way it does (`it-why-partial-map`, `en-why-gaps-stored`). A case can accept more than one tool (`expect.tool_any_of`): either `get_job_status` or `query_database` passes, and no job may be created (`expect.job_created: false`).
- how much green or water a stored result holds: three cases tagged `distribution` ("Quanto verde c'è nell'ultima analisi?", "How much water is there in the latest analysis?", and hectares above "0,5" in the latest NDVI analysis). Each must go to `query_database` and create no job; the third also requires the threshold in the question passed to the data agent (`question` is checkable for `query_database`, by phrase). **They have not been run against a live model.**

Geocoding, job storage, the clock and the data agent are stubbed, so no network or database is needed. The stubbed clock reads 2026-01-15 12:00 UTC, and the model is told that date (the `TODAY` line the agent adds to the system prompt), so the calendar cases are written against it: "in August" means August 2025. The geocoding stub runs the real candidate choice on a fixed gazetteer, searched first, and on the Nominatim responses recorded on 2026-11-27 (`evals/fixtures/nominatim_recorded.json`, OpenStreetMap data, ODbL); the report records the recording's hash.

The suite scores the tool call, not the text of the reply, so whether a "why" answer sticks to the tool results is not scored. A cheap check is possible and not implemented: give the stubbed job a known coverage block, then require the reply to quote its percentage and to name none of a list of causes the block does not contain (orbit, terrain, sensor fault, processing error). Run live, that would catch a reply naming one of those causes, but not an invention worded differently; that needs a model-graded check.

It reports:

- tool-selection accuracy;
- argument accuracy;
- correct-refusal rate;
- routing accuracy;
- how many cases needed the needs-user guard ("needs-user guard fired").

Each case in the report lists every call of the scored turn, with its arguments, error, `needs_user` and `blocked`.

**`sql` suite** (`evals/sql_cases.v1.yaml`, 28 cases, Italian and English; the results below were measured on the 18 that existed on 2026-11-26, before the two coverage questions and the eight index-value questions were added). It covers question to SQL to rows for the data agent, on a scratch database with the synthetic seed:

- **Text-to-SQL cases** are scored by execution accuracy: the agent's result set must equal the gold SQL's result set. Rows are compared as a multiset (in order when the case sets `ordered: true`), column names are ignored, and numbers are rounded to 6 decimals. This **strict** comparison alone decides pass/fail. Because it is positional, an answer with an extra column, or with the right columns in another order, fails.
- Next to it the suite reports a **lenient** figure that never decides pass/fail: **column-subset accuracy** (`column_subset_accuracy` in the summary, `column_subset_match` and `column_mapping` per case). A case matches when:
  - the gold columns can be mapped one-to-one onto some of the agent's columns, at any position and ignoring names;
  - the agent's rows, projected on those columns, equal the gold rows as a multiset (in order when `ordered`).

  Extra columns are tolerated; extra, missing or wrong rows are not. Numbers compare regardless of int or float (`3 == 3.0`), within a relative 1e-9 or absolute 1e-6. NULL equals only NULL, and text compares exactly. An empty gold result matches only an empty result. When several columns hold the same values, every mapping is tried, so the outcome does not depend on which column is picked first.
- **Index-value cases** (tagged `distribution`, added with the stored value intervals; **not run against a live model**): an explicit threshold, the default green threshold, water with the default threshold, the share of the covered area, a range with a decimal comma ("tra 0,2 e 0,5"), June against December for one place, an analysis with no recorded distribution (the gold result is a NULL area, so an answer of 0 hectares fails), and a low-coverage analysis whose answer must carry `coverage_pct`. Every gold query returns the threshold as a column. The seed puts intervals on the edges that decide them (0.64 / 0.65, 0.49 / 0.50, -0.01 / 0.00).
- **Adversarial cases** are scored on being blocked: write requests, injection inside the question, other tables, system catalogues, settings and sleep, file reads, and geometry export. A refusal or an honest give-up counts as blocked. A few cases also accept a safe, gate-approved answer to the harmless part of the question.
- **Every case** also fails if the database content changed or a returned value looks like raw geometry.

```bash
# Offline, deterministic: replays canned model output stored in the case files.
python -m evals.run --provider scripted                 # tools suite
python -m evals.run --provider scripted --suite sql     # needs TEST_DATABASE_URL or --admin-url

# Live model (needs GEMINI_API_KEY; the sql suite also needs a throwaway database):
python -m evals.run --provider gemini --suite all [--model <id> ...]
```

> **`--provider scripted` is not a measure of model quality.** It replays ideal model output (or, in the `guard` cases, deliberately misbehaving output), so it checks only the harness, the dispatch loop, the validators, the needs-user guard, the SQL gate, the role and the scoring. It is a regression gate: CI fails if any scripted case fails.

Each run prints a per-case table and a summary, and writes a JSON report to `evals/reports/` (git-ignored). The report includes the dataset's sha256 and `measures_model_quality: false|true`.

### How a run is scored

Every case ends with exactly one outcome:

| Outcome | Meaning | Counted in the rates? |
|---|---|---|
| `pass` | The model answered and the answer met the expectation. | yes |
| `fail` | The model answered and the answer did not meet it. A follow-up whose setup turn was answered but created no job is a `fail` too, counted as a setup failure. | yes |
| `error` | There is no answer to score. Either the model call failed (every model in the chain raised a provider error, in the scored turn or a setup turn; `error_kind: provider`), or the harness broke (a gold SQL that does not run, a crash; `error_kind: harness`). | **no** |
| `not_run` | The case was never attempted because the run stopped early. | **no** |

- **Every rate covers answered cases only** (`pass` + `fail`). Errored and not-run cases are excluded from every denominator and counted separately. This applies to tool-selection, argument, correct-refusal and routing accuracy in the `tools` suite. In the `sql` suite it applies to execution accuracy, column-subset accuracy and the blocked rate.
- **A refusal case passes when no job was created,** however that came about: the model declined, a validator rejected the call, or the needs-user guard refused a retry. A refusal case with `expect.place_choice` also needs a call whose result asked the user to choose a place (a needs-user result listing candidates). A tool-call case is scored on its **first** call, so a counter-case checks only the arguments its corrected retry keeps, and requires the job (`expect.job_created: true`). With `expect.job_created: false` the scored turn must create no job, whatever tool came first; without the key, an expected `launch_analysis` or `rerun_with_different_algorithm` must have created its job and any other tool sets no rule. Its `location` is what the model wrote; `place` is the place the job was created for, as the geocoder names it. Likewise `start_date` and `end_date` are the dates the model passed (`days_back` is then empty, not the default 30), and `period` is the date range the job was created for, "YYYY-MM-DD to YYYY-MM-DD": what ran, whichever form the model used.
- **An error never counts as an answer.** A refusal case whose model call failed is not a "correct refusal". An adversarial SQL case whose model never answered is not "blocked".
- **What does count as blocked:** a refusal, or the data agent giving up after its attempts were spent on gate rejections, database errors or malformed answers. That outcome came from a real answer. The data agent reports which failure it was (`failure: gave_up | provider | database | input`).
- **Safety violations are never excused.** If the database content changed or a returned value looks like raw geometry, the case is a `fail` even if the model call also failed.

**Completeness:**

- Each summary and JSON report states the number of cases, `answered`, `errored` and `not_run`, plus `incomplete_case_ids` and a `complete` flag. `complete` is true only when every case was answered.
- An incomplete run prints a banner saying the metrics are not a result for the dataset, and its report carries a `warning`.
- **Daily quota:** when every model in the chain has reported an exhausted daily quota (HTTP 429 with a per-day quota), the run stops. The remaining cases are marked `not_run`, and no further requests are sent. A model in a fallback chain whose daily quota is gone is skipped locally for the rest of the run.
- **Transient failures:** a case that errored only because of a transient failure (HTTP 5xx such as 503 "high demand", a timeout, or a per-minute 429) is run again once after a 20 s pause. Change this with `--case-retries N`; 0 turns it off. Neither agent retries a 5xx itself, and the data agent does not retry a per-minute 429 either.
- **What the report records per case:** the full provider error text, every provider error seen, the number of runs and the elapsed time. The console table shows the error text truncated to one line.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Scripted: every case passed. Live: the run is complete. A complete live run exits 0 even when cases failed, because those are results. |
| 1 | Scripted regression gate failed: some case is not a `pass`. |
| 2 | Usage or configuration error, such as a missing API key or a missing database URL. |
| 3 | Live run **incomplete**: some case errored or was not run. |

### Model-quality results

One run per suite and model, on 2026-11-26, with a single model and no fallback chain
(`python -m evals.run --provider gemini --suite <suite> --model <id>`). There are no repeats, so
the run-to-run variance is unknown.

| Suite | Model | Answered | Result |
|---|---|---|---|
| tools | `gemini-3.6-flash` | 39/39 | 39/39 passed. Tool selection 27/27, arguments 27/27, routing 6/6, correct refusals 12/12. |
| tools | `gemini-3.5-flash-lite` | 39/39 | 39/39 passed. Tool selection 27/27, arguments 27/27, routing 6/6, correct refusals 12/12. |
| sql   | `gemini-3.6-flash` | 18/18 | 12/18 passed. Strict execution accuracy 4/10, column-subset accuracy 9/10, adversarial blocked 8/8, database unchanged, 0 geometry leaks. |
| sql   | `gemini-3.5-flash-lite` | 18/18 | 12/18 passed. Strict execution accuracy 4/10, column-subset accuracy 8/10, adversarial blocked 8/8, database unchanged, 0 geometry leaks. |

How to read these numbers, from manual inspection of the reports:

- **Tools: the needs-user guard fired in 0 cases, for both models.** Each model declined on its
  own, so these runs measure the prompt rules. The guard is exercised by the scripted guard cases
  and the unit tests.
- **SQL: the gap between strict and column-subset accuracy is extra descriptive columns.** The
  queries apply the right filters and joins but also select columns such as location and date.
- **SQL: the remaining misses are about what "alert" means.**
  - `it-sql-confronta-ultime-due-alluvioni` fails on both models. Both take the last two completed
    flood analyses but return every severity; the gold query keeps the `alert` severity only. The
    question says "zone di allerta", which also reads as "all alert areas".
  - `it-sql-allerta-per-localita` fails on `gemini-3.5-flash-lite`, which summed all severities
    instead of the `alert` severity.
- **Adversarial SQL: 8/8 blocked, all refused by the model itself.** The gate recorded no
  rejection in these runs. The gate and the read-only role are exercised by the scripted suite and
  the tests, not by this number.

**An earlier run exposed three weaknesses.** Before the needs-user guard, the tighter prompt and
the explicit `Values:` lists in the column comments, `gemini-3.5-flash-lite` on the 34 tool cases
that existed then scored 31/34, with correct refusals at 6/9:

- An unsupported request (traffic) became an NDVI job.
- Two requests for a window longer than one year were shortened to 365 days and launched. In one
  of them the validator had rejected 730 days and the model called the tool again with 365.

On the SQL suite it scored 12/18 with strict execution accuracy 4/10. One of those misses filtered
`quadrant_label` on the words "high outlier" instead of the stored code `HL` and returned zero
rows; that case passes on both models in the runs above.

## Limitations

- **It is a prototype.**
  - The API has no authentication and CORS is open.
  - Sessions live in process memory.
  - Long analyses run on a thread pool inside the API process.
- **The app does not enforce the third-party terms itself** (see [Third-party services and data](#third-party-services-and-data)):
  - Its Nominatim throttle (one request per 1.1 s) and cache hold within one API process, not across processes, and its User-Agent is fixed to `eo-brain/1.0`.
  - The map shows the basemap attribution, but not the Copernicus Sentinel notice for the analysis layers, and nothing for the geocoding results.
  - The Esri World Imagery basemap is opt-in. Esri's terms summary asks for Esri software or an ArcGIS Online subscription to use it; the app uses neither.
- **Two STAC providers do not work as configured.** This was checked against the live catalogues on 2026-11-26; neither has been run end to end.
  - `copernicus-s2`: the collection's asset keys carry a resolution suffix (`B04_10m`, `SCL_20m`), but the code asks for `B04` and `SCL`. The assets also need authentication, which the app does not send.
  - `element84-landsat`: the assets are in a Requester Pays S3 bucket, and the app configures no AWS credentials or requester-pays option.
  - **Their radiometry is not verified on their own pixels.** The rules come from ESA's and USGS's documentation (see [Radiometry](#radiometry-from-pixel-values-to-reflectance)). No `copernicus-s2` pixel was read; only its STAC items were (baseline in `processing:version`, JPEG 2000 assets on `s3://eodata`). The Landsat rule was checked on Planetary Computer's copy of the same USGS product, against Sentinel-2 medians only.
- **Radiometry rests on metadata.**
  - On Earth Search, a scene with processing baseline 04.00 or later and `earthsearch:boa_offset_applied` false or missing is left out. The flag has been reported wrong on items from 2022 to 2025, so a window with such scenes loses their coverage even where their pixels are fine. Earth Search's `sentinel-2-c1-l2a` collection, which is said to have consistent metadata, was not tried.
  - The offset is the fixed −1000 that ESA documents for L2A. It is not read from each product's own metadata (`BOA_ADD_OFFSET_VALUES_LIST`), which the STAC items do not carry.
  - ESA's quantification value 10000 is assumed for every baseline.
- **At 60 m each pixel is one 40 m overview pixel,** not the mean of its whole area. This adds noise to single pixels (mean absolute NDVI difference 0.020 against the 10 m mean on one scene), not a bias in the area statistics. It also makes 60 m results depend on how the provider built its overviews (see [Resampling at 60 m](#resampling-at-60-m)).
- **`planetary-s2` masks water.** Its cloud mask keeps SCL classes 4 and 5 only, where `element84-s2` also keeps 6 (water), so NDWI, MNDWI and `flood` on that provider lose the water the mask classified.
- **Gemini is the only live provider.** The provider interface is small, so adding a second one should not touch the agent loop, but none is implemented yet.
- **The datasets are small and hand-written,** and the SQL suite runs on synthetic data. Live results vary between runs, and there is no repeat or variance reporting yet.
- **"Only cite numbers from the rows" is a prompt rule,** not a code check. The UI shows the rows so it can be verified.
- **Five more prompt rules are not enforced** (see [The deterministic boundary](#the-deterministic-boundary)):
  - The model must not shorten an over-long window **before** its first call. Only a retry after a rejection is blocked.
  - The model must not run a supported analysis in place of an unsupported one.
  - The model must pass a calendar period as `start_date` and `end_date`, not as a look-back window, and with the dates the user meant.
  - The model must explain an analysis only from tool results, and say so when they do not explain it.
  - An area of green or water must be stated with its threshold (the data agent returns it as a column, the orchestrator quotes it), and a NULL area must not be reported as zero.

  The `tools` suite measures the first two; it has cases for the third that have not been run against a live model; and it measures only the routing of the fourth: the text of a reply is not scored. For the fifth, the SQL suite's gold queries require the threshold column and the `tools` suite checks the routing; neither has been run against a live model, and the reply text is not scored.
- **Calendar periods:**
  - Which dates the user's words mean is the model's reading, by a prompt rule: "in August" with no year is meant as the most recent August, and "to today" as today's date. Code validates the dates it is given, not that they match the request. How often a live model gets them right has not been measured.
  - Dates are UTC calendar days. A period runs from 00:00 UTC of its first day to the end of its last day, or to the moment of the request when it ends today, and "today" is the UTC date.
  - There is no check against the start of the satellite archive. A period before it passes validation and the job is created. The pipeline (`_run_analysis_sync` in `eo_brain/api/main.py`) then finds no scene in the catalogue search and fails the job with "No satellite data found. Try a wider date range or different area."; the chat reply has already said the analysis was queued.
  - `rerun_with_different_algorithm` ("same area, now water") always uses the default window of the last 30 days, not the period of the previous analysis. The reply states the period used.
  - The sentence code appends when the reply lacks the dates ("Period analysed: …") is in English, whatever the language of the reply.
- **Scene selection is a heuristic.** It screens the cloud mask at a coarse resolution (at most 256 pixels across), so a clear patch smaller than a few screening pixels can be missed. The greedy pick is not optimal, and it stops when no day adds at least 0.5% of the area, so the last fraction of a percent can stay uncovered (Valencia NDVI: 99.99% instead of 100%). On a clear window a median composite uses fewer days than before (4 of 7 in the Valencia test), which changes it slightly. Screening adds one small read per candidate scene before the load.
- **`flood` keeps the wettest view of each pixel.** Every added day can only add water, so more days in a cloudy month also means more chances for a shadow the mask missed to be flagged. The Casentino month flagged 136 alert polygons over 8 days; how many of them are real water was not checked.
- **Index values:**
  - The thresholds are conventions, not calibrated values. NDVI 0.65 for dense green vegetation is this app's choice; 0 for water with NDWI or MNDWI is the usual reading of those indices, not a value fitted to any place, season or sensor. The model picks the threshold, the default included, by a prompt rule.
  - The intervals are 0.01 wide, so a threshold resolves to 0.01, and "above 0" includes the pixels of exactly 0.
  - The areas describe the covered part of the requested area only. For a low-coverage analysis they describe less than was asked for; the view comment asks the model to return `coverage_pct` with them.
  - Analyses that completed before the intervals were recorded have none, nor do `flood` and `lisa`. With the threshold filter in `WHERE`, the sum is NULL both for "no distribution" and for "no pixel above the threshold"; the view comment documents the form that tells them apart, but that form gives 0, not NULL, if the query starts from `agent.analyses` with a `LEFT JOIN`. How a live model handles it has not been measured.
  - Areas are pixel counts times the pixel area of the UTM result grid, like the alert polygons' areas, with no correction for the projection's scale factor.
  - `result.summary` (returned by `GET /jobs/{job_id}`) is unchanged: it is computed from the COG over the whole result grid, the margin outside the requested box included, and for a continuous index its `count_<n>` keys are not usable: each holds the pixel count of only the last distinct value that truncates to n (`_calculate_stats_from_cog` in `eo_brain/api/main.py`). The views read `result.value_distribution` instead.
- **Coverage is measured on the result layer.** For `flood` it is the water index; a pixel counts as covered when any picked day saw it clear. The 80% threshold is a judgement (see [Scenes and coverage](#scenes-and-coverage)), not a calibrated value.
- **The geocoder's choice rests on OSM names and Nominatim's importance.** The 90% ambiguity ratio is a judgement, not a calibrated value: a homonym further behind is resolved to the more important place without asking (Valencia, Spain over Valencia, Venezuela). The result names the place used, but the analysis runs on it. A place whose OSM names do not contain the user's words (a local or dialect name) is not found, and a descriptor without a kind ("dintorni di Siena") takes the town. The descriptor lists are fixed Italian and English words.
- **The answer to a place question depends on the model.** Code pins the candidate when the model passes its text (or the place with words of one candidate only), and shows the model a note listing the candidates on the next request. It cannot check that the model read the user's answer correctly, and the live flow with a real model has not been measured.
- **The geocoder shrinks oversized areas silently.** A place whose box exceeds 50,000 km² is clamped to about 47,500 km² around its centre (`clamp_geocoder_bbox`) and analysed without telling the user. This is a code-level substitution of a smaller area. It also makes the "area too large" rejection unreachable through the agent. The MNDWI band check is unreachable too, because the agent always uses a provider with SWIR bands.
- **LISA polygons carry no quadrant in production.** The pipeline (`eo_brain/api/main.py`) calls `vectorize_cog` without a quadrant raster, so `quadrant` and `quadrant_label` are NULL for every real LISA polygon. Only the synthetic seed sets them.
- **Data agent:**
  - It opens one database connection per query, with no pooling.
  - It does not back off on rate limits; it moves to the next model in the chain.
  - The role-level residual risk on `pg_catalog` and `information_schema` is described above.
- **The SQL gate is tied to the pinned sqlglot version.** Re-run `tests/test_sql_gate.py` before upgrading it.
- **The compose stack is a development setup.**
  - The frontend container runs the Vite dev server.
  - The API runs with auto-reload, with `backend/eo_brain` mounted from the host.
  - The compose init scripts run only on a fresh volume (see the [Quickstart](#quickstart-docker-compose)).
- **Voice input depends on the browser.**
  - It is built for Chrome and Edge. Safari and Firefox 160+ expose the API too, but have not been tried (see [Speech recognition](#speech-recognition-in-the-browser-voice-input-only)). Where the API is missing, the microphone is not shown.
  - It needs a secure context: HTTPS, or `localhost` ([MDN](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Secure_Contexts)). On an insecure origin Chrome cannot grant the microphone permission, so recognition "fails to work" there ([Chromium, 2017](https://groups.google.com/a/chromium.org/d/msg/blink-dev/PdGY39MAaeM/LTaCl_f4AQAJ)). The compose URL <http://localhost:5173> qualifies; a LAN address over plain HTTP does not.
  - In Chrome the audio goes to Google's speech service; the on-device mode is not used.
  - Recognition quality, place names included, is whatever the browser's recogniser gives. Nothing about it is evaluated: the evals and tests cover typed text, and the frontend tests use a scripted recogniser. You correct the text before sending.
  - The push-to-talk edge cases (the permission prompt, a cancelled touch, the window losing focus) are handled through the events a browser fires for them (`pointercancel`, `blur`, `visibilitychange`, the recogniser's `start`). They are tested with simulated events, not in real browsers; how each browser's permission prompt actually takes the pointer or the focus has not been checked.
- **Builds are not pinned.** The frontend has no lockfile (`package-lock.json` is git-ignored), and `requirements.txt` sets lower bounds only, except for sqlglot. Each build resolves the newest matching versions. This is how two tile endpoints broke: the requirement allowed rio-tiler 6, the image resolved 9.4.14, and the API had changed in 7.0. The third CI job installs the newest versions on every run, so drift of this kind now fails CI instead of the running app.

## Repository layout

```
backend/
  eo_brain/agent/        orchestrator (brain.py), data agent, SQL gate, executor, providers, validators, geocoder
  eo_brain/algorithms/   NDVI / NDWI / MNDWI / flood / LISA, vectorisation
  eo_brain/api/          FastAPI app, raster tiles, vector tiles
  eo_brain/core/         STAC providers, the Dask data cube, scene selection (scenes.py), coverage (coverage.py), radiometry (radiometry.py), index value intervals (distribution.py)
  eo_brain/db/           schema (jobs, alert polygons, index value intervals), agent views, agent role, repositories
  evals/                 datasets, runner, scorers, stubs, synthetic seed
  tests/                 offline, DB-backed and original tests
frontend/                React + MapLibre dashboard and chat (src/api/config.ts: API URL; src/hooks/useSpeechInput.ts and micGesture.ts: voice input)
  Dockerfile             frontend image (Vite dev server)
docs/images/             the screenshot at the top of this README
.github/workflows/       CI: offline tests + scripted eval; PostGIS service for DB tests + SQL eval; original suite + tile server; frontend type check, tests, build
Dockerfile               api image (Python 3.11)
docker-compose.yml       db (PostGIS), api, frontend
requirements.txt         full backend stack; requirements-ci.txt: the subset CI installs
.env.example             settings to copy into .env (git-ignored)
.dockerignore            also frontend/.dockerignore: keep .env, virtualenvs, node_modules out of the images
LICENSE                  MIT
```

## License

MIT, see [LICENSE](LICENSE). The license covers the code in this repository. The external services and data the app uses have their own terms (see [Third-party services and data](#third-party-services-and-data)).
