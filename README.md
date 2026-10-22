# SpotiAnalyst

Compare popular Spotify tracks with a listener's streaming history through eight charts of stream counts, listening time, and overlap. Created by Noah Boyd and Jack Bertram for a data visualization course project.

The website is static HTML and CSS. `visuals.py` generates the PNG charts shown in `index.html` from the two bundled CSV files.

## View the site

From the repository root, run:

```sh
python3 -m http.server 8000
```

Open [localhost:8000](http://localhost:8000). The existing charts are ready to view.

## Regenerate charts

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python visuals.py
```

Run these commands from the repository root, then refresh the page. Chart images are written to `img/`; the CSV inputs remain unchanged.

## Data sources

The bundled examples use [Spotify Most Streamed Songs from Kaggle](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs) and [Spotify Streaming History from Maven Analytics](https://mavenanalytics.io/data-playground/spotify-streaming-history). The website includes the data-source credits and chart interpretations.

## Input columns

| File | Columns used by the chart script |
| --- | --- |
| `moststreamedsongs.csv` | `track_name`, `artist_name` (or `artist(s)_name`), `streams`, and `release_year` (or `released_year`) |
| `spotify_history.csv` | `track_name` (or `trackName`), `artist_name` (or `artistName`), `ms_played` (or `msPlayed`), and `end_time` (or `endTime`, `timestamp`, or `ts`) |

Listening time is supplied in milliseconds. The script parses timestamps to group listening by hour. Streams and listening times are converted to numbers before plotting.

To use other data, prepare CSV files with these columns and the same filenames, then rerun `visuals.py`. Export formats may need conversion to this schema first. Personal listening history stays in your local copy unless you publish it; keep it out of commits and use the bundled sample for public demonstrations.
