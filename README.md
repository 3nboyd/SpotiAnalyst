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
