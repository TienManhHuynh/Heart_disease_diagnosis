# Data and acknowledgments

## Cleveland Heart Disease

`data/raw/cleveland.csv` is the local, headerless Cleveland subset used by this
project: 303 records, 13 input features and original target values 0-4.
The loader converts the target to binary at runtime; the CSV is kept unchanged.

- Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).
- Creators credited by UCI: Andras Janosi, William Steinbrunn, Matthias Pfisterer
  and Robert Detrano.
- Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
  as listed on the UCI dataset page.
- Reference SHA-256: `70a023033aaf840596737ad250eea055762ad825036ac73b5b29168dc638c7c2`.

This data notice applies to the dataset; it does not grant rights to third-party
course PDFs or establish a license for other material in the repository.

## Learning context

The project was developed as an AI course assignment using the course Description
and Slide-v1 documents. Those documents are credited in [docs/SOURCES.md](docs/SOURCES.md)
and are not included in the source distribution. Implementation API references
are listed there as well.
