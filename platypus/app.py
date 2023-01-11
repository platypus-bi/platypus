from pathlib import Path
import json
import requests
import datetime

BORIS_BRW_BASE = "https://www.opengeodata.nrw.de/produkte/infrastruktur_bauen_wohnen/boris/BRW"
BORIS_BRW_JSON_INDEX = f"{BORIS_BRW_BASE}/index.json"

BORIS_IRW_BASE = "https://www.opengeodata.nrw.de/produkte/infrastruktur_bauen_wohnen/boris/IRW"
BORIS_IRW_JSON_INDEX = f"{BORIS_IRW_BASE}/index.json"


def download_brw():
    # Download the index.json file using requests
    r = requests.get(BORIS_BRW_JSON_INDEX)
    # Convert the JSON string to a Python dictionary
    index = json.loads(r.text)

    print(index["datasets"])
    for dataset in index["datasets"]:
        if dataset["name"] == "BRW_aktuell":
            # Aktueller Datensatz, keine Jahreszahl im Namen
            download_brw_aktuell(dataset)
        elif dataset["name"] == "BRW_historisch":
            # Alte Datensätze, Jahreszahl im Namen
            pass
        else:
            print("Unbekannter Datensatz", dataset["name"])


def download_large_file(url: str, name: Path):
    # NOTE the stream=True parameter
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(name, "wb") as file:
            # 10 MB chunk size
            for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                file.write(chunk)


def download_brw_aktuell(dataset: dict):
    print("Datensatz herunterladen:", dataset["name"])
    for file in dataset["files"]:
        print("Datei von:", file["timestamp"])
        year = datetime.datetime.fromisoformat(file["timestamp"]).year
        print("Datei für das Jahr:", year)
        print("Datei herunterladen:", file["name"])

        download_dir = Path(".") / str(year) / "BRW"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_large_file(
            f"{BORIS_BRW_BASE}/{file['name']}", download_dir / file["name"])
        print("Datei herunterladen:", file["name"], "fertig")


def main():
    download_brw()

    # Download the index.json file using requests
    r = requests.get(BORIS_IRW_JSON_INDEX)
    # Convert the JSON string to a Python dictionary
    index = json.loads(r.text)

    print(index["datasets"])


if __name__ == "__main__":
    main()
