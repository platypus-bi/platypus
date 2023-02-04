import datetime
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, NamedTuple, Any, Optional

import numpy as np
import pandas as pd
import pyodbc
import requests
from pandas import DataFrame

BORIS_BRW_BASE = "https://www.opengeodata.nrw.de/produkte/infrastruktur_bauen_wohnen/boris/BRW"
BORIS_BRW_JSON_INDEX = f"{BORIS_BRW_BASE}/index.json"

BORIS_IRW_BASE = "https://www.opengeodata.nrw.de/produkte/infrastruktur_bauen_wohnen/boris/IRW"
BORIS_IRW_JSON_INDEX = f"{BORIS_IRW_BASE}/index.json"

ZIP_NAME_PATTERN = re.compile(r"(?P<type>[A-Z]{3})_(?P<year>\d{4}).*?\.zip")

Year = int
DatasetType = str
Datasets = dict[DatasetType, dict[Year, datetime.datetime]]

Column = NamedTuple("Column", [("name", str), ("conversion", Callable[[str], Any])])


def identity(value):
    return value


def parse_float(value) -> Optional[float]:
    if value:
        return float(value.replace(",", "."))
    else:
        return None


def parse_int(value) -> Optional[float]:
    if value:
        return int(value)
    else:
        return None


# Columns to insert into the database
COLUMNS: list[Column] = [
    Column("JAHR", identity),
    Column("BEZUG", identity),
    Column("BRKE", int),
    Column("ENTW", identity),
    Column("FARBE", int),
    Column("FLAE", identity),
    Column("GBREI", parse_int),
    Column("GEZ", identity),
    Column("GFZ", parse_float),
    Column("GRZ", parse_float),
    Column("GTIE", parse_int),
    Column("HBRW", parse_float),
    Column("NUTA", identity),
    Column("WHNLA", parse_int),
    Column("IMRW", int),
    Column("IRKE", int),
    Column("TEILMA", int),
    Column("OBJGR", parse_int),
    Column("GART", parse_int),
    Column("EGART", parse_int),
    Column("BJ", int),
    Column("WHNFL", identity),
    Column("AKL", parse_int),
    Column("MTYP", parse_int),
    Column("KELLER", parse_int),
    Column("DGA", parse_int),
    Column("GESLA", parse_int),
    Column("BK", parse_int),
    Column("BEHING", parse_int),
    Column("RANZ", parse_int),
    Column("WHNA", identity),
    Column("ANZG", identity),
    Column("GRDA", parse_int),
    Column("FLAE_2", identity),
    Column("MIETS", parse_int),
    Column("IRWTYP", int),
    Column("BRWTYP", parse_int),
    Column("NUZFL", identity),
    Column("IMMIS", parse_int),
    Column("DENKS", parse_int),
    Column("VERAN", parse_int),
    Column("BVERU", parse_int),
    Column("WART", parse_int),
    Column("GSTAND", parse_int),
    Column("MGRAD", parse_int),
    Column("GARSTP", parse_int),
    Column("ANZEGEB", identity),
    Column("TAGBAD", parse_int),
    Column("GANU", parse_int),
    Column("OPTIK", parse_int),
    Column("ALTERJ", parse_int),
    Column("WFNF", identity),
    Column("AUFZUG", parse_int),
    Column("FARBE_2", parse_int),
    Column("x", parse_float),
    Column("y", parse_float),
    Column("WKT", identity),
]

# Make sure that print is always flushed
original_print = print


# noinspection PyShadowingBuiltins
def print(*args, **kwargs):
    kwargs["flush"] = True
    original_print(*args, **kwargs)


def retrieve_downloaded_datasets(dataset_type: DatasetType) -> Datasets:
    connection = connect_to_database()
    cursor: pyodbc.Cursor
    with connection.cursor() as cursor:
        cursor.execute("USE [BigData]")
        cursor.execute("SELECT * FROM [Datenbestand] WHERE [Datenbestand].[Typ] = ?", dataset_type)

        rows = cursor.fetchall()
        datasets = {}
        for row in rows:
            year = row[0]
            dataset_type = row[1]
            last_updated = row[2].replace(microsecond=0)
            if dataset_type not in datasets:
                datasets[dataset_type] = {}
            datasets[dataset_type][year] = last_updated
        return datasets


def connect_to_database(**kwargs) -> pyodbc.Connection:
    return pyodbc.connect("DSN=MSSQLServerDatabase",
                          user="sa",
                          password=os.environ["MSSQL_SA_PASSWORD"],
                          **kwargs)


def download_large_file(url: str, name: Path):
    # NOTE the stream=True parameter
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(name, "wb") as file:
            # 10 MB chunk size
            for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
                file.write(chunk)


def unpack_file(path: Path):
    # Unpack the zip file using the zipfile module
    # Unpack into a folder with the same name as the zip file (without the .zip extension)
    # Only unpack files that are not PDF, TXT and XLS files

    output_folder = path.parent / path.stem

    with zipfile.ZipFile(path, "r") as zip_file:
        for file in zip_file.namelist():
            if not file.endswith(".pdf") and not file.endswith(".txt") and not file.endswith(".xls"):
                zip_file.extract(file, output_folder)
    # Delete the zip file
    path.unlink()


def determine_latest_year(dataset_type: DatasetType, datasets: dict) -> Year:
    latest_year = 0
    for dataset in datasets:
        if dataset["name"] == f"{dataset_type}_historisch":
            files = dataset["files"]
            for file in files:
                filename = file["name"]
                match = ZIP_NAME_PATTERN.fullmatch(filename)
                current_year = int(match["year"])
                if latest_year < current_year:
                    latest_year = current_year
    if latest_year == 0:
        latest_year = datetime.datetime.now().year
    else:
        latest_year += 1
    return latest_year


def determine_year(filename: str) -> Year:
    match = ZIP_NAME_PATTERN.fullmatch(filename)
    return int(match["year"])


def download_datasets(url: str,
                      dataset_type: DatasetType,
                      current_datasets: Datasets,
                      current_callback: Callable[[Year, Datasets, dict, list[Year]], None],
                      history_callback: Callable[[Datasets, dict, list[Year]], None]) -> list[Year]:
    r = requests.get(url)
    index = json.loads(r.text)

    downloaded_years = []

    latest_year = determine_latest_year(dataset_type, index["datasets"])

    for dataset in index["datasets"]:
        if dataset["name"] == f"{dataset_type}_aktuell":
            # Aktueller Datensatz, keine Jahreszahl im Namen
            current_callback(latest_year, current_datasets, dataset, downloaded_years)
        elif dataset["name"] == f"{dataset_type}_historisch":
            if os.environ["ENABLE_HISTORIC_DATASETS"] == "1":
                # Alte Datensätze, Jahreszahl im Namen
                history_callback(current_datasets, dataset, downloaded_years)
            else:
                print("Historische Datensätze werden nicht heruntergeladen")
        else:
            print("Unbekannter Datensatz", dataset["name"])

    return downloaded_years


def download_brw(current_brw_datasets: Datasets) -> list[Year]:
    return download_datasets(BORIS_BRW_JSON_INDEX,
                             "BRW",
                             current_brw_datasets,
                             download_brw_aktuell,
                             download_brw_historisch)


def download_irw(current_irw_datasets: Datasets) -> list[Year]:
    return download_datasets(BORIS_IRW_JSON_INDEX,
                             "IRW",
                             current_irw_datasets,
                             download_irw_aktuell,
                             download_irw_historisch)


def save_current_dataset(dataset_type: DatasetType, year: Year, timestamp: datetime.datetime):
    with connect_to_database() as connection:
        cursor: pyodbc.Cursor
        with connection.cursor() as cursor:
            cursor.execute("USE [BigData]")
            cursor.execute("SELECT * FROM [dbo].[Datenbestand] WHERE [TYP] = ? AND [JAHR] = ?", (dataset_type, year))
            if cursor.fetchone():
                cursor.execute("UPDATE [dbo].[Datenbestand] SET [AKTUALISIERT] = ? WHERE [TYP] = ? AND [JAHR] = ?",
                               (timestamp, dataset_type, year))
            else:
                cursor.execute(
                    "INSERT INTO [dbo].[Datenbestand] ([TYP], [JAHR], [AKTUALISIERT]) VALUES (?, ?, ?)",
                    (dataset_type, year, timestamp))
            cursor.commit()


def download_historisch(base_url: str,
                        dataset_type: DatasetType,
                        current_datasets: Datasets,
                        dataset: dict,
                        downloaded_years: list[Year]):
    print("Datensatz herunterladen:", dataset["name"])
    for file in dataset["files"]:
        year = determine_year(file["name"])
        print("Datei von:", file["timestamp"])
        timestamp = datetime.datetime.fromisoformat(file["timestamp"])
        timestamp = timestamp.replace(microsecond=0)
        print("Datei für das Jahr:", year)
        print("Datei prüfen:", file["name"])
        print("Datei aktualisiert:", timestamp.isoformat())

        datasets_by_year = current_datasets.get(dataset_type)
        if datasets_by_year is not None:
            timestamp_for_year = datasets_by_year.get(year)
            if timestamp_for_year is not None:
                print("Datei bereits vorhanden:", timestamp_for_year.isoformat())
                if timestamp_for_year >= timestamp:
                    print("Datei noch aktuell, überspringe...")
                    return

        print("Datei herunterladen:", file["name"])

        download_dir = Path(".") / str(year) / dataset_type
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / file["name"]
        download_large_file(f"{base_url}/{file['name']}", download_path)
        print("Datei herunterladen:", file["name"], "fertig")
        print("Entpacke:", file["name"])
        unpack_file(download_path)
        print("Datei entpackt und bereinigt:", file["name"])
        save_current_dataset(dataset_type, year, timestamp)
        downloaded_years.append(year)


def download_aktuell(base_url: str,
                     dataset_type: DatasetType,
                     latest_year: Year,
                     current_datasets: Datasets,
                     dataset: dict,
                     downloaded_years: list[Year]):
    print("Datensatz herunterladen:", dataset["name"])
    for file in dataset["files"]:
        print("Datei von:", file["timestamp"])
        timestamp = datetime.datetime.fromisoformat(file["timestamp"])
        timestamp = timestamp.replace(microsecond=0)
        print("Datei für das Jahr:", latest_year)
        print("Datei prüfen:", file["name"])
        print("Datei aktualisiert:", timestamp.isoformat())

        datasets_by_year = current_datasets.get(dataset_type)
        if datasets_by_year is not None:
            timestamp_for_year = datasets_by_year.get(latest_year)
            if timestamp_for_year is not None:
                print("Datei bereits vorhanden:", timestamp_for_year.isoformat())
                if timestamp_for_year >= timestamp:
                    print("Datei noch aktuell, überspringe...")
                    return

        print("Datei herunterladen:", file["name"])

        download_dir = Path(".") / str(latest_year) / dataset_type
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / file["name"]
        download_large_file(f"{base_url}/{file['name']}", download_path)
        print("Datei herunterladen:", file["name"], "fertig")
        print("Entpacke:", file["name"])

        # Old name: BRW_EPSG25832_Shape.zip
        # New name: BRW_2022_EPSG25832_Shape.zip
        download_path = download_path.rename(download_dir / f"{dataset_type}_{latest_year}_EPSG25832_Shape.zip")
        unpack_file(download_path)
        print("Datei entpackt und bereinigt:", file["name"])
        save_current_dataset(dataset_type, latest_year, timestamp)
        downloaded_years.append(latest_year)


def download_brw_aktuell(latest_year: Year, current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    download_aktuell(BORIS_BRW_BASE, "BRW", latest_year, current_datasets, dataset, downloaded_years)


def download_brw_historisch(current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    download_historisch(BORIS_BRW_BASE, "BRW", current_datasets, dataset, downloaded_years)


def download_irw_aktuell(latest_year: Year, current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    download_aktuell(BORIS_IRW_BASE, "IRW", latest_year, current_datasets, dataset, downloaded_years)


def download_irw_historisch(current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    download_historisch(BORIS_IRW_BASE, "IRW", current_datasets, dataset, downloaded_years)


def run_sql_script(cursor: pyodbc.Cursor, script: str):
    with open((Path("/app/sql") / script).with_suffix(".sql")) as f:
        try:
            for statement in f.read().split("---"):
                cursor.execute(statement)
            cursor.commit()
        except pyodbc.ProgrammingError as e:
            cursor.rollback()
            raise e


def initialize_database():
    with connect_to_database(autocommit=True) as connection:
        with connection.cursor() as cursor:
            run_sql_script(cursor, "01-create-database")
            run_sql_script(cursor, "02-create-table-types")
            run_sql_script(cursor, "03-create-table-stockpile")
            run_sql_script(cursor, "04-create-table-data")


def process_years(years: set[Year]):
    brw_datasets = retrieve_downloaded_datasets("BRW")
    irw_datasets = retrieve_downloaded_datasets("IRW")

    brw_datasets = brw_datasets.get("BRW")
    if brw_datasets is None:
        return

    irw_datasets = irw_datasets.get("IRW")
    if irw_datasets is None:
        return

    Path("intersection").mkdir(parents=True, exist_ok=True)

    for year in years:
        if year in brw_datasets and year in irw_datasets:
            output_dir = Path("intersection") / str(year)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = output_dir.absolute()
            output_file = str(output_dir / f"intersection_{year}.shp")

            brw_path = Path(".") / str(year) / "BRW" / f"BRW_{year}_EPSG25832_Shape" / f"BRW_{year}_Polygon.shp"
            brw_path = brw_path.absolute()
            irw_path = Path(".") / str(year) / "IRW" / f"IRW_{year}_EPSG25832_Shape" / f"IRW_{year}_Polygon.shp"
            irw_path = irw_path.absolute()

            print(f"Verarbeite {year}...")
            print("Erzeuge Schnittmenge...")
            now = datetime.datetime.now()
            subprocess.run([
                "qgis_process.bin",
                "run",
                "native:intersection",
                f"INPUT={brw_path}",
                f"OVERLAY={irw_path}",
                f"OUTPUT={output_file}"
            ])
            print(f"Erzeugung der Schnittmenge von {year} abgeschlossen in",
                  (datetime.datetime.now() - now).seconds,
                  "Sekunden")

            # Convert polygons to centroids
            print(f"Erstelle Zentroide für {year}...")
            centroid_file = str(output_dir / f"centroids_{year}.shp")
            now = datetime.datetime.now()
            subprocess.run([
                "qgis_process.bin",
                "run",
                "native:pointonsurface",
                f"INPUT={output_file}",
                f"OUTPUT={centroid_file}",
            ])
            print(f"Zentroide für {year} erstellt in",
                  (datetime.datetime.now() - now).seconds,
                  "Sekunden")

            # Add X, Y (latitute, longitude) to CSV
            print(f"Füge Breitengrad und Längengrad für {year} hinzu...")
            now = datetime.datetime.now()
            lat_long_file = str(output_dir / f"lat_long_{year}.shp")
            subprocess.run([
                "qgis_process.bin",
                "run",
                "native:addxyfields",
                f"INPUT={centroid_file}",
                f"OUTPUT={lat_long_file}",
                "CRS=EPSG:4326"
            ])
            print(
                f"Breitengrad und Längengrad für {year} hinzugefügt in",
                (datetime.datetime.now() - now).seconds,
                "Sekunden")

            # Create CSV
            print(f"Erstelle CSV für {year}...")
            intersection_csv_file = output_dir / f'intersection_{year}.csv'
            now = datetime.datetime.now()
            subprocess.run([
                "qgis_process.bin",
                "run",
                "native:savefeatures",
                f"INPUT={lat_long_file}",
                f"OUTPUT={intersection_csv_file}",
                "LAYER_OPTIONS=GEOMETRY=AS_WKT",
            ])
            print(f"CSV für {year} erstellt in",
                  (datetime.datetime.now() - now).seconds,
                  "Sekunden")

            # Clean up
            # This requires deleting all .shp files and their .cpg, .dbf, .prj, .shx files as well
            for file in output_dir.glob("*.shp"):
                file.unlink()
                for extension in (".cpg", ".dbf", ".prj", ".shx"):
                    (file.with_suffix(extension)).unlink()

            intersection_df: DataFrame
            with open(intersection_csv_file) as csv:
                intersection_df = pd.read_csv(csv, dtype=str)

            if intersection_df is None:
                print(f"Keine Daten für {year} vorhanden?!")
                continue

            # Add year
            intersection_df["JAHR"] = year

            # Run transformations
            for column, transformation in COLUMNS:
                intersection_df[column] = intersection_df[column].map(transformation, na_action="ignore")
            intersection_df.fillna(np.nan, inplace=True)
            intersection_df.replace({np.nan: None}, inplace=True)

            print(intersection_df.max().to_string())

            save_in_database(intersection_df, year)


def save_in_database(intersection_df, year):
    print(f"Speichere {year} in Datenbank...")
    now = datetime.datetime.now()
    with connect_to_database() as connection:
        cursor: pyodbc.Cursor
        with connection.cursor() as cursor:
            cursor.fast_executemany = True
            cursor.execute("USE [BigData]")
            # Delete old data
            cursor.execute("DELETE FROM [Daten] WHERE [JAHR] = ?", year)

            columns = str.join(",", (f"[{column.name}]" for column in COLUMNS))
            placeholders = str.join(",", ("?" for _ in COLUMNS))
            try:
                for row in intersection_df[[c.name for c in COLUMNS]].values.tolist():
                    cursor.execute(
                        f"INSERT INTO [Daten]({columns}) VALUES({placeholders})",
                        row)
                cursor.commit()
            except pyodbc.DatabaseError as e:
                cursor.rollback()
                raise e
    print(
        f"Speichern von {year} in Datenbank abgeschlossen in",
        (datetime.datetime.now() - now).seconds,
        "Sekunden")


def main():
    initialize_database()
    brw_datasets = retrieve_downloaded_datasets("BRW")
    irw_datasets = retrieve_downloaded_datasets("IRW")
    print("Vorhandene BRW Datensätze:", brw_datasets)
    print("Vorhandene IRW Datensätze:", irw_datasets)
    downloaded_brw = download_brw(brw_datasets)
    downloaded_irw = download_irw(irw_datasets)

    process_years(set(downloaded_brw).union(downloaded_irw))


if __name__ == "__main__":
    main()
