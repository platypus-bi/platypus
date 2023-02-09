"""
This script downloads the latest data from the BORIS database and inserts it into the database.

The script first checks if a new dataset is available. If so, it downloads it and inserts it into the database.

The script is intended to be run as a scheduled task every day.
"""

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
from apscheduler.schedulers.blocking import BlockingScheduler

BORIS_BRW_BASE = "https://www.opengeodata.nrw.de/produkte/infrastruktur_bauen_wohnen/boris/BRW"
BORIS_BRW_JSON_INDEX = f"{BORIS_BRW_BASE}/index.json"

BORIS_IRW_BASE = "https://www.opengeodata.nrw.de/produkte/infrastruktur_bauen_wohnen/boris/IRW"
BORIS_IRW_JSON_INDEX = f"{BORIS_IRW_BASE}/index.json"

ZIP_NAME_PATTERN = re.compile(r"(?P<type>[A-Z]{3})_(?P<year>\d{4}).*?\.zip")

Year = int
DatasetType = str
Datasets = dict[DatasetType, dict[Year, datetime.datetime]]


class Column(NamedTuple):
    """
    Represents a column in the database.
    The name is the same as the column name in the database.
    The conversion is a function that converts the value from the CSV file to the correct type.
    The placeholder is the placeholder for the value in the SQL query. It defaults to "?". This is
    only necessary if the value needs to be converted to a different type before inserting it into
    the database.
    """
    name: str
    conversion: Callable[[str], Any]
    placeholder: str = "?"


def identity(value):
    """
    Returns the value without any conversion.
    :param value: The value to return.
    :return: The value.
    """
    return value


def parse_float(value) -> Optional[float]:
    """
    Parses a float from a string.
    Replace "," with "." to make sure that the float is parsed correctly.
    :param value: The string to parse.
    :return: The parsed float.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        return float(value.replace(",", ".").replace("*", ""))

    return float(value)


def parse_int(value) -> Optional[float]:
    """
    Parses an int from a string.
    Remove all "." from the string to make sure that the int is parsed correctly.
    :param value: The string to parse.
    :return: The parsed int.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value.replace(".", ""))

    return int(value)


# Columns to insert into the database
COLUMNS: list[Column] = [
    Column("JAHR", identity),
    Column("BEZUG", identity),
    Column("BRKE", int),
    Column("PLZ", identity),
    Column("GENA", identity),
    Column("ORTST", identity),
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
    Column("WKT", identity, "geography::STGeomFromText(?, 4326)"),
]


def print_flush(*args, **kwargs):
    """
    Prints the given arguments and flushes the output.
    :param args: Positional arguments to print.
    :param kwargs: Keyword arguments to print.
    :return: None
    """
    kwargs["flush"] = True

    # Print the current date and time for debugging purposes
    print('[', datetime.datetime.now(), ']', sep='', end=": ")
    print(*args, **kwargs)


def retrieve_downloaded_datasets(dataset_type: DatasetType) -> Datasets:
    """
    Retrieves the datasets that have already been downloaded from the database.
    :param dataset_type: The type of the dataset to retrieve. Either "IRW" or "BRW".
    :return: A dictionary mapping the year to the last updated date.
    """
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
    """
    Connects to the database.
    Uses the environment variable "MSSQL_SA_PASSWORD" as the password for the "sa" user.
    :param kwargs: Additional keyword arguments to pass to the pyodbc.connect function.
    :return: The connection to the database.
    """
    conn = pyodbc.connect("DSN=MSSQLServerDatabase",
                          user="sa",
                          password=os.environ["MSSQL_SA_PASSWORD"],
                          **kwargs)
    conn.setencoding(encoding='utf-8')
    return conn


def download_large_file(url: str, destination_path: Path):
    """
    Downloads a large file from the given URL.
    Uses the requests module to download the file in chunks of 10 MB.
    :param url: The URL to download the file from.
    :param destination_path: The path to save the file to.
    :return: None
    """
    # NOTE the stream=True parameter
    with requests.get(url, stream=True, timeout=120) as request:
        request.raise_for_status()
        with open(destination_path, "wb") as file:
            # 10 MB chunk size
            for chunk in request.iter_content(chunk_size=10 * 1024 * 1024):
                file.write(chunk)


def unpack_file(path: Path):
    """
    Unpacks the given zip file.
    PDF, TXT and XLS files are not unpacked.
    The output folder is the same as the name of zip file (without the .zip extension).
    The zip file is deleted after unpacking.
    :param path: The path to the zip file.
    :return: None
    """
    # Unpack the zip file using the zipfile module
    # Unpack into a folder with the same name as the zip file (without the .zip extension)
    # Only unpack files that are not PDF, TXT, XLSX and XLS files

    output_folder = path.parent / path.stem

    with zipfile.ZipFile(path, "r") as zip_file:
        for file in zip_file.namelist():
            extension = os.path.splitext(file)[1]
            if extension not in (".pdf", ".txt", ".xlsx", ".xls"):
                zip_file.extract(file, output_folder)
    # Delete the zip file
    path.unlink()


def determine_latest_year(dataset_type: DatasetType, datasets: dict) -> Year:
    """
    Determines the latest year for which a dataset is available.
    :param dataset_type: The type of the dataset. Either "IRW" or "BRW".
    :param datasets: The datasets to check.
    :return: The latest year for which a dataset is available.
    """
    latest_year = 0
    for dataset in datasets:
        if dataset["name"] == f"{dataset_type}_historisch":
            files = dataset["files"]
            for file in files:
                filename = file["name"]
                match = ZIP_NAME_PATTERN.fullmatch(filename)
                latest_year = max(latest_year, int(match["year"]))
    if latest_year == 0:
        latest_year = datetime.datetime.now().year
    else:
        latest_year += 1
    return latest_year


def determine_year(filename: str) -> Year:
    """
    Determines the year from the given filename.
    :param filename: The filename to determine the year from.
    :return: The year from the given filename.
    """
    match = ZIP_NAME_PATTERN.fullmatch(filename)
    return int(match["year"])


def download_datasets(url: str,
                      dataset_type: DatasetType,
                      current_datasets: Datasets,
                      current_callback: Callable[[Year, Datasets, dict, list[Year]], None],
                      history_callback: Callable[[Datasets, dict, list[Year]], None]) -> list[Year]:
    """
    Downloads the datasets from the given base URL.
    Only downloads datasets that have not been downloaded yet.

    :param url: The url containing the index of the datasets.
    :param dataset_type: The type of the dataset to download. Either "IRW" or "BRW".
    :param current_datasets: The current datasets already downloaded.
    :param current_callback: A callback function to call for the current dataset.
    :param history_callback: A callback function to call for the historical datasets.
    :return: A list of years for which a dataset was newly downloaded.
    """
    request = requests.get(url, timeout=120)
    index = json.loads(request.text)

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
                print_flush("Historische Datensätze werden nicht heruntergeladen")
        else:
            print_flush("Unbekannter Datensatz", dataset["name"])

    return downloaded_years


def download_brw(current_brw_datasets: Datasets) -> list[Year]:
    """
    Downloads the BRW datasets.
    Only downloads datasets that have not been downloaded yet.
    :param current_brw_datasets: The current BRW datasets already downloaded.
    :return: A list of years for which a dataset was newly downloaded.
    """
    return download_datasets(BORIS_BRW_JSON_INDEX,
                             "BRW",
                             current_brw_datasets,
                             download_brw_aktuell,
                             download_brw_historisch)


def download_irw(current_irw_datasets: Datasets) -> list[Year]:
    """
    Downloads the IRW datasets.
    Only downloads datasets that have not been downloaded yet.
    :param current_irw_datasets: The current IRW datasets already downloaded.
    :return: A list of years for which a dataset was newly downloaded.
    """
    return download_datasets(BORIS_IRW_JSON_INDEX,
                             "IRW",
                             current_irw_datasets,
                             download_irw_aktuell,
                             download_irw_historisch)


def save_dataset_download(dataset_type: DatasetType, year: Year, timestamp: datetime.datetime):
    """
    Save the information about the downloaded dataset to the database.
    :param dataset_type: The type of the dataset. Either "IRW" or "BRW".
    :param year: The year of the dataset.
    :param timestamp: The timestamp when the dataset was last updated.
    :return: None
    """
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
    """
    Downloads a historical dataset.
    Only downloads datasets that have not been downloaded yet.

    :param base_url: The base URL of the dataset.
    :param dataset_type: The type of the dataset to download. Either "IRW" or "BRW".
    :param current_datasets: The current datasets already downloaded.
    :param dataset: The dataset to download.
    :param downloaded_years: A list of years for which a dataset was newly downloaded.
    :return: None
    """
    print_flush("Datensatz herunterladen:", dataset["name"])
    for file in dataset["files"]:
        year = determine_year(file["name"])
        print_flush("Datei von:", file["timestamp"])
        timestamp = datetime.datetime.fromisoformat(file["timestamp"])
        timestamp = timestamp.replace(microsecond=0)
        print_flush("Datei für das Jahr:", year)
        print_flush("Datei prüfen:", file["name"])
        print_flush("Datei aktualisiert:", timestamp.isoformat())

        datasets_by_year = current_datasets.get(dataset_type)
        if datasets_by_year is not None:
            timestamp_for_year = datasets_by_year.get(year)
            if timestamp_for_year is not None:
                print_flush("Datei bereits vorhanden:", timestamp_for_year.isoformat())
                if timestamp_for_year >= timestamp:
                    print_flush("Datei noch aktuell, überspringe...")
                    continue

        print_flush("Datei herunterladen:", file["name"])

        download_dir = Path(".") / str(year) / dataset_type
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / file["name"]
        download_large_file(f"{base_url}/{file['name']}", download_path)
        print_flush("Datei heruntergeladen:", file["name"])
        print_flush("Entpacke:", file["name"])
        unpack_file(download_path)
        print_flush("Datei entpackt und bereinigt:", file["name"])
        save_dataset_download(dataset_type, year, timestamp)
        downloaded_years.append(year)


def download_aktuell(*,
                     base_url: str,
                     dataset_type: DatasetType,
                     latest_year: Year,
                     current_datasets: Datasets,
                     dataset: dict,
                     downloaded_years: list[Year]):
    """
    Downloads the current dataset.
    Only downloads datasets that have not been downloaded yet.

    :param base_url: The base URL of the dataset.
    :param dataset_type: The type of the dataset to download. Either "IRW" or "BRW".
    :param latest_year: The latest year for which a dataset is available.
    :param current_datasets: The current datasets already downloaded.
    :param dataset: The dataset to download.
    :param downloaded_years: A list of years for which a dataset was newly downloaded.
    :return: None
    """
    print_flush("Datensatz herunterladen:", dataset["name"])
    for file in dataset["files"]:
        print_flush("Datei von:", file["timestamp"])
        timestamp = datetime.datetime.fromisoformat(file["timestamp"])
        timestamp = timestamp.replace(microsecond=0)
        print_flush("Datei für das Jahr:", latest_year)
        print_flush("Datei prüfen:", file["name"])
        print_flush("Datei aktualisiert:", timestamp.isoformat())

        datasets_by_year = current_datasets.get(dataset_type)
        if datasets_by_year is not None:
            timestamp_for_year = datasets_by_year.get(latest_year)
            if timestamp_for_year is not None:
                print_flush("Datei bereits vorhanden:", timestamp_for_year.isoformat())
                if timestamp_for_year >= timestamp:
                    print_flush("Datei noch aktuell, überspringe...")
                    continue

        print_flush("Datei herunterladen:", file["name"])

        download_dir = Path(".") / str(latest_year) / dataset_type
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / file["name"]
        download_large_file(f"{base_url}/{file['name']}", download_path)
        print_flush("Datei herunterladen:", file["name"], "fertig")
        print_flush("Entpacke:", file["name"])

        # Old name: BRW_EPSG25832_Shape.zip
        # New name: BRW_2022_EPSG25832_Shape.zip
        download_path = download_path.rename(download_dir / f"{dataset_type}_{latest_year}_EPSG25832_Shape.zip")
        unpack_file(download_path)
        print_flush("Datei entpackt und bereinigt:", file["name"])
        save_dataset_download(dataset_type, latest_year, timestamp)
        downloaded_years.append(latest_year)


def download_brw_aktuell(latest_year: Year, current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    """
    Downloads the current BRW dataset.
    :param latest_year: The latest year for which a dataset is available.
    :param current_datasets: The current datasets already downloaded.
    :param dataset: The dataset to download.
    :param downloaded_years: A list of years for which a dataset was newly downloaded.
    :return: None
    """
    download_aktuell(base_url=BORIS_BRW_BASE,
                     dataset_type="BRW",
                     latest_year=latest_year,
                     current_datasets=current_datasets,
                     dataset=dataset,
                     downloaded_years=downloaded_years)


def download_brw_historisch(current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    """
    Downloads the historical BRW dataset.
    :param current_datasets: The current datasets already downloaded.
    :param dataset: The dataset to download.
    :param downloaded_years: A list of years for which a dataset was newly downloaded.
    :return: None
    """
    download_historisch(BORIS_BRW_BASE, "BRW", current_datasets, dataset, downloaded_years)


def download_irw_aktuell(latest_year: Year, current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    """
    Downloads the current IRW dataset.
    :param latest_year: The latest year for which a dataset is available.
    :param current_datasets: The current datasets already downloaded.
    :param dataset: The dataset to download.
    :param downloaded_years: A list of years for which a dataset was newly downloaded.
    :return: None
    """
    download_aktuell(base_url=BORIS_IRW_BASE,
                     dataset_type="IRW",
                     latest_year=latest_year,
                     current_datasets=current_datasets,
                     dataset=dataset,
                     downloaded_years=downloaded_years)


def download_irw_historisch(current_datasets: Datasets, dataset: dict, downloaded_years: list[Year]):
    """
    Downloads the historical IRW dataset.
    :param current_datasets: The current datasets already downloaded.
    :param dataset: The dataset to download.
    :param downloaded_years: A list of years for which a dataset was newly downloaded.
    :return: None
    """
    download_historisch(BORIS_IRW_BASE, "IRW", current_datasets, dataset, downloaded_years)


def run_sql_script(cursor: pyodbc.Cursor, script: str):
    """
    Runs a SQL script stored in the sql directory shipped with the application.
    :param cursor: The cursor to use for executing the script.
    :param script: The name of the script to run.
    :return: None
    """
    with open((Path("/app/sql") / script).with_suffix(".sql"), encoding="utf-8") as sql_file:
        try:
            for statement in sql_file.read().split("---"):
                cursor.execute(statement)
            cursor.commit()
        except pyodbc.ProgrammingError as exception:
            cursor.rollback()
            raise exception


def initialize_database():
    """
    Initializes the database.
    Makes use of the SQL scripts in the sql directory shipped with the application.
    :return: None
    """
    with connect_to_database(autocommit=True) as connection:
        with connection.cursor() as cursor:
            run_sql_script(cursor, "01-create-database")
            run_sql_script(cursor, "02-create-table-types")
            run_sql_script(cursor, "03-create-table-stockpile")
            run_sql_script(cursor, "04-create-table-data")


def process_years(years: set[Year]):
    """
    Processes the given years.
    For each year, the BRW and IRW datasets are intersected and the resulting shapefiles are imported into the database.
    The following steps are performed:
    1. The BRW and IRW datasets are intersected.
    2. The resulting geopackage is reprojected to EPSG:4326.
    3. A column is added to the geopackage containing the WKT representation of the geometry.
    4. The centroid of the geometry is calculated and replaces the geometry.
    5. The location of the centroid as latitude and longitude is added as separate columns.
    6. The final geopackage is exported to a CSV file.
    7. The CSV file is imported into the database.

    :param years: The years to process.
    :return: None
    """
    brw_datasets = retrieve_downloaded_datasets("BRW")
    irw_datasets = retrieve_downloaded_datasets("IRW")

    brw_datasets = brw_datasets.get("BRW")
    if brw_datasets is None:
        return

    irw_datasets = irw_datasets.get("IRW")
    if irw_datasets is None:
        return

    # Create the intersection directory if it does not exist.
    Path("intersection").mkdir(parents=True, exist_ok=True)

    for year in years:
        if year in brw_datasets and year in irw_datasets:
            output_dir = Path("intersection") / str(year)
            output_dir = output_dir.absolute()
            output_dir.mkdir(parents=True, exist_ok=True)

            brw_path = Path(".") / str(year) / "BRW" / f"BRW_{year}_EPSG25832_Shape" / f"BRW_{year}_Polygon.shp"
            brw_path = brw_path.absolute()
            irw_path = Path(".") / str(year) / "IRW" / f"IRW_{year}_EPSG25832_Shape" / f"IRW_{year}_Polygon.shp"
            irw_path = irw_path.absolute()

            print_flush(f"Verarbeite {year}...")
            intersection_file = create_intersection(brw_path, irw_path, output_dir, year)

            reprojection_file = create_reprojection(intersection_file, output_dir, year)

            wkt_file = create_wkt_field(reprojection_file, output_dir, year)

            centroid_file = create_centroids(wkt_file, output_dir, year)

            lat_long_file = create_lat_long(centroid_file, output_dir, year)

            intersection_csv_file = create_csv_file(lat_long_file, output_dir, year)

            final_process_csv_file(intersection_csv_file, output_dir, year)


def final_process_csv_file(csv_file: Path, output_dir: Path, year: int):
    """
    Processes the final CSV file and imports it into the database.
    :param csv_file: The CSV file to process.
    :param output_dir: The output directory.
    :param year: The year of the dataset.
    :return:
    """
    intersection_df: pd.DataFrame
    with open(csv_file, encoding="utf-8") as csv:
        intersection_df = pd.read_csv(csv, dtype=str)
    if intersection_df is None:
        print_flush(f"Keine Daten für {year} vorhanden?!")
        return

    # Add year
    intersection_df["JAHR"] = year

    # Run transformations
    column: str
    transformation: Callable[[str], Any]
    for column, transformation, _ in COLUMNS:
        intersection_df[column] = intersection_df[column].map(transformation, na_action="ignore")
        if column == "HBRW":
            # If the value in column STAG is less than 2022-01-01, then parse the column BRW as HBRW, else keep the
            # column HBRW as is.
            intersection_df["BRW"] = intersection_df["BRW"].map(transformation, na_action="ignore")
            intersection_df["HBRW"] = intersection_df.apply(
                lambda row: row["BRW"] if datetime.datetime.fromisoformat(row["STAG"]) < datetime.datetime(2022, 1, 1)
                else row["HBRW"], axis=1
            )

    intersection_df.fillna(np.nan, inplace=True)
    intersection_df.replace({np.nan: None}, inplace=True)

    save_in_database(intersection_df, year)
    clean_up_geo_files(output_dir)


def clean_up_geo_files(output_dir):
    """
    Cleans up the files created during the processing of the data.
    This function deletes all Shapefiles and GeoPackage files, as well as their auxiliary files.
    :param output_dir: The directory containing the files to clean up.
    :return: None
    """
    # This requires deleting all .shp files and their .cpg, .dbf, .prj, .shx files as well
    for file in output_dir.glob("*.shp"):
        file.unlink()
        for extension in (".cpg", ".dbf", ".prj", ".shx"):
            (file.with_suffix(extension)).unlink()

    # Delete the .gpkg files
    for file in output_dir.glob("*.gpkg"):
        file.unlink()


def create_intersection(brw_path: Path, irw_path: Path, output_dir: Path, year) -> Path:
    """
    Creates the intersection of the BRW and IRW files.
    This is done using the QGIS native:intersection algorithm.
    The result is a dataset containing the intersection of the two input files, i.e. the areas where both BRW and IRW
    are present.
    :param brw_path: The path to the BRW file
    :param irw_path: The path to the IRW file
    :param output_dir: The output directory
    :param year: The year of the input file
    :return: The path to the intersection file
    """
    print_flush("Erzeuge Schnittmenge...")
    intersection_file = output_dir / f"intersection_{year}.gpkg"
    now = datetime.datetime.now()
    subprocess.run([
        "qgis_process.bin",
        "run",
        "native:intersection",
        f"INPUT={brw_path}",
        f"OVERLAY={irw_path}",
        f"OUTPUT={intersection_file}"
    ],
        check=True)
    print_flush(f"Erzeugung der Schnittmenge von {year} abgeschlossen in",
                (datetime.datetime.now() - now).seconds,
                "Sekunden")
    return intersection_file


def create_reprojection(input_file: Path, output_dir: Path, year) -> Path:
    """
    Reprojects the input file to WGS84/EPSG:4326 (GPS coordinates/lat/long).
    :param input_file: The input file (should be a GPKG or SHP file)
    :param output_dir: The output directory
    :param year: The year of the input file
    :return: The path to the reprojected file
    """
    # Reproject to WGS84
    print_flush(f"Reprojiziere {year}...")
    reprojection_file = output_dir / f"reprojection_{year}.gpkg"
    now = datetime.datetime.now()
    subprocess.run([
        "qgis_process.bin",
        "run",
        "native:reprojectlayer",
        f"INPUT={input_file}",
        "TARGET_CRS=EPSG:4326",
        f"OUTPUT={reprojection_file}"
    ],
        check=True)
    print_flush(f"Reprojizierung von {year} abgeschlossen in",
                (datetime.datetime.now() - now).seconds,
                "Sekunden")
    return reprojection_file


def create_centroids(input_file: Path, output_dir: Path, year) -> Path:
    """
    Create centroids for the geometry of the given input file and save them to the given output directory.
    :param input_file: The input file (should be a GPKG or SHP file)
    :param output_dir: The output directory
    :param year: The year of the input file
    :return: The path to the created file
    """
    # Convert polygons to centroids
    print_flush(f"Erstelle Zentroide für {year}...")
    centroid_file = output_dir / f"centroids_{year}.gpkg"
    now = datetime.datetime.now()
    subprocess.run([
        "qgis_process.bin",
        "run",
        "native:pointonsurface",
        f"INPUT={input_file}",
        f"OUTPUT={centroid_file}",
    ],
        check=True)
    print_flush(f"Zentroide für {year} erstellt in",
                (datetime.datetime.now() - now).seconds,
                "Sekunden")
    return centroid_file


def create_lat_long(input_file: Path, output_dir: Path, year) -> Path:
    """
    Add latitude and longitude to the given file for points of the geometry
    :param input_file: The input file (should be a GPKG or SHP file)
    :param output_dir: The output directory
    :param year: The year of the input file
    :return: The path to the output file
    """
    print_flush(f"Füge Breitengrad und Längengrad für {year} hinzu...")
    now = datetime.datetime.now()
    lat_long_file = output_dir / f"lat_long_{year}.gpkg"
    subprocess.run([
        "qgis_process.bin",
        "run",
        "native:addxyfields",
        f"INPUT={input_file}",
        f"OUTPUT={lat_long_file}",
        "CRS=EPSG:4326"
    ],
        check=True)
    print_flush(
        f"Breitengrad und Längengrad für {year} hinzugefügt in",
        (datetime.datetime.now() - now).seconds,
        "Sekunden")
    return lat_long_file


def create_wkt_field(input_file: Path, output_dir: Path, year) -> Path:
    """
    Create a new GeoPackage file with a WKT field of the geometry
    added to the input file.
    :param input_file: The input file (should be a GPKG or SHP file)
    :param output_dir: The output directory
    :param year: The year of the input file
    :return: The path to the new file
    """
    print_flush(f"Füge WKT für {year} hinzu...")
    now = datetime.datetime.now()
    wkt_file = output_dir / f"wkt_{year}.gpkg"
    subprocess.run([
        "qgis_process.bin",
        "run",
        "qgis:fieldcalculator",
        f"INPUT={input_file}",
        f"OUTPUT={wkt_file}",
        "FIELD_NAME=WKT",
        "FIELD_TYPE=2",
        "FIELD_LENGTH=0",
        "NEW_FIELD=1",
        "FORMULA=geom_to_wkt($geometry)",
        "CRS=EPSG:4326"
    ],
        check=True)
    print_flush(
        f"WKT für {year} hinzugefügt in",
        (datetime.datetime.now() - now).seconds,
        "Sekunden")
    return wkt_file


def create_csv_file(input_file: Path, output_dir: Path, year) -> Path:
    """
    Create a CSV file from the input file using QGIS
    :param input_file: The input file (should be a GPKG or SHP file)
    :param output_dir: The output directory
    :param year: The year of the input file
    :return: The path to the CSV file
    """
    print_flush(f"Erstelle CSV für {year}...")
    intersection_csv_file = output_dir / f'intersection_{year}.csv'
    now = datetime.datetime.now()
    subprocess.run([
        "qgis_process.bin",
        "run",
        "native:savefeatures",
        f"INPUT={input_file}",
        f"OUTPUT={intersection_csv_file}"
    ],
        check=True)
    print_flush(f"CSV für {year} erstellt in",
                (datetime.datetime.now() - now).seconds,
                "Sekunden")
    return intersection_csv_file


def save_in_database(intersection_df: pd.DataFrame, year: int):
    """
    Save the intersection dataframe in the database.
    :param intersection_df: The dataframe to save.
    :param year: The year of the data.
    :return: None
    """
    print_flush(f"Speichere {year} in Datenbank...")
    now = datetime.datetime.now()
    with connect_to_database() as connection:
        cursor: pyodbc.Cursor
        with connection.cursor() as cursor:
            cursor.fast_executemany = True
            cursor.execute("USE [BigData]")
            # Delete old data
            cursor.execute("DELETE FROM [Daten] WHERE [JAHR] = ?", year)

            columns = str.join(",", (f"[{column.name}]" for column in COLUMNS))
            placeholders = str.join(",", (c.placeholder for c in COLUMNS))
            try:
                for index, row in enumerate(intersection_df[[c.name for c in COLUMNS]].values.tolist()):
                    cursor.execute(
                        f"INSERT INTO [Daten]({columns}) VALUES({placeholders})",
                        row)
                    if index % 1000 == 0:
                        print_flush(f"{index} Datensätze gespeichert")
                        cursor.commit()
                cursor.commit()
            except pyodbc.DatabaseError as exception:
                cursor.rollback()
                raise exception
    print_flush(
        f"Speichern von {year} in Datenbank abgeschlossen in",
        (datetime.datetime.now() - now).seconds,
        "Sekunden")


def main():
    """
    Main function

    :return: None
    """
    initialize_database()
    brw_datasets = retrieve_downloaded_datasets("BRW")
    irw_datasets = retrieve_downloaded_datasets("IRW")
    print_flush("Vorhandene BRW Datensätze:", brw_datasets)
    print_flush("Vorhandene IRW Datensätze:", irw_datasets)
    downloaded_brw = download_brw(brw_datasets)
    downloaded_irw = download_irw(irw_datasets)

    process_years(set(downloaded_brw).union(downloaded_irw))


if __name__ == "__main__":
    main()
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'cron', day_of_week='mon-sun', hour=0)
    scheduler.start()
