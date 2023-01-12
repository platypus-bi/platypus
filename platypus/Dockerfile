FROM debian:bullseye-slim

ENV LANG=en_EN.UTF-8


RUN apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
        gnupg \
        ca-certificates \
        wget \
        locales \
        cron \
    && localedef -i en_US -f UTF-8 en_US.UTF-8 \
    # Add the current key for package downloading
    # Please refer to QGIS install documentation (https://www.qgis.org/fr/site/forusers/alldownloads.html#debian-ubuntu)
    && mkdir -m755 -p /etc/apt/keyrings \
    && wget -O /etc/apt/keyrings/qgis-archive-keyring.gpg https://download.qgis.org/downloads/qgis-archive-keyring.gpg \
    # Add repository for latest version of qgis-server
    # Please refer to QGIS repositories documentation if you want other version (https://qgis.org/en/site/forusers/alldownloads.html#repositories)
    && echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/debian bullseye main" | tee /etc/apt/sources.list.d/qgis.list \
    && apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
        qgis python3-pip \
    && wget -O - https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/microsoft.gpg arch=amd64] https://packages.microsoft.com/debian/11/prod bullseye main" | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install --no-install-recommends --no-install-suggests -y msodbcsql18 unixodbc \
    && apt-get remove --purge -y \
        gnupg \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY run.sh /app/
COPY app.py /app/
COPY requirements.txt /app/
COPY crontab /app/

RUN crontab /app/crontab
RUN pip install -r /app/requirements.txt

# Setup ODBC DSN
COPY mssql-dsn /app/
RUN odbcinst -i -s -f /app/mssql-dsn -l

USER root
VOLUME [ "/work" ]
WORKDIR /work
ARG MSSQL_SA_PASSWORD
ENV MSSQL_SA_PASSWORD="${MSSQL_SA_PASSWORD}"
CMD ["/bin/bash", "-c", "/app/run.sh"]