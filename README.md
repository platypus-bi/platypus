# Platypus

![Logo of Platypus](.assets/logo.jpg)

A BI application for comparing and correlating land prices and property prices.

## What is this application about?

## Deploy it on your own

### Requirements

To run Platypus, you need the following:

#### Server

- A recent version of Docker (with Docker Compose)
- At least 2 GB of RAM

#### Your PC

- An installation of [Microsoft Power BI Desktop](https://aka.ms/pbidesktopstore)

### Server installation instructions

Firstly make sure you have *Docker Compose* properly installed.  
To do so, run `docker compose version` and verify that it returns `Docker Compose version v2.12.2` or a later version.

Next obtain root/superuser privileges. Unless you are running the Docker daemon as a non-root user (<https://docs.docker.com/engine/security/rootless/>), you need root/superuser privileges to work with it.  
In order to deploy this application, just clone this repository and run the `deploy.sh` file as root.  
Read through the installer carefully and save the database password.
You need it to connect to the database later.
