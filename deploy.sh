#!/bin/bash

function echo_error {
    echo "$@" >&2 # Write to stderr
}

function check_installed {
    if [ $# -ne 1 ]; then
        echo_error "Pass exactly one executable to check!"
        exit 1
    fi

    if ! command -v "$1" &>/dev/null; then
        echo_error "Required executable \"$1\" is not installed."
        echo_error "Please install and run the deploy script again."
        exit 1
    fi
}

function license_prompt_mssql {
    echo "Please make sure to read the Microsoft SQL Server End-User License Agreement"
    echo "https://go.microsoft.com/fwlink/?linkid=857698"
    read -p "Do you agree to the license agreement? (y/N)" -n 1 -r

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

check_installed "dirname"
check_installed "readlink"
check_installed "docker"

SCRIPTDIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")

license_prompt_mssql
LICENSE_PROMPT=$?

# TODO: Check .env file for configuration (e.g. sa password)

if [ "$LICENSE_PROMPT" -ne 0 ]; then
    exit 1
fi

docker compose pull --file "$SCRIPTDIR/basement/docker-compose.yml"
docker compose up -d --file "$SCRIPTDIR/basement/docker-compose.yml"
