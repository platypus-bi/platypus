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

function generate_password {
    if command -v "apg" &>/dev/null; then
        echo "Using apg to generate the password"
        PASSWORD="$(apg -a 1 -n 1 -m 20)"
    else
        echo "Falling back to /dev/urandom for password generation"
        PASSWORD="$(< /dev/urandom tr -dc \[:graph:\] | head -c 16)"
    fi
    echo "Generated the random password $PASSWORD"
    echo "Please save it for later use"
}

check_installed "sed"
check_installed "dirname"
check_installed "readlink"
check_installed "docker"

SCRIPTDIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")

license_prompt_mssql
LICENSE_PROMPT=$?

if [ "$LICENSE_PROMPT" -ne 0 ]; then
    exit 1
fi

# Check env file
BASEMENT_ENV="$SCRIPTDIR/basement/.env"
# shellcheck source=./basement/.env
source "$BASEMENT_ENV"

# Check sa password
if [ -z "$MSSQL_SA_PASSWORD" ]; then
    generate_password
    sed -n -i "s/MSSQL_SA_PASSWORD=/MSSQL_SA_PASSWORD=$PASSWORD/" "$BASEMENT_ENV"
fi

# TODO: Check .env file for configuration (e.g. sa password)

docker compose pull --file "$SCRIPTDIR/basement/docker-compose.yml"
docker compose up -d --file "$SCRIPTDIR/basement/docker-compose.yml"
