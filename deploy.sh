#!/usr/bin/env bash

function echo_error {
    echo "$@" >&2 # Write to stderr
}

function root_check {
    if [ "$(id -u)" -ne 0 ]; then
        echo_error "Please run this script as root!"
        exit 2
    fi
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
    read -p "Do you agree to the license agreement? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

function generate_password {
    echo "Using apg to generate the password"
    PASSWORD="$(apg -a 0 -n 1 -m 20 -M SNCL -E \")"
    echo "Generated the random password \"$PASSWORD\""
    echo "Please save it for later use"
}

check_installed "sed"
check_installed "dirname"
check_installed "readlink"
check_installed "docker"
check_installed "apg"

SCRIPTDIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")

echo "Welcome to Platypus!"
echo ""

root_check

license_prompt_mssql
LICENSE_PROMPT=$?

if [ "$LICENSE_PROMPT" -ne 0 ]; then
    exit 1
fi

# Check env file
ENV_FILE="$SCRIPTDIR/.env"
# shellcheck source=./.env
source "$ENV_FILE"

# Check sa password
if [ -z "$MSSQL_SA_PASSWORD" ]; then
    generate_password
    
    # Substitute the variable in the .env file, making sure to use # as a
    # delimiter for sed and escaping it in the password.
    sed -i -r -n "s#^(MSSQL_SA_PASSWORD=).*#\1\"${PASSWORD//#/\\#}\"#p" "$ENV_FILE"
fi

docker compose --file "$SCRIPTDIR/docker-compose.yml" pull
docker compose --file "$SCRIPTDIR/docker-compose.yml" up -d --build

echo "Platypus installed successfully!"