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

function uninstall {
    check_installed "docker"

    SCRIPTDIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")

    docker compose --file "$SCRIPTDIR/basement/docker-compose.yml" down
    docker compose --file "$SCRIPTDIR/basement/docker-compose.yml" rm --force --volumes
}

root_check

read -p "Do you really want to stop and uninstall platypus? (y/N)" -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    uninstall
else
    exit 0
fi
