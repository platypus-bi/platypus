#!/bin/sh
exec /usr/bin/xvfb-run --auto-servernum --server-num=1 python3 /app/app.py
cron -f
