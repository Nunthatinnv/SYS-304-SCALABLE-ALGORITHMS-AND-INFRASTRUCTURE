#!/bin/sh
# Rewrite config.js at container start so the API base URL is configurable
# without rebuilding the image.
set -eu

: "${API_URL:=http://localhost:8000}"

cat > /usr/share/nginx/html/config.js <<CONFIG
window.API_URL = "${API_URL}";
CONFIG

exec nginx -g "daemon off;"
