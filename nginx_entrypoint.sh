#!/bin/sh
set -eu

envsubst '$AUTH_TOKEN' < /etc/nginx/conf.d/default.tmpl.conf > /etc/nginx/conf.d/default.conf

rm -f /etc/nginx/conf.d/default.tmpl.conf

exec nginx -g 'daemon off;'