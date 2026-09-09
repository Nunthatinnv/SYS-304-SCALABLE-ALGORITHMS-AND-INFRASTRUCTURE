# Static frontend. No build step: plain HTML, CSS and JavaScript served by nginx.
FROM nginx:stable-alpine

RUN apk add --no-cache curl

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/index.html frontend/app.js frontend/style.css frontend/config.js \
     /usr/share/nginx/html/
COPY frontend/docker-entrypoint.sh /docker-entrypoint-custom.sh

RUN chmod +x /docker-entrypoint-custom.sh

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost/healthz || exit 1

ENTRYPOINT ["/docker-entrypoint-custom.sh"]
