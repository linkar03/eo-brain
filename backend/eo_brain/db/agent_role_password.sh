#!/usr/bin/env bash
# docker-entrypoint-initdb.d hook: set the data agent role's password from
# the environment, so it never lives in a versioned SQL file.
set -euo pipefail
: "${AGENT_DB_PASSWORD:?AGENT_DB_PASSWORD must be set for the data agent role}"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v agent_password="$AGENT_DB_PASSWORD" <<'EOSQL'
ALTER ROLE agent_reader PASSWORD :'agent_password';
EOSQL
