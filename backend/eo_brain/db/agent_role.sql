-- Least-privilege login role for the data agent. Run AFTER agent_views.sql,
-- as a superuser, in the application database. Idempotent.
--
-- The password is NOT set here; set it separately so it never lives in a
-- versioned file (see agent_role_password.sh for docker compose).
--
-- What agent_reader can do: connect, SELECT from views in schema "agent",
-- call functions (PostGIS lives in schema "public", so it keeps USAGE there).
-- What it cannot do: read any table in "public", create objects anywhere,
-- create temporary tables, write, or run longer than statement_timeout.
--
-- Residual: like every role, it can read the system catalogs (pg_catalog,
-- information_schema), which PostgreSQL grants to PUBLIC. Only the SQL gate
-- (eo_brain/agent/sql_gate.py) stops the model from querying them.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_reader') THEN
        CREATE ROLE agent_reader LOGIN
            NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS
            CONNECTION LIMIT 10;
    END IF;
END
$$;

-- Session defaults, applied at login.
ALTER ROLE agent_reader SET default_transaction_read_only = on;
ALTER ROLE agent_reader SET statement_timeout = '3s';
ALTER ROLE agent_reader SET lock_timeout = '1s';
ALTER ROLE agent_reader SET idle_in_transaction_session_timeout = '10s';
ALTER ROLE agent_reader SET search_path = agent, public;

-- Database: connect, but no temporary tables for anyone except superusers.
DO $$
BEGIN
    EXECUTE format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO agent_reader', current_database());
END
$$;

-- Schemas: no CREATE anywhere; USAGE on "agent" only for agent_reader.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA agent FROM PUBLIC;
GRANT USAGE ON SCHEMA agent TO agent_reader;

-- Relations: nothing in "public" (including PostGIS metadata readable by
-- PUBLIC by default), SELECT on the agent views only: analyses, alert_areas,
-- analysis_severity_summary and index_value_distribution. The tables behind
-- them (jobs, alert_features, index_value_bins) stay unreadable. The grant
-- covers the views that exist when this file runs: run it after agent_views.sql.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM agent_reader;
REVOKE SELECT ON public.spatial_ref_sys, public.geometry_columns, public.geography_columns FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA agent FROM PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA agent TO agent_reader;
-- One exception: PostGIS reads the EPSG reference table spatial_ref_sys with
-- the caller's privileges for geography casts and ST_Transform, so ST_Area
-- on the views fails without it. It holds only public CRS definitions.
GRANT SELECT ON public.spatial_ref_sys TO agent_reader;

-- Extension schemas created by the postgis image (tiger geocoder, topology):
-- remove PUBLIC access so agent_reader cannot read them either.
DO $$
DECLARE s text;
BEGIN
    FOREACH s IN ARRAY ARRAY['tiger', 'tiger_data', 'topology'] LOOP
        IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = s) THEN
            EXECUTE format('REVOKE ALL ON SCHEMA %I FROM PUBLIC', s);
            EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA %I FROM PUBLIC', s);
        END IF;
    END LOOP;
END
$$;
