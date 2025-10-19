-- CRONOS AI Database Initialization Script
-- This script initializes the TimescaleDB extension and creates the database schema

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID extension for generating UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for encryption functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create schema for CRONOS AI
CREATE SCHEMA IF NOT EXISTS cronos_ai;

-- Set default schema
SET search_path TO cronos_ai, public;

-- Grant privileges to the application user
GRANT ALL PRIVILEGES ON SCHEMA cronos_ai TO cronos_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cronos_ai TO cronos_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA cronos_ai TO cronos_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA cronos_ai GRANT ALL ON TABLES TO cronos_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA cronos_ai GRANT ALL ON SEQUENCES TO cronos_user;

-- Log initialization completion
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB and required extensions initialized successfully';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
    RAISE NOTICE 'TimescaleDB version: %', (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb');
END
$$;
