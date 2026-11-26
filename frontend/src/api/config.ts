/**
 * Base URL of the EO Brain API. Every request the frontend makes (the axios
 * client and the agent chat) uses this one definition.
 *
 * docker compose publishes the API on host port 8001 and sets VITE_API_URL to
 * http://localhost:8001. When the API runs elsewhere, for example
 * `python -m eo_brain.api.main` on port 8000, set VITE_API_URL before starting
 * the dev server or building.
 */
export const API_BASE: string = import.meta.env.VITE_API_URL || 'http://localhost:8001';
