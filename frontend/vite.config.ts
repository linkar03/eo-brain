import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The app calls the API directly at VITE_API_URL (see src/api/config.ts),
// so the dev server needs no proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  }
})
