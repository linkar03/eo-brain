import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
        },
        severity: {
          alert: '#dc2626',
          suspect: '#f97316',
          watch: '#eab308',
          normal: '#22c55e',
        }
      }
    },
  },
  plugins: [],
} satisfies Config