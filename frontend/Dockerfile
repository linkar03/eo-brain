FROM node:24-alpine

WORKDIR /app

# There is no lockfile in the repository (package-lock.json is git-ignored), so
# npm resolves the version ranges in package.json at build time.
COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

# Vite dev server. VITE_API_URL (set by docker compose) is read when the server
# starts; the browser on the host then calls the API at that URL directly.
CMD ["npm", "run", "dev", "--", "--host"]
