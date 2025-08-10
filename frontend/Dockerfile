# Stage 1: The "builder" stage, where we build the app
FROM node:20-slim AS builder
WORKDIR /app

# Copy package files first
COPY package.json package-lock.json* ./

# Install ALL dependencies, including devDependencies like typescript
RUN npm install --include=dev

# Copy the rest of your application code
COPY . .

# Run the build command. 'tsc' is available here.
RUN npm run build


# Stage 2: The "production" stage, which serves the built app
FROM node:20-slim
WORKDIR /app

# Only install the 'serve' package for a lean final image
RUN npm install -g serve

# Copy only the built assets from the "builder" stage
COPY --from=builder /app/dist .

# Expose the port
EXPOSE 8080

# Run the server
CMD ["sh", "-c", "serve -s . -n -l tcp://0.0.0.0:${PORT:-8080}"]