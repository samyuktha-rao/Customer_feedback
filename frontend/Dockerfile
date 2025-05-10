# First stage to install dependencies
FROM node:18-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci

# Copy all other files and build
COPY . . 
RUN npm run build

# Final stage for serving the app
FROM nginx:alpine

# Copy the built app to Nginx's public directory
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
