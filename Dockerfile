# Use an official Node.js runtime as the base image
FROM node:14

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the app's code
COPY . .

# Expose the port the app will run on
EXPOSE 3000

# Run the app
CMD ["npm", "start"]
