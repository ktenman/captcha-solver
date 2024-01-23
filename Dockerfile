# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Copy the saved_model999.zip into the container
COPY saved_model999.zip .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 52525 available to the world outside this container
EXPOSE 52525

# Run app.py when the container launches
CMD ["python", "./app.py"]
