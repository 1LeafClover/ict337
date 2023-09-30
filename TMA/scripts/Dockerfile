# ! JAVA_HOME is not set
# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
ENV NAME "PySpark_program"

# Run app.py when the container launches
CMD ["python", "PySpark_program.py"]