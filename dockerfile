# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /pocrag

# Copy the current directory contents into the container at /app
COPY ./src ./src
COPY ./requirements.txt ./requirements.txt
COPY ./app.py ./app.py
COPY ./poc.ipynb ./poc.ipynb
COPY ./db ./db
COPY ./docs ./docs


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]