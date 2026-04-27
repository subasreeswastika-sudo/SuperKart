# 1. Base Image: Use the official lightweight Python image
FROM python:3.10

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# We do this before copying the rest of the code to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code 
# (This includes app.py and any local assets)
COPY . .

# 5. Environment Variables
# Ensures Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# 6. Expose the default Streamlit port (7860 is the default for HF Spaces)
EXPOSE 7860

# 7. Command to run the application
# We bind to 0.0.0.0 and port 7860 to satisfy HF Space requirements
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
