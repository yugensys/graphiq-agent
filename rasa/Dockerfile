FROM rasa/rasa:3.6.20-full

# Create app directory and set permissions
RUN mkdir -p /app \
    && chown -R 1001:0 /app \
    && chmod -R g+rwX /app

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY --chown=1001:0 requirements.txt .

# Install extra Python dependencies
USER 1001
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the rest of the application
COPY --chown=1001:0 . .

# Create models directory
RUN mkdir -p /app/models \
    && chown -R 1001:0 /app/models \
    && chmod -R g+rwX /app/models

# Expose port for Rasa server
EXPOSE 5005

# Set the user to run the application
USER 1001

# Set the command to run Rasa
CMD ["run", "--enable-api", "--cors", "*", "--debug", "--endpoints", "endpoints.yml"]
