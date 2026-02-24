
# Use your existing AI app image as base
FROM cuda_torch_tensorrt_detection_extra1:latest



RUN pip install --no-cache qdrant-client
WORKDIR /app
COPY . /app

# RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# COPY supervisord.conf /etc/supervisord.conf

# # Expose ports
# EXPOSE 8000 

# # Start Supervisor
# CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
