## Running Qdrant

```
docker run -p 7000:6333 -p 7001:6334 -v "$(pwd)//Qdrant:/qdrant/storage:z" qdrant/qdrant
```

alembic revision --autogenerate -m "add area to log table"
alembic upgrade head
docker run --gpus all -v "${pwd}:/app" -p 8000:8000 -w /app face_recognition:v1 python3 run.py

```
docker run --gpus all -v "${pwd}:/app" -p 8000:8000 -w /app cuda_torch_tensorrt_detection_extra1:latest uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

```
docker run --gpus all -it -v "${pwd}:/app" face_qdrant:latest bash

-   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

-   /usr/local/bin/qdrant --config-path /etc/qdrant/config.yaml

```

docker build -t ai_qdrant_combined .

'python webcam_retina_arcface.py --camera 0 --collection n3 --threshold 0.5'

'uvicorn app.main:app --reload --host 0.0.0.0 --port 8000'
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

docker run --gpus all -v "${pwd}:/app" -p 8000:8000 -w /app ai_qdrant_combined:latest uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

```

```
