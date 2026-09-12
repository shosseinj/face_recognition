# Face Recognition Service with Vector Search

This repository contains an experimental face-recognition backend that combines GPU-based face processing with vector similarity search and a web API. It includes application code, database migrations, Docker deployment material, and Qdrant integration used during development.

## Components

The working tree contains:

- a backend API service;
- face-processing modules;
- Qdrant vector search;
- database migrations through Alembic;
- Docker/GPU execution paths;
- utilities for generating and testing detection records.

## Development Services

Qdrant can be started locally with Docker and the API can be run with Uvicorn. Exact model paths, collection names, thresholds, and deployment parameters should be configured for the target environment rather than hard-coded into application code.

## Scope

The project explores end-to-end integration of face detection/recognition, embedding search, API services, and persistent metadata. It is primarily an engineering prototype and should not be treated as a biometric benchmark implementation.

## Privacy and Data Handling

Face images, embeddings, identity metadata, and application databases are sensitive biometric data. Public repositories should contain only synthetic or explicitly shareable examples. Local Qdrant storage and application database files should be excluded from version control for a public research portfolio.
