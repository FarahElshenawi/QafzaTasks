# FastAPI ML Application: Diabetes Prediction Model

This repository contains a FastAPI application that serves a machine learning model for predicting diabetes. The project includes everything from model training to deployment, using Docker for containerization.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
  - [Modeling](#modeling)
  - [API Development with FastAPI](#api-development-with-fastapi)
  - [Dockerization](#dockerization)
- [Docker Hub Repository](#docker-hub-repository)


## Project Overview

The Diabetes Prediction model is built using a machine learning algorithm to predict the likelihood of a person developing diabetes. The project was developed with a focus on creating a REST API using FastAPI for model inference. The model was trained, saved as a serialized `.sav` file, and served via an API endpoint. This application is then containerized using Docker for easy deployment.

## Technologies Used

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.9+.
- **Scikit-Learn**: A machine learning library for building the diabetes prediction model.
- **Pickle**: A Python module used to serialize the model.
- **Docker**: Used to containerize the application.
- **Uvicorn**: ASGI server used to run FastAPI applications.
- **Python**: The programming language used to implement the model and API.
  
## Setup Instructions

### Modeling

1. The first step in the project was to train a machine learning model to predict diabetes. The dataset was preprocessed, and a classification model (e.g., Logistic Regression, Random Forest, etc.) was used for training.
2. The trained model was serialized using `pickle` and saved as `diabetes_model.sav`.
3. The model file is then used for inference in the FastAPI application.

### API Development with FastAPI

1. A FastAPI app was created to expose an endpoint for predicting diabetes based on input features.
2. The model file is loaded in the FastAPI app, and the `POST` endpoint takes input data, processes it, and returns a prediction.
3. The application was tested locally before containerizing it.

### Dockerization

1. A Dockerfile was created to containerize the FastAPI application. The necessary dependencies such as `uvicorn`, `fastapi`, and `scikit-learn` were added to the Docker image.
2. The application was built into a Docker image, and the container was run locally to test the deployment.
3. Finally, the Docker image was pushed to Docker Hub for easier sharing and deployment.

## Docker Hub Repository
You can find the Docker image on Docker Hub [here](https://hub.docker.com/repository/docker/farahelshenawy/my-fastapi-ml-app/general).
