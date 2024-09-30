# Chess AI Engine: Play Chess with an AI-Powered Game

Welcome to the **Chess AI Engine** project! This is a web-based chess game where you can play against an AI that predicts moves using a neural network trained on thousands of chess games. The project seamlessly integrates a frontend for playing chess and a backend that processes AI moves using machine learning.

Due to processing power constraints, the AI was trained on a subset of 100,000 games out of a much larger dataset (2.5 million games). Despite the limited dataset, the AI is capable of competitive move predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Backend (Chess Engine)](#backend-chess-engine)
  - [Frontend (Chess UI)](#frontend-chess-ui)
- [Setup and Installation](#setup-and-installation)
  - [Running Locally with Docker](#running-locally-with-docker)
  - [Deploying on Google Cloud Run](#deploying-on-google-cloud-run)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

The **Chess AI Engine** project combines frontend and backend components to create a playable chess game with AI-powered move suggestions. The backend hosts a machine learning model trained on thousands of historical chess games, while the frontend enables players to interact with the AI in real-time.

The model has been trained on a limited subset of chess games due to computational constraints, but it provides reasonable move suggestions for a casual chess-playing experience.

## Dataset

The dataset used for training the AI model consists of over 2.5 million chess games, but due to constraints, only 100,000 games were used. You can access and download the full dataset from [here](https://database.nikonoel.fr/).

## Features

- **Play Against AI**: Interact with a chess AI that predicts moves based on historical game data.
- **User-Friendly Interface**: The chessboard is rendered in the browser, featuring move highlighting and interactive controls.
- **Real-Time AI Responses**: The AI responds to each player move by suggesting its own move based on its training data.
- **Cloud Deployment**: The backend will be deployed on Google Cloud Run, making the AI accessible from anywhere.

## How It Works

### Backend (Chess Engine)

The backend is built using Python and Flask, serving as an API for the AI chess engine. It consists of:
- **Flask API**: Handles incoming requests for move predictions. It takes a FEN (Forsyth-Edwards Notation) string as input and returns the best move based on the model's output.
- **PyTorch Model**: The neural network trained on chess games to predict the next best move based on the current board state.
- **Move Encoding**: Moves are encoded as integers and decoded from integers back to UCI format using the `move_to_int.pkl` mapping file.

### Frontend (Chess UI)

The frontend is designed for player interaction:
- **HTML/JavaScript/CSS**: The chessboard is rendered with `chessboard.js`, and game logic is handled by `chess.js`. Custom JavaScript handles user interactions, drag-and-drop moves, and communication with the backend API.
- **Responsive Design**: Ensures that the game works across devices, from mobile to desktop.

## Setup and Installation

### Running Locally with Docker

#### Prerequisites
- Ensure you have [Docker](https://www.docker.com/get-started) installed on your machine.

#### Running the Backend

The backend is containerized with Docker. To run the backend locally, follow these steps:

1. **Build the Docker Image**:
   ```bash
   docker build -t chess-engine .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 chess-engine
   ```

This will start a Flask server running inside a Docker container, accessible at `http://localhost:5000`.

#### Running the Frontend

Open the `index.html` file in your browser. You can play chess directly in the browser, and the frontend will communicate with the Dockerized backend to fetch AI-predicted moves.

### Deploying on Google Cloud Run

The backend can be deployed on Google Cloud Run for scalable, serverless hosting.

Steps to Deploy on Cloud Run:

1. **Build and Push the Docker Image**:

   First, build your Docker image and push it to Google Container Registry:

   ```bash
   docker build -t gcr.io/[YOUR_PROJECT_ID]/chess-engine .
   docker push gcr.io/[YOUR_PROJECT_ID]/chess-engine
   ```

2. **Deploy to Google Cloud Run**:

   Use the Google Cloud CLI to deploy your image to Cloud Run:

   ```bash
   gcloud run deploy chess-engine \
       --image gcr.io/[YOUR_PROJECT_ID]/chess-engine \
       --platform managed \
       --region [YOUR_REGION] \
       --allow-unauthenticated
   ```

   Replace `[YOUR_PROJECT_ID]` and `[YOUR_REGION]` with your project details. This will deploy the backend, and you will receive a public URL for accessing the API.

3. **Update Frontend to Point to Cloud Backend**:

   Modify your `script.js` in the frontend to use the public URL provided by Cloud Run instead of localhost.

## Usage

### Playing the Game:

Open the frontend (`index.html`) in your browser. You can start playing immediately, making moves on the chessboard. The AI will respond with its predicted move in real-time.

### Backend Interaction:

The frontend sends requests to the backend API, whether running locally in Docker or deployed on Google Cloud Run, which processes the request and returns the AI's move.

## Future Improvements

1. **Expand Training Dataset**:
   Increasing the number of games used for training (from 100,000 to the full 2.5 million) could improve the AI's accuracy and performance.

2. **More Epochs**:
   Training the model for more epochs would likely result in a more refined AI with better performance.

3. **Merge with Position Analyzer**:
   The next phase of development will integrate the Position Analyzer project, allowing the AI to analyze specific board positions in-depth while also predicting the next best move.
