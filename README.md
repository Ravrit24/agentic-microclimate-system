# Agentic AI-Based Microclimate Decision Support System

## Overview

This project is an AI-powered precision agriculture platform designed to help smallholder farmers make data-driven irrigation and crop management decisions using microclimate forecasting and crop stress analysis.

The system leverages multi-year environmental time-series data and machine learning models to predict future conditions, classify crop health status, and generate actionable recommendations through an interactive Streamlit dashboard.

## Features

* LSTM-based microclimate forecasting
* Crop stress classification (SAFE / WARNING / STRESS)
* Decision-support engine for irrigation planning
* Interactive Streamlit dashboard
* Real-time prediction and visualization

## Machine Learning Models

* LSTM Forecasting Model
* Feed-Forward Neural Network (TFFN)
* Crop Stress Classifier
* Baseline Random Forest Model

## Dataset

The project uses microclimate and agricultural sensor data including:

* Temperature
* Humidity
* Rainfall
* Soil moisture
* Environmental variables

Dataset Size: 100,000+ time-series records

## Results

* Crop Stress Classification Accuracy: ~85%
* Forecasting Performance Improvement: ~18% over baseline models
* Evaluated using RMSE, MAE, and R² metrics

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-Learn
* Pandas
* NumPy
* Streamlit

## Repository Structure

config/ – configuration files

data/ – dataset files

models/ – trained machine learning models

src/ – preprocessing, training, inference, and decision engine code

app.py – Streamlit application

requirements.txt – project dependencies

## Future Improvements

* Integration with live weather APIs
* Mobile-friendly deployment
* Enhanced crop recommendation engine
* Larger-scale field validation
