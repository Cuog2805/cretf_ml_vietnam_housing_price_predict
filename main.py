from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Prediction Microservice",
    description="Microservice for real estate price prediction with standardized format",
    version="2.0.0"
)

# CORS for microservice communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Spring Boot URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load standardized models
try:
    if os.path.exists('models/standardized_model.pkl'):
        logger.info("Loading standardized model...")
        model = joblib.load('models/standardized_model.pkl')
        encoders = joblib.load('models/standardized_encoders.pkl')
        scaler = joblib.load('models/standardized_scaler.pkl')

        with open('models/standardized_model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        feature_columns = model_info['feature_columns']
        categorical_features = model_info['categorical_features']
        model_type = "standardized"
        logger.info("Standardized model loaded successfully!")

    elif os.path.exists('models/my_custom_model.pkl'):
        logger.info("Loading custom model...")
        model = joblib.load('models/my_custom_model.pkl')
        encoders = joblib.load('models/my_encoders.pkl')
        scaler = joblib.load('models/my_scaler.pkl')
        with open('models/my_features.json', 'r') as f:
            feature_columns = json.load(f)
        categorical_features = ['district', 'province']
        model_type = "legacy_custom"
        logger.info("Legacy custom model loaded successfully!")

    else:
        logger.error("Model files not found")
        model = None
        model_type = None

except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    model_type = None


# Pydantic models with standardized format
class PredictRequest(BaseModel):
    district: str
    province: str
    area: float
    frontage: Optional[float] = 0
    access_road: Optional[float] = 0
    direction: Optional[str] = "Unknown"
    property_type: str
    floors: Optional[int] = 1
    bedrooms: Optional[int] = 2
    bathrooms: Optional[int] = 1


class PredictResponse(BaseModel):
    success: bool
    data: float
    message: str
    timestamp: str
    model_version: str
    input_summary: dict


class ModelInfoResponse(BaseModel):
    model_type: str
    features: list
    categorical_features: list
    input_format: dict


def clean_location_input(district, province):
    import re

    # Clean district - remove all prefixes
    clean_district = re.sub(r'^(Huyện|Thành phố|Quận|Thị xã|TP)\s+', '', district.strip())

    # Clean province - remove prefixes
    clean_province = re.sub(r'^(Tỉnh|Thành phố|TP)\s+', '', province.strip())

    logger.info(f"Location cleanup: '{district}' → '{clean_district}', '{province}' → '{clean_province}'")

    return clean_district, clean_province


def encode_categorical_safe(value, encoder_name):
    if encoder_name not in encoders:
        logger.warning(f"No encoder found for {encoder_name}")
        return 0

    encoder = encoders[encoder_name]

    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        logger.warning(f"Unseen value '{value}' for {encoder_name}, using default encoding")
        return 0


# Global variables for health check
start_time = datetime.now()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model service unavailable")

    try:
        logger.info(f"Raw prediction request: {request.dict()}")

        # Clean location input - remove prefixes if present
        clean_district, clean_province = clean_location_input(request.district, request.province)

        # Log cleaned input
        logger.info(f"Cleaned input: district='{clean_district}', province='{clean_province}'")

        if model_type == "standardized":
            # Use standardized model prediction
            features_dict = {
                'Area': float(request.area),
                'Frontage': float(request.frontage),
                'Access Road': float(request.access_road),
                'Floors': float(request.floors),
                'Bedrooms': float(request.bedrooms),
                'Bathrooms': float(request.bathrooms)
            }

            # Encode categorical features using cleaned values
            categorical_mapping = {
                'District': clean_district,  # Use cleaned district
                'Province': clean_province,  # Use cleaned province
                'Direction': request.direction,
                'Type': request.property_type
            }

            for cat_feature in categorical_features:
                encoded_value = encode_categorical_safe(categorical_mapping[cat_feature], cat_feature)
                features_dict[f'{cat_feature}_encoded'] = encoded_value

            # Create prediction DataFrame
            input_df = pd.DataFrame([features_dict], columns=feature_columns)

            # Scale features
            input_scaled = scaler.transform(input_df)

        else:
            # Legacy model prediction (backward compatibility)
            df = pd.DataFrame({
                'area': [request.area],
                'bedrooms': [request.bedrooms],
                'bathrooms': [request.bathrooms],
                'district': [clean_district],  # Use cleaned district
                'province': [clean_province]  # Use cleaned province
            })

            # Encode categorical variables
            df['district_encoded'] = encode_categorical_safe(clean_district, 'district')
            df['province_encoded'] = encode_categorical_safe(clean_province, 'province')

            # Prepare features
            X = df[['area', 'bedrooms', 'bathrooms', 'district_encoded', 'province_encoded']]
            input_scaled = scaler.transform(X)

        # Predict
        predicted_price = model.predict(input_scaled)[0]
        predicted_price = max(0, predicted_price)  # Ensure positive

        logger.info(f"Prediction result: {predicted_price:.2f} tỷ VNĐ")

        # Input summary for response - show both original and cleaned values
        input_summary = {
            "location": {
                "original": f"{request.district}, {request.province}",
                "cleaned": f"{clean_district}, {clean_province}",
                "note": "Prefixes automatically removed for model processing"
            },
            "area": f"{request.area}m²",
            "type": request.property_type,
            "bedrooms": request.bedrooms,
            "bathrooms": request.bathrooms,
            "additional_features": {
                "frontage": f"{request.frontage}m" if request.frontage > 0 else "Not specified",
                "access_road": f"{request.access_road}m" if request.access_road > 0 else "Not specified",
                "direction": request.direction if request.direction != "Unknown" else "Not specified",
                "floors": request.floors
            }
        }

        return PredictResponse(
            success=True,
            data=round(predicted_price, 2),
            message="Prediction successful",
            timestamp=datetime.now().isoformat(),
            model_version=model_type,
            input_summary=input_summary
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(requests: list[PredictRequest]):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model service unavailable")

    results = []

    for i, request in enumerate(requests):
        try:
            # Reuse single prediction logic
            prediction_response = await predict(request)
            results.append({
                "index": i,
                "success": True,
                "prediction": prediction_response.data,
                "input": request.dict()
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e),
                "input": request.dict()
            })

    return {
        "batch_results": results,
        "total_requests": len(requests),
        "successful_predictions": sum(1 for r in results if r["success"]),
        "timestamp": datetime.now().isoformat()
    }


# For microservice deployment
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="localhost",  # Allow external connections
        port=8083,  # Different port for microservice
        log_level="info"
    )