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
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Prediction",
)

# CORS for microservice communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Spring Boot URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
scaler = None
location_encoders = None
model_info = None
model_type = None


# Location tier and city functions
def create_location_tiers():
    location_tiers = {}

    # Tier 5: Premium Central HN/HCM
    tier_5_districts = [
        'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Đống Đa', 'Hai Bà Trưng', 'Cầu Giấy',
        'Quận 1', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 7', 'Quận 10', 'Bình Thạnh'
    ]

    # Tier 4: Expanded HN/HCM
    tier_4_districts = [
        'Nam Từ Liêm', 'Bắc Từ Liêm', 'Thanh Xuân', 'Long Biên',
        'Quận 2', 'Quận 6', 'Quận 8', 'Quận 9', 'Quận 11', 'Quận 12', 'Thủ Đức'
    ]

    # Tier 3: Provincial Cities
    tier_3_districts = [
        'Vinh', 'Huế', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng', 'Nha Trang', 'Vũng Tàu',
        'Thái Nguyên', 'Nam Định', 'Hải Dương', 'Bắc Ninh', 'Hưng Yên'
    ]

    # Tier 2: Suburban HN/HCM
    tier_2_districts = [
        'Gia Lâm', 'Đông Anh', 'Sóc Sơn', 'Mê Linh', 'Chương Mỹ', 'Hoài Đức',
        'Gò Vấp', 'Tân Bình', 'Tân Phú', 'Phú Nhuận', 'Bình Tân', 'Hóc Môn'
    ]

    # Assign tiers
    for district in tier_5_districts:
        location_tiers[district] = 5.0

    for district in tier_4_districts:
        location_tiers[district] = 4.0

    for district in tier_3_districts:
        location_tiers[district] = 3.0

    for district in tier_2_districts:
        location_tiers[district] = 2.0

    # Tier 1 is default (1.0) for any district not in the above lists

    return location_tiers


def get_city_flags(district):
    """Determine city classification based on 5-tier system (matching create_location_tiers)"""

    # Tier 5: Premium Central HN/HCM
    tier_5_hanoi = ['Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Đống Đa', 'Hai Bà Trưng', 'Cầu Giấy']
    tier_5_hcmc = ['Quận 1', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 7', 'Quận 10', 'Bình Thạnh']

    # Tier 4: Expanded HN/HCM
    tier_4_hanoi = ['Nam Từ Liêm', 'Bắc Từ Liêm', 'Thanh Xuân', 'Long Biên']
    tier_4_hcmc = ['Quận 2', 'Quận 6', 'Quận 8', 'Quận 9', 'Quận 11', 'Quận 12', 'Thủ Đức']

    # Tier 2: Suburban HN/HCM (Note: Tier 3 is provincial cities, not HN/HCM)
    tier_2_hanoi = ['Gia Lâm', 'Đông Anh', 'Sóc Sơn', 'Mê Linh', 'Chương Mỹ', 'Hoài Đức']
    tier_2_hcmc = ['Gò Vấp', 'Tân Bình', 'Tân Phú', 'Phú Nhuận', 'Bình Tân', 'Hóc Môn']

    # Combine all HN/HCM districts
    all_hanoi = tier_5_hanoi + tier_4_hanoi + tier_2_hanoi
    all_hcmc = tier_5_hcmc + tier_4_hcmc + tier_2_hcmc

    is_hanoi = 1 if district in all_hanoi else 0
    is_hcmc = 1 if district in all_hcmc else 0
    is_major_city = 1 if (is_hanoi or is_hcmc) else 0

    return is_hanoi, is_hcmc, is_major_city


def clean_location_input(district, province):
    """Clean location input by removing prefixes"""
    # Clean district - remove all prefixes
    clean_district = re.sub(r'^(Huyện|Thành phố|Quận|Thị xã|TP)\s+', '', district.strip())

    # Clean province - remove prefixes
    clean_province = re.sub(r'^(Tỉnh|Thành phố|TP)\s+', '', province.strip())

    logger.info(f"Location cleanup: '{district}' → '{clean_district}', '{province}' → '{clean_province}'")

    return clean_district, clean_province


# Load enhanced models
try:
    if os.path.exists('models/enhanced_standardized_model.pkl'):
        logger.info("Loading enhanced standardized model...")
        model = joblib.load('models/enhanced_standardized_model.pkl')
        scaler = joblib.load('models/enhanced_standardized_scaler.pkl')
        location_encoders = joblib.load('models/enhanced_location_encoders.pkl')

        with open('models/enhanced_standardized_model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        model_type = "enhanced_location_aware"
        logger.info("Enhanced model loaded successfully!")
        logger.info(f"Features: {len(model_info['feature_columns'])}")

    elif os.path.exists('models/standardized_model.pkl'):
        logger.info("Loading fallback standardized model...")
        model = joblib.load('models/standardized_model.pkl')
        encoders = joblib.load('models/standardized_encoders.pkl')
        scaler = joblib.load('models/standardized_scaler.pkl')

        with open('models/standardized_model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        model_type = "standardized_fallback"
        logger.info("Fallback model loaded successfully!")

    else:
        logger.error("No model files found")
        model = None
        model_type = None

except Exception as e:
    logger.error(f"Error loading models: {e}")
    model = None
    model_type = None


# Pydantic models
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
    location_insights: Optional[dict] = None


class ModelInfoResponse(BaseModel):
    model_type: str
    features: list
    performance: dict
    tier_system: Optional[dict] = None


def get_safe_encoded_value(value, mapping, default_value):
    """Safely get encoded value from mapping with fallback"""
    if value in mapping:
        return mapping[value]
    else:
        logger.warning(f"Unseen value '{value}', using default: {default_value}")
        return default_value


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model service unavailable")

    try:
        logger.info(f"Prediction request: {request.dict()}")

        # Clean location input
        clean_district, clean_province = clean_location_input(request.district, request.province)

        if model_type == "enhanced_location_aware":
            # Enhanced model prediction with new features

            # Get location encoders with fallback to ensure consistency
            district_mean_map = location_encoders['district_mean_map']
            province_mean_map = location_encoders['province_mean_map']

            # Use location_tiers from encoders if available, otherwise create fresh
            if 'location_tiers' in location_encoders:
                location_tiers = location_encoders['location_tiers']
            else:
                logger.warning("Location tiers not found in encoders, creating fresh tiers")
                location_tiers = create_location_tiers()

            direction_mean_map = location_encoders['direction_mean_map']
            type_mean_map = location_encoders['type_mean_map']

            # Calculate global mean as fallback
            all_prices = list(district_mean_map.values()) + list(province_mean_map.values())
            global_mean = np.mean(all_prices) if all_prices else 5.0

            # Encode categorical features with mean target encoding
            district_encoded = get_safe_encoded_value(clean_district, district_mean_map, global_mean)
            province_encoded = get_safe_encoded_value(clean_province, province_mean_map, global_mean)
            direction_encoded = get_safe_encoded_value(request.direction, direction_mean_map, global_mean)
            type_encoded = get_safe_encoded_value(request.property_type, type_mean_map, global_mean)

            # Get location tier
            location_tier = location_tiers.get(clean_district, 1.0)

            # Get city flags
            is_hanoi, is_hcmc, is_major_city = get_city_flags(clean_district)

            # Create feature vector
            # ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
            #  'District_mean_encoded', 'Province_mean_encoded', 'Location_Tier',
            #  'Is_Hanoi', 'Is_HCMC', 'Is_Major_City',
            #  'Direction_mean_encoded', 'Type_mean_encoded']

            features = [
                float(request.area),
                float(request.frontage),
                float(request.access_road),
                float(request.floors),
                float(request.bedrooms),
                float(request.bathrooms),
                district_encoded,
                province_encoded,
                location_tier,
                is_hanoi,
                is_hcmc,
                is_major_city,
                direction_encoded,
                type_encoded
            ]

            # Scale features
            input_scaled = scaler.transform([features])

            # Location insights for response
            tier_names = {
                5.0: "Premium Central (HN/HCM center)",
                4.0: "Expanded Areas (HN/HCM expanded)",
                3.0: "Provincial Cities",
                2.0: "Suburban (HN/HCM suburbs)",
                1.0: "Other Districts/Counties"
            }

            location_insights = {
                "location_tier": location_tier,
                "tier_description": tier_names.get(location_tier, "Unknown"),
                "is_hanoi": bool(is_hanoi),
                "is_hcmc": bool(is_hcmc),
                "is_major_city": bool(is_major_city),
                "district_avg_price": f"{district_encoded:.2f} tỷ VNĐ",
                "province_avg_price": f"{province_encoded:.2f} tỷ VNĐ"
            }

        else:
            # Fallback to standard model
            logger.info("Using fallback standardized model")

            features_dict = {
                'Area': float(request.area),
                'Frontage': float(request.frontage),
                'Access Road': float(request.access_road),
                'Floors': float(request.floors),
                'Bedrooms': float(request.bedrooms),
                'Bathrooms': float(request.bathrooms)
            }

            # Standard encoding (if available)
            if 'encoders' in locals():
                categorical_mapping = {
                    'District': clean_district,
                    'Province': clean_province,
                    'Direction': request.direction,
                    'Type': request.property_type
                }

                for cat_feature in model_info.get('categorical_features', []):
                    if cat_feature in encoders:
                        encoder = encoders[cat_feature]
                        value = categorical_mapping[cat_feature]
                        if value in encoder.classes_:
                            encoded_value = encoder.transform([value])[0]
                        else:
                            encoded_value = 0
                        features_dict[f'{cat_feature}_encoded'] = encoded_value

            # Create feature vector
            feature_columns = model_info['feature_columns']
            input_df = pd.DataFrame([features_dict], columns=feature_columns)
            input_scaled = scaler.transform(input_df)

            location_insights = None

        # Make prediction
        predicted_price = model.predict(input_scaled)[0]
        predicted_price = max(0, predicted_price)  # Ensure positive

        logger.info(f"Prediction result: {predicted_price:.2f} tỷ VNĐ")

        # Input summary
        input_summary = {
            "location": {
                "original": f"{request.district}, {request.province}",
                "cleaned": f"{clean_district}, {clean_province}",
                "note": "Prefixes automatically removed for model processing"
            },
            "property_details": {
                "area": f"{request.area}m²",
                "type": request.property_type,
                "bedrooms": request.bedrooms,
                "bathrooms": request.bathrooms,
                "floors": request.floors
            },
            "additional_features": {
                "frontage": f"{request.frontage}m" if request.frontage > 0 else "Not specified",
                "access_road": f"{request.access_road}m" if request.access_road > 0 else "Not specified",
                "direction": request.direction if request.direction != "Unknown" else "Not specified"
            }
        }

        return PredictResponse(
            success=True,
            data=round(predicted_price, 2),
            message="Prediction successful with enhanced location-aware model" if model_type == "enhanced_location_aware" else "Prediction successful",
            timestamp=datetime.now().isoformat(),
            model_version=model_type,
            input_summary=input_summary,
            location_insights=location_insights
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(requests: list[PredictRequest]):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model service unavailable")

    results = []
    successful_count = 0

    for i, request in enumerate(requests):
        try:
            # Reuse single prediction logic
            prediction_response = await predict(request)
            results.append({
                "index": i,
                "success": True,
                "prediction": prediction_response.data,
                "location_insights": prediction_response.location_insights,
                "input": request.dict()
            })
            successful_count += 1
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
        "successful_predictions": successful_count,
        "failed_predictions": len(requests) - successful_count,
        "model_version": model_type,
        "timestamp": datetime.now().isoformat()
    }


# Global variables for health check
start_time = datetime.now()

# For microservice deployment
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="localhost",  # Allow external connections
        port=8083,  # Different port for microservice
        log_level="info"
    )