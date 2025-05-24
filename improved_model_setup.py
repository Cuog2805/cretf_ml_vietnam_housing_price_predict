# standardized_model_trainer.py - Train model với dữ liệu chuẩn hóa
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import re

print("STANDARDIZED Model Training...")

# Tạo thư mục
os.makedirs("models", exist_ok=True)

# Đọc dữ liệu
with open('data/vietnam_housing_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Raw data: {len(df)} records")

# 1. CHUẨN HÓA DỮ LIỆU VỀ FORMAT MỚI
print("\nStandardizing data format...")


def extract_location(address):
    """Extract district and province from address"""
    if pd.isna(address) or address == "":
        return "Unknown", "Unknown"

    parts = [part.strip() for part in address.split(',')]

    if len(parts) >= 2:
        # Province thường là phần cuối
        province = parts[-1]
        # District thường là phần gần cuối
        district = parts[-2] if len(parts) >= 2 else "Unknown"

        # Clean up common prefixes chỉ cho province
        province = re.sub(r'^(Tỉnh|Thành phố|TP)\s+', '', province)

        # Format district: Bỏ tất cả tiền tố
        district = re.sub(r'^(Huyện|Thành phố|Quận|Thị xã|TP)\s+', '', district)

        return district, province
    else:
        return "Unknown", "Unknown"


def determine_type_and_direction(row):
    """Determine property type and direction"""
    house_dir = row.get('House direction', '')
    balcony_dir = row.get('Balcony direction', '')

    # Xác định Type
    if pd.notna(house_dir) and house_dir != '':
        property_type = 'Nhà'
        direction = house_dir
    elif pd.notna(balcony_dir) and balcony_dir != '':
        property_type = 'Căn hộ'
        direction = balcony_dir
    else:
        property_type = 'Unknown'
        direction = 'Unknown'

    return property_type, direction


# Apply transformations
print("Extracting location info...")
location_info = df['Address'].apply(extract_location)
df['District'] = [loc[0] for loc in location_info]
df['Province'] = [loc[1] for loc in location_info]

print("Determining Type and Direction...")
type_direction_info = df.apply(determine_type_and_direction, axis=1)
df['Type'] = [info[0] for info in type_direction_info]
df['Direction'] = [info[1] for info in type_direction_info]

print(f"Extracted {df['District'].nunique()} districts and {df['Province'].nunique()} provinces")
print(f"Property types: {df['Type'].value_counts().to_dict()}")

# Hiển thị sample District formatting
print(f"\nSample District formatting:")
sample_districts = df['District'].value_counts().head(5)
for district, count in sample_districts.items():
    print(f"  {district} ({count} properties)")

# 2. CHUẨN HÓA CÁC TRƯỜNG KHÁC
print("\nStandardizing other fields...")

# Rename và standardize columns
column_mapping = {
    'Area': 'Area',
    'Frontage': 'Frontage',
    'Access Road': 'Access Road',
    'Floors': 'Floors',
    'Bedrooms': 'Bedrooms',
    'Bathrooms': 'Bathrooms',
    'Price': 'Price'
}

# Đảm bảo tất cả columns tồn tại
for old_col, new_col in column_mapping.items():
    if old_col not in df.columns:
        df[old_col] = np.nan

# 3. DATA CLEANING
print("\nCleaning data...")

# Xóa rows có price hoặc area invalid
initial_count = len(df)
df = df.dropna(subset=['Price', 'Area'])
df = df[df['Price'] > 0]  # Price > 0
df = df[df['Area'] > 0]  # Area > 0

print(f"After basic cleaning: {len(df)} records ({initial_count - len(df)} removed)")

# 4. HANDLE NUMERIC FIELDS
print("\nProcessing numeric fields...")

# Convert numeric fields and handle missing values
numeric_fields = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']

for field in numeric_fields:
    # Convert to numeric, coerce errors to NaN
    df[field] = pd.to_numeric(df[field], errors='coerce')

    # Fill missing values
    if field in ['Bedrooms', 'Bathrooms']:
        df[field] = df[field].fillna(df[field].median())
    elif field == 'Floors':
        df[field] = df[field].fillna(1)  # Default 1 tầng
    elif field in ['Frontage', 'Access Road']:
        df[field] = df[field].fillna(0)  # 0 = không có data

print("Numeric fields summary:")
for field in numeric_fields:
    print(f"  {field}: mean={df[field].mean():.1f}, range=[{df[field].min():.1f}, {df[field].max():.1f}]")

# 5. HANDLE CATEGORICAL FIELDS
print("\nProcessing categorical fields...")

categorical_fields = ['District', 'Province', 'Direction', 'Type']
for field in categorical_fields:
    df[field] = df[field].fillna('Unknown')
    print(f"  {field}: {df[field].nunique()} unique values")

# 6. OUTLIER REMOVAL
print("\nHandling outliers...")


def remove_outliers(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    before = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    after = len(df_clean)

    print(f"  {column}: removed {before - after} outliers ({(before - after) / before * 100:.1f}%)")
    return df_clean


df = remove_outliers(df, 'Price', factor=2.0)
df = remove_outliers(df, 'Area', factor=2.0)

print(f"After outlier removal: {len(df)} records")

# 7. CHUẨN BỊ DỮ LIỆU CUỐI CÙNG
standardized_features = ['District', 'Province', 'Area', 'Frontage', 'Access Road',
                         'Direction', 'Type', 'Floors', 'Bedrooms', 'Bathrooms']

df_final = df[standardized_features + ['Price']].copy()

print(f"\nFinal standardized dataset:")
print(f"  Records: {len(df_final)}")
print(f"  Features: {len(standardized_features)}")

# Display sample
print(f"\nSample data:")
sample = df_final.head(3)[standardized_features].to_dict('records')
for i, row in enumerate(sample):
    print(f"  Sample {i + 1}: {row}")

# 8. SAVE STANDARDIZED DATA
print(f"\nSaving standardized data...")
standardized_data = df_final.to_dict('records')
with open('data/standardized_data.json', 'w', encoding='utf-8') as f:
    json.dump(standardized_data, f, ensure_ascii=False, indent=2)

print("Standardized data saved to 'data/standardized_data.json'")

# 9. CHECK CORRELATIONS
print(f"\nCorrelations with price:")
numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms', 'Price']
correlations = df_final[numeric_cols].corr()['Price'].sort_values(ascending=False)
for col, corr in correlations.items():
    if col != 'Price':
        print(f"  {col}: {corr:.3f}")

# 10. ENCODE CATEGORICAL VARIABLES
print("\nEncoding categorical variables...")

encoders = {}
categorical_features = ['District', 'Province', 'Direction', 'Type']

for feature in categorical_features:
    le = LabelEncoder()
    df_final[f'{feature}_encoded'] = le.fit_transform(df_final[feature])
    encoders[feature] = le
    print(f"  {feature}: {len(le.classes_)} categories")

# 11. PREPARE FINAL FEATURES
feature_columns = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
feature_columns += [f'{cat}_encoded' for cat in categorical_features]

X = df_final[feature_columns]
y = df_final['Price']

print(f"\nFinal dataset for training:")
print(f"  Features shape: {X.shape}")
print(f"  Feature columns: {feature_columns}")
print(f"  Price range: {y.min():.2f} - {y.max():.2f} tỷ VNĐ")
print(f"  Price mean: {y.mean():.2f} tỷ VNĐ")

# 12. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 13. STANDARDIZE FEATURES
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 14. TRAIN MODEL
print("\n🌲 Training RandomForest...")
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# 15. EVALUATE
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nMODEL PERFORMANCE:")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")
print(f"Overfitting gap: {abs(train_r2 - test_r2):.4f}")
print(f"Test RMSE: {test_rmse:.2f} tỷ VNĐ")
print(f"Test MAE: {test_mae:.2f} tỷ VNĐ")

# 16. FEATURE IMPORTANCE
print(f"\nFeature Importance:")
importances = model.feature_importances_
feature_importance = list(zip(feature_columns, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for name, importance in feature_importance:
    print(f"  {name}: {importance:.4f}")

# 17. SAVE MODEL
if test_r2 > 0.3:
    print(f"\nSaving standardized model...")
    joblib.dump(model, 'models/standardized_model.pkl')
    joblib.dump(encoders, 'models/standardized_encoders.pkl')
    joblib.dump(scaler, 'models/standardized_scaler.pkl')

    # Save feature info
    model_info = {
        'feature_columns': feature_columns,
        'categorical_features': categorical_features,
        'input_format': {
            'District': 'string (Huyện → tên huyện, Thành phố → giữ nguyên)',
            'Province': 'string',
            'Area': 'numeric',
            'Frontage': 'numeric',
            'Access Road': 'numeric',
            'Direction': 'string',
            'Type': 'string (Nhà/Căn hộ)',
            'Floors': 'numeric',
            'Bedrooms': 'numeric',
            'Bathrooms': 'numeric'
        }
    }

    with open('models/standardized_model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("Standardized model saved successfully!")

    # Test prediction với format mới
    print(f"\nTest prediction with standardized format:")
    test_input = {
        "District": "Văn Giang",
        "Province": "Hưng Yên",
        "Area": 84,
        "Frontage": 5,
        "Access Road": 8,
        "Direction": "Đông - Bắc",
        "Type": "Nhà",
        "Floors": 4,
        "Bedrooms": 3,
        "Bathrooms": 2
    }

    # Manual prediction để test
    features_dict = {
        'Area': test_input['Area'],
        'Frontage': test_input['Frontage'],
        'Access Road': test_input['Access Road'],
        'Floors': test_input['Floors'],
        'Bedrooms': test_input['Bedrooms'],
        'Bathrooms': test_input['Bathrooms']
    }

    # Encode categorical
    for cat_feature in categorical_features:
        encoder = encoders[cat_feature]
        value = test_input[cat_feature]
        if value in encoder.classes_:
            features_dict[f'{cat_feature}_encoded'] = encoder.transform([value])[0]
        else:
            features_dict[f'{cat_feature}_encoded'] = 0

    # Create prediction DataFrame với feature names
    sample_df = pd.DataFrame([features_dict], columns=feature_columns)
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)[0]

    print(f"Input: {test_input}")
    print(f"Predicted Price: {prediction:.2f} tỷ VNĐ")

else:
    print(f"\nModel performance too poor (R2 = {test_r2:.4f}). Not saving.")
    print("Please check your data quality!")

print(f"\nStandardized training completed!")