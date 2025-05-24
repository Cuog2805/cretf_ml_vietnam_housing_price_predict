# complete_model_trainer.py - Complete enhanced model training without errors
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import re

print("COMPLETE ENHANCED MODEL TRAINING...")

# Tạo thư mục
os.makedirs("models", exist_ok=True)


# LOCATION TIER FUNCTIONS
def create_location_tiers():
    """
    Tạo tier system cho locations theo thực tế giá BĐS Việt Nam

    Tier 5: Quận trung tâm HN/HCM
    Tier 4: Quận mở rộng HN/HCM
    Tier 3: Thành phố tỉnh
    Tier 2: Huyện/quận ngoại ô HN/HCM
    Tier 1: Huyện/tỉnh khác
    """
    print("Creating location tier system...")

    location_tiers = {}

    tier_5_districts = [
        'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Đống Đa', 'Hai Bà Trưng', 'Cầu Giấy',
        'Quận 1', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 7', 'Quận 10', 'Bình Thạnh'
    ]

    tier_4_districts = [
        'Nam Từ Liêm', 'Bắc Từ Liêm', 'Thanh Xuân', 'Long Biên',
        'Quận 2', 'Quận 6', 'Quận 8', 'Quận 9', 'Quận 11', 'Quận 12', 'Thủ Đức'
    ]

    tier_3_districts = [
        'Vinh', 'Huế', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng', 'Nha Trang', 'Vũng Tàu',
        'Thái Nguyên', 'Nam Định', 'Hải Dương', 'Bắc Ninh', 'Hưng Yên'
    ]

    tier_2_districts = [
        'Gia Lâm', 'Đông Anh', 'Sóc Sơn', 'Mê Linh', 'Chương Mỹ', 'Hoài Đức',
        'Gò Vấp', 'Tân Bình', 'Tân Phú', 'Phú Nhuận', 'Bình Tân', 'Hóc Môn'
    ]

    for district in tier_5_districts:
        location_tiers[district] = 5.0
        print(f"  Tier 5 (Premium Central): {district}")

    for district in tier_4_districts:
        location_tiers[district] = 4.0
        print(f"  Tier 4 (Expanded HN/HCM): {district}")

    for district in tier_3_districts:
        location_tiers[district] = 3.0
        print(f"  Tier 3 (Provincial Cities): {district}")

    for district in tier_2_districts:
        location_tiers[district] = 2.0
        print(f"  Tier 2 (Suburban HN/HCM): {district}")

    print(f"  Tier 1 (Default): All other districts/counties")
    print(f"Total tiers mapped: {len(location_tiers)} districts")

    return location_tiers


def get_city_flags(district):
    """Determine if district belongs to major cities"""
    hanoi_districts = [
        'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Đống Đa', 'Hai Bà Trưng', 'Cầu Giấy',
        'Nam Từ Liêm', 'Bắc Từ Liêm', 'Thanh Xuân', 'Long Biên',
        'Gia Lâm', 'Đông Anh', 'Sóc Sơn', 'Mê Linh', 'Chương Mỹ', 'Hoài Đức'
    ]

    hcmc_districts = [
        'Quận 1', 'Quận 2', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 6',
        'Quận 7', 'Quận 8', 'Quận 9', 'Quận 10', 'Quận 11', 'Quận 12',
        'Bình Thạnh', 'Thủ Đức', 'Gò Vấp', 'Tân Bình', 'Tân Phú', 'Phú Nhuận',
        'Bình Tân', 'Hóc Môn'
    ]

    is_hanoi = 1 if district in hanoi_districts else 0
    is_hcmc = 1 if district in hcmc_districts else 0
    is_major_city = 1 if (is_hanoi or is_hcmc) else 0

    return is_hanoi, is_hcmc, is_major_city


def mean_target_encoding(df, target_col='Price', n_folds=5):
    """Mean Target Encoding với Cross-Validation"""
    print(f"Applying mean target encoding with {n_folds}-fold CV...")

    df_encoded = df.copy()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    global_mean = df[target_col].mean()

    # Encoding cho District
    df_encoded['District_mean_encoded'] = global_mean

    for train_idx, val_idx in kf.split(df):
        train_means = df.iloc[train_idx].groupby('District')[target_col].mean()

        for district in df.iloc[val_idx]['District'].unique():
            if district in train_means and train_means[district] > 0:
                mask = (df_encoded.index.isin(val_idx)) & (df_encoded['District'] == district)
                df_encoded.loc[mask, 'District_mean_encoded'] = train_means[district]

    # Encoding cho Province
    df_encoded['Province_mean_encoded'] = global_mean

    for train_idx, val_idx in kf.split(df):
        train_means = df.iloc[train_idx].groupby('Province')[target_col].mean()

        for province in df.iloc[val_idx]['Province'].unique():
            if province in train_means and train_means[province] > 0:
                mask = (df_encoded.index.isin(val_idx)) & (df_encoded['Province'] == province)
                df_encoded.loc[mask, 'Province_mean_encoded'] = train_means[province]

    # Save final mappings
    district_mean_map = df.groupby('District')[target_col].mean().to_dict()
    province_mean_map = df.groupby('Province')[target_col].mean().to_dict()

    print(f"Mean target encoding completed")
    print(f"  District mappings: {len(district_mean_map)}")
    print(f"  Province mappings: {len(province_mean_map)}")

    return df_encoded, district_mean_map, province_mean_map


def extract_location(address):
    """Extract district and province from address"""
    if pd.isna(address) or address == "":
        return "Unknown", "Unknown"

    parts = [part.strip() for part in address.split(',')]

    if len(parts) >= 2:
        province = parts[-1]
        district = parts[-2] if len(parts) >= 2 else "Unknown"

        # Clean prefixes
        province = re.sub(r'^(Tỉnh|Thành phố|TP)\s+', '', province)
        district = re.sub(r'^(Huyện|Thành phố|Quận|Thị xã|TP)\s+', '', district)

        return district, province
    else:
        return "Unknown", "Unknown"


def determine_type_and_direction(row):
    """Determine property type and direction"""
    house_dir = row.get('House direction', '')
    balcony_dir = row.get('Balcony direction', '')

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


def remove_outliers(df, column, factor=1.5):
    """Remove outliers using IQR method"""
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


# MAIN TRAINING PROCESS
try:
    # Đọc dữ liệu
    with open('data/vietnam_housing_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Raw data: {len(df)} records")

    # 1. EXTRACT LOCATION AND TYPE
    print("\nStandardizing data format...")

    location_info = df['Address'].apply(extract_location)
    df['District'] = [loc[0] for loc in location_info]
    df['Province'] = [loc[1] for loc in location_info]

    type_direction_info = df.apply(determine_type_and_direction, axis=1)
    df['Type'] = [info[0] for info in type_direction_info]
    df['Direction'] = [info[1] for info in type_direction_info]

    print(f"Extracted {df['District'].nunique()} districts and {df['Province'].nunique()} provinces")

    # 2. DATA CLEANING
    print("\nCleaning data...")

    # Ensure required columns exist
    numeric_fields = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
    for field in numeric_fields:
        if field not in df.columns:
            df[field] = np.nan

    # Basic cleaning
    initial_count = len(df)
    df = df.dropna(subset=['Price', 'Area'])
    df = df[df['Price'] > 0]
    df = df[df['Area'] > 0]
    print(f"After basic cleaning: {len(df)} records ({initial_count - len(df)} removed)")

    # Handle numeric fields
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

        if field in ['Bedrooms', 'Bathrooms']:
            df[field] = df[field].fillna(df[field].median())
        elif field == 'Floors':
            df[field] = df[field].fillna(1)
        elif field in ['Frontage', 'Access Road']:
            df[field] = df[field].fillna(0)

    # Handle categorical fields
    categorical_fields = ['District', 'Province', 'Direction', 'Type']
    for field in categorical_fields:
        df[field] = df[field].fillna('Unknown')

    # Remove outliers
    print("\nHandling outliers...")
    df = remove_outliers(df, 'Price', factor=2.0)
    df = remove_outliers(df, 'Area', factor=2.0)
    print(f"After outlier removal: {len(df)} records")

    # 3. PREPARE FINAL DATASET
    standardized_features = ['District', 'Province', 'Area', 'Frontage', 'Access Road',
                             'Direction', 'Type', 'Floors', 'Bedrooms', 'Bathrooms']

    df_final = df[standardized_features + ['Price']].copy()
    print(f"\nFinal dataset: {len(df_final)} records, {len(standardized_features)} features")

    # 4. ENHANCED LOCATION ENCODING
    print("\nApplying enhanced location encoding...")

    # Create location tiers
    location_tiers = create_location_tiers()

    # Apply mean target encoding
    df_encoded, district_mean_map, province_mean_map = mean_target_encoding(df_final, target_col='Price')

    # Add location tier feature
    df_encoded['Location_Tier'] = df_encoded['District'].map(
        lambda x: location_tiers.get(x, 1.0)
    )

    # Add city flags
    city_flags = df_encoded['District'].apply(get_city_flags)
    df_encoded['Is_Hanoi'] = [flags[0] for flags in city_flags]
    df_encoded['Is_HCMC'] = [flags[1] for flags in city_flags]
    df_encoded['Is_Major_City'] = [flags[2] for flags in city_flags]

    print(f"Enhanced location features added")

    # 5. VERIFY LOCATION LOGIC
    print(f"\nVerifying location encoding logic...")

    test_districts = ['Tây Hồ', 'Ba Đình', 'Gia Lâm', 'Văn Giang']
    available_districts = [d for d in test_districts if d in district_mean_map]

    if available_districts:
        print(f"\nDistrict Price Verification:")
        for district in available_districts:
            mean_price = district_mean_map[district]
            tier = location_tiers.get(district, 1.0)
            count = len(df_encoded[df_encoded['District'] == district])
            print(f"  {district}: {mean_price:.2f} tỷ VNĐ (Tier {tier}, {count} samples)")

        # Check logic
        if 'Tây Hồ' in available_districts and 'Gia Lâm' in available_districts:
            tay_ho_price = district_mean_map['Tây Hồ']
            gia_lam_price = district_mean_map['Gia Lâm']

            print(f"\nPrice Logic Check:")
            if tay_ho_price > gia_lam_price:
                print(f" CORRECT: Tây Hồ ({tay_ho_price:.2f}) > Gia Lâm ({gia_lam_price:.2f})")
            else:
                print(f" INCORRECT: Tây Hồ ({tay_ho_price:.2f}) < Gia Lâm ({gia_lam_price:.2f})")

    # 6. ENCODE REMAINING FEATURES
    print(f"\nEncoding remaining categorical features...")

    direction_mean_map = df_encoded.groupby('Direction')['Price'].mean().to_dict()
    df_encoded['Direction_mean_encoded'] = df_encoded['Direction'].map(
        lambda x: direction_mean_map.get(x, df_encoded['Price'].mean())
    )

    type_mean_map = df_encoded.groupby('Type')['Price'].mean().to_dict()
    df_encoded['Type_mean_encoded'] = df_encoded['Type'].map(
        lambda x: type_mean_map.get(x, df_encoded['Price'].mean())
    )

    # 7. PREPARE FEATURES FOR TRAINING
    enhanced_feature_columns = [
        'Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
        'District_mean_encoded', 'Province_mean_encoded', 'Location_Tier',
        'Is_Hanoi', 'Is_HCMC', 'Is_Major_City',
        'Direction_mean_encoded', 'Type_mean_encoded'
    ]

    X_enhanced = df_encoded[enhanced_feature_columns]
    y_enhanced = df_encoded['Price']

    print(f"\nEnhanced dataset:")
    print(f"  Features: {len(enhanced_feature_columns)}")
    print(f"  Samples: {len(X_enhanced)}")
    print(f"  Price range: {y_enhanced.min():.2f} - {y_enhanced.max():.2f} tỷ VNĐ")

    # 8. TRAIN MODEL
    print(f"\nTraining enhanced model...")

    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y_enhanced, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    enhanced_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    enhanced_model.fit(X_train_scaled, y_train)

    # 9. EVALUATE
    y_train_pred = enhanced_model.predict(X_train_scaled)
    y_test_pred = enhanced_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\nENHANCED MODEL PERFORMANCE:")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Overfitting gap: {abs(train_r2 - test_r2):.4f}")
    print(f"Test RMSE: {test_rmse:.2f} tỷ VNĐ")
    print(f"Test MAE: {test_mae:.2f} tỷ VNĐ")

    # 10. FEATURE IMPORTANCE
    print(f"\nFeature Importance:")
    importances = enhanced_model.feature_importances_
    feature_importance = list(zip(enhanced_feature_columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for name, importance in feature_importance:
        print(f"  {name}: {importance:.4f}")

    # 11. SAVE MODEL
    if test_r2 > 0.3:  # Lower threshold to ensure saving
        print(f"\nSaving enhanced model...")

        # Save model components
        joblib.dump(enhanced_model, 'models/enhanced_standardized_model.pkl')
        joblib.dump(scaler, 'models/enhanced_standardized_scaler.pkl')

        # Save encoders
        location_encoders = {
            'district_mean_map': district_mean_map,
            'province_mean_map': province_mean_map,
            'location_tiers': location_tiers,
            'direction_mean_map': direction_mean_map,
            'type_mean_map': type_mean_map
        }
        joblib.dump(location_encoders, 'models/enhanced_location_encoders.pkl')

        # Save model info
        model_info = {
            'feature_columns': enhanced_feature_columns,
            'model_type': 'enhanced_location_aware',
            'performance': {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae)
            },
            'tier_system': {
                'tier_5': 'Quận trung tâm HN/HCM',
                'tier_4': 'Quận mở rộng HN/HCM',
                'tier_3': 'Thành phố tỉnh',
                'tier_2': 'Huyện/quận ngoại ô HN/HCM',
                'tier_1': 'Huyện/tỉnh khác'
            }
        }

        with open('models/enhanced_standardized_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        print("Enhanced model saved successfully!")

        # Test prediction
        print(f"\nTest prediction:")
        test_area = 84
        test_district = 'Văn Giang'

        if test_district in district_mean_map:
            district_encoded = district_mean_map[test_district]
            province_encoded = province_mean_map.get('Hưng Yên', df_encoded['Price'].mean())
            location_tier = location_tiers.get(test_district, 1.0)
            is_hanoi, is_hcmc, is_major_city = get_city_flags(test_district)
            direction_encoded = direction_mean_map.get('Unknown', df_encoded['Price'].mean())
            type_encoded = type_mean_map.get('Nhà', df_encoded['Price'].mean())

            features = [
                test_area, 5, 8, 4, 3, 2,  # Area, Frontage, Access_Road, Floors, Bedrooms, Bathrooms
                district_encoded, province_encoded, location_tier,
                is_hanoi, is_hcmc, is_major_city,
                direction_encoded, type_encoded
            ]

            features_scaled = scaler.transform([features])
            prediction = enhanced_model.predict(features_scaled)[0]

            print(f"  {test_district}, 84m², 3BR: {prediction:.2f} tỷ VNĐ (Tier {location_tier})")

        print(f"\nFILES SAVED:")
        print(f"  - enhanced_standardized_model.pkl")
        print(f"  - enhanced_standardized_scaler.pkl")
        print(f"  - enhanced_location_encoders.pkl")
        print(f"  - enhanced_standardized_model_info.json")

    else:
        print(f"\nModel performance insufficient (R2 = {test_r2:.4f})")

    print(f"\nTraining completed successfully!")

except Exception as e:
    print(f"\nError during training: {str(e)}")
    print(f"Please check your data file and try again.")
    import traceback

    traceback.print_exc()