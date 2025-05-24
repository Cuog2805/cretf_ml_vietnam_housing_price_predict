# #improved_model_setup.py
# # Script để tạo model files với khắc phục overfitting
#
# import json
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import joblib
# import os
# import warnings
#
# warnings.filterwarnings('ignore')
#
# print("🏠 Improved Model Setup - Khắc phục overfitting...")
#
# # Tạo thư mục models
# os.makedirs("models", exist_ok=True)
#
# try:
#     # Đọc dữ liệu
#     print("📊 Đọc dữ liệu từ batdongsan_data.json...")
#     with open('data/batdongsan_data.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     df = pd.DataFrame(data)
#     print(f"Loaded {len(df)} records")
#
#     # 1. XỬ LÝ DỮ LIỆU VỚI DUPLICATE REMOVAL
#     print("🔍 Xóa dữ liệu trùng lặp...")
#
#     # Xóa duplicate hoàn toàn
#     initial_count = len(df)
#     df = df.drop_duplicates()
#
#     # Xóa duplicate theo các cột quan trọng
#     important_cols = ['title', 'price', 'area', 'district', 'province']
#     available_cols = [col for col in important_cols if col in df.columns]
#     df = df.drop_duplicates(subset=available_cols, keep='first')
#
#     # Xóa duplicate theo URL nếu có
#     if 'url' in df.columns:
#         df = df.drop_duplicates(subset=['url'], keep='first')
#
#     removed_count = initial_count - len(df)
#     print(f"✅ Đã xóa {removed_count} bản ghi trùng lặp ({removed_count / initial_count * 100:.1f}%)")
#     print(f"Còn lại: {len(df)} bản ghi unique")
#
#     # 2. FEATURE ENGINEERING NÂNG CAO
#     print("🔧 Feature Engineering nâng cao...")
#
#
#     # Chuyển đổi bedrooms và bathrooms về numeric
#     def convert_to_numeric(value):
#         if pd.isna(value) or value is None:
#             return np.nan
#         try:
#             return float(value)
#         except:
#             return np.nan
#
#
#     df['bedrooms'] = df['bedrooms'].apply(convert_to_numeric)
#     df['bathrooms'] = df['bathrooms'].apply(convert_to_numeric)
#
#     # Tạo features mới
#     df['has_bedroom_info'] = df['bedrooms'].notna().astype(int)
#     df['has_bathroom_info'] = df['bathrooms'].notna().astype(int)
#     df['total_rooms'] = df['bedrooms'].fillna(0) + df['bathrooms'].fillna(0)
#
#     # Features interaction mới
#     df['price_per_sqm'] = df['price'] / df['area']  # Có thể dùng để validate
#     df['room_density'] = df['total_rooms'] / df['area']  # Mật độ phòng
#
#     # Feature encoding cho district phổ biến
#     district_counts = df['district'].value_counts()
#     top_districts = district_counts.head(10).index.tolist()
#     df['is_popular_district'] = df['district'].isin(top_districts).astype(int)
#
#     # 3. XỬ LÝ OUTLIERS TỐTỐT HƠN
#     print("🎯 Xử lý outliers bằng IQR method...")
#
#     # Sử dụng IQR để loại bỏ outliers
#     for col in ['price', 'area']:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#
#         before_count = len(df)
#         df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
#         after_count = len(df)
#
#         print(f"Removed {before_count - after_count} outliers from {col}")
#
#     print(f"Data sau khi xử lý outliers: {len(df)} records")
#
#     # 4. CHUẨN BỊ DỮ LIỆU
#     features = ['area', 'bedrooms', 'bathrooms', 'district', 'province',
#                 'has_bedroom_info', 'has_bathroom_info', 'total_rooms',
#                 'room_density', 'is_popular_district']
#
#     df_model = df[features + ['price']].copy()
#     df_model = df_model.dropna(subset=['price'])
#
#     print(f"Data for modeling: {len(df_model)} records")
#
#     # Xử lý missing values
#     numeric_features = ['area', 'bedrooms', 'bathrooms', 'total_rooms', 'room_density']
#     imputer = SimpleImputer(strategy='median')
#     df_model[numeric_features] = imputer.fit_transform(df_model[numeric_features])
#
#     # Xử lý categorical features
#     categorical_features = ['district', 'province']
#     for col in categorical_features:
#         df_model[col] = df_model[col].fillna('Unknown')
#
#     # Encode categorical variables
#     label_encoders = {}
#     for col in categorical_features:
#         le = LabelEncoder()
#         df_model[col + '_encoded'] = le.fit_transform(df_model[col])
#         label_encoders[col] = le
#
#     # Prepare final features
#     final_features = [col for col in numeric_features] + \
#                      [col + '_encoded' for col in categorical_features] + \
#                      ['has_bedroom_info', 'has_bathroom_info', 'is_popular_district']
#
#     X = df_model[final_features]
#     y = df_model['price']
#
#     print(f"Features: {final_features}")
#     print(f"X shape: {X.shape}")
#
#     # 5. CHIA DỮ LIỆU VÀ TRAIN MODEL
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train model với hyperparameters chống overfitting
#     print("🌲 Training Random Forest với regularization...")
#     model = RandomForestRegressor(
#         n_estimators=100,
#         max_depth=10,  # Giảm từ 15 xuống 10
#         min_samples_split=10,  # Tăng từ 5 lên 10
#         min_samples_leaf=5,  # Tăng từ 2 lên 5
#         max_features='sqrt',  # Thêm để giảm overfitting
#         random_state=42,
#         n_jobs=-1
#     )
#
#     model.fit(X_train, y_train)
#
#     # 6. ĐÁNH GIÁ MODEL TOÀN DIỆN
#     print("\n📊 Đánh giá hiệu suất model...")
#
#     # Predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#
#     # Scores
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#
#     # Error metrics
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#     train_mae = mean_absolute_error(y_train, y_train_pred)
#     test_mae = mean_absolute_error(y_test, y_test_pred)
#
#     # Cross validation
#     cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
#
#     print(f"\n🎯 KẾT QUẢ ĐÁNH GIÁ:")
#     print(f"Train R² Score: {train_score:.4f}")
#     print(f"Test R² Score: {test_score:.4f}")
#     print(f"Overfitting gap: {abs(train_score - test_score):.4f}")
#     print(f"\nTrain RMSE: {train_rmse:.2f} tỷ VNĐ")
#     print(f"Test RMSE: {test_rmse:.2f} tỷ VNĐ")
#     print(f"Train MAE: {train_mae:.2f} tỷ VNĐ")
#     print(f"Test MAE: {test_mae:.2f} tỷ VNĐ")
#     print(f"\nCV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
#
#     # Feature importance
#     print(f"\n🔝 TOP 5 FEATURES:")
#     feature_importance = pd.DataFrame({
#         'feature': final_features,
#         'importance': model.feature_importances_
#     }).sort_values('importance', ascending=False)
#
#     for i, row in feature_importance.head().iterrows():
#         print(f"{row['feature']}: {row['importance']:.4f}")
#
#     # 7. LƯU MODEL
#     print("\n💾 Saving improved model files...")
#
#     joblib.dump(model, 'models/real_estate_model_improved.pkl')
#     print("✅ Saved: models/real_estate_model_improved.pkl")
#
#     joblib.dump(label_encoders, 'models/label_encoders_improved.pkl')
#     print("✅ Saved: models/label_encoders_improved.pkl")
#
#     joblib.dump(imputer, 'models/imputer_improved.pkl')
#     print("✅ Saved: models/imputer_improved.pkl")
#
#     # Lưu feature names
#     with open('models/feature_names.json', 'w') as f:
#         json.dump(final_features, f)
#     print("✅ Saved: models/feature_names.json")
#
#     print("\n🎉 Improved model setup completed!")
#
#     # 8. KIỂM TRA SAMPLE PREDICTION
#     print("\n🧪 Testing improved prediction...")
#     sample_data = pd.DataFrame({
#         'area': [100.0],
#         'bedrooms': [3.0],
#         'bathrooms': [2.0],
#         'total_rooms': [5.0],
#         'room_density': [0.05],
#         'district_encoded': [0],
#         'province_encoded': [0],
#         'has_bedroom_info': [1],
#         'has_bathroom_info': [1],
#         'is_popular_district': [1]
#     })
#
#     predicted_price = model.predict(sample_data[final_features])[0]
#     print(f"Sample prediction (100m², 3BR, 2BA): {predicted_price:.2f} tỷ VNĐ")
#
#     # Đánh giá cải thiện
#     overfitting_gap = abs(train_score - test_score)
#     if overfitting_gap < 0.15:
#         print(f"\n✅ THÀNH CÔNG: Overfitting gap giảm xuống {overfitting_gap:.4f} (<15%)")
#     else:
#         print(f"\n⚠️ CHÚ Ý: Overfitting gap vẫn cao {overfitting_gap:.4f} (>15%)")
#
# except FileNotFoundError as e:
#     print(f"❌ Error: File not found - {e}")
#     print("Please check your data file path.")
# except Exception as e:
#     print(f"❌ Error: {e}")
#     import traceback
#
#     traceback.print_exc()
#
# print("\n🔚 Script completed. Safe to close.")  # Thêm dòng này để tránh exception cuối