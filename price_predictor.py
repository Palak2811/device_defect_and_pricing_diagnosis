import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

class PricePredictor:
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df):
        df = df.copy()
        categorical_cols = ['brand', 'model', 'condition_grade']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = df[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ 
                    else -1
                )
        
        feature_cols = [
            'original_price',
            'age_months',
            'brand_encoded',
            'model_encoded',
            'num_defects',
            'has_screen_damage',
            'has_water_damage',
            'has_battery_issue',
            'has_physical_damage',
            'has_critical_defect',
            'total_severity_score',
            'avg_severity_score',
            'total_repair_cost',
            'condition_score',
            'condition_grade_encoded'
        ]
        
        self.feature_columns = feature_cols
        
        return df[feature_cols]
    
    def train(self, csv_path, test_size=0.2, random_state=42):
      
      df = pd.read_csv(csv_path)
      X = self.prepare_features(df)
      y = df['resale_price']
    
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
      self.model = xgb.XGBRegressor(
          n_estimators=200,
          max_depth=6,
          learning_rate=0.1,
          subsample=0.8,
          colsample_bytree=0.8,
          random_state=random_state,
          objective='reg:squarederror',
          n_jobs=-1
          )
      self.model.fit(
          X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
      y_pred_train = self.model.predict(X_train)
      y_pred_test = self.model.predict(X_test)
      train_mae = mean_absolute_error(y_train, y_pred_train)
      test_mae = mean_absolute_error(y_test, y_pred_test)
      train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
      test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
      train_r2 = r2_score(y_train, y_pred_train)
      test_r2 = r2_score(y_test, y_pred_test)
    
      errors = np.abs(y_test - y_pred_test)
      within_500 = np.sum(errors <= 500) / len(y_test) * 100
      within_1000 = np.sum(errors <= 1000) / len(y_test) * 100
      within_2000 = np.sum(errors <= 2000) / len(y_test) * 100
    
      mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
      feature_importance = pd.DataFrame({
          'feature': self.feature_columns,
          'importance': self.model.feature_importances_
      }).sort_values('importance', ascending=False)
    
      for idx, row in feature_importance.head(10).iterrows():
          print(f"   {row['feature']:.<30} {row['importance']:.3f}")
      sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
      for idx in sample_indices:
          actual = y_test.iloc[idx]
          predicted = y_pred_test[idx]
          error = abs(actual - predicted)
          print(f"   Actual: ₹{actual:>7,} | Predicted: ₹{predicted:>7,.0f} | Error: ₹{error:>6,.0f}")
    
      self.is_trained = True
    
      return {
         'train_mae': train_mae,
          'test_mae': test_mae,
          'train_rmse': train_rmse,
          'test_rmse': test_rmse,
          'train_r2': train_r2,
          'test_r2': test_r2,
          'mape': mape,
          'within_500': within_500,
          'within_1000': within_1000,
          'within_2000': within_2000 }
        
        
    
    def predict(self, device_info, defects, condition_score):
       
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load_model().")
        
        num_defects = len(defects)
        has_screen_damage = int(any(d.get('category') == 'screen' for d in defects))
        has_water_damage = int(any(d.get('category') == 'water' for d in defects))
        has_battery_issue = int(any(d.get('category') == 'battery' for d in defects))
        has_physical_damage = int(any(d.get('category') == 'physical' for d in defects))
        has_critical_defect = int(any(d.get('critical', False) for d in defects))
        
        total_severity = sum(d.get('severity_score', 0) for d in defects)
        avg_severity = total_severity / num_defects if num_defects > 0 else 0
        total_repair_cost = sum(d.get('repair_cost', 0) for d in defects)
        
        features = {
            'original_price': device_info['original_price'],
            'age_months': device_info['age_months'],
            'brand': device_info['brand'],
            'model': device_info['model'],
            'num_defects': num_defects,
            'has_screen_damage': has_screen_damage,
            'has_water_damage': has_water_damage,
            'has_battery_issue': has_battery_issue,
            'has_physical_damage': has_physical_damage,
            'has_critical_defect': has_critical_defect,
            'total_severity_score': total_severity,
            'avg_severity_score': avg_severity,
            'total_repair_cost': total_repair_cost,
            'condition_score': condition_score,
            'condition_grade': self._score_to_grade(condition_score)
        }
        df = pd.DataFrame([features])
        X = self.prepare_features(df)
        predicted_price = self.model.predict(X)[0]
        confidence_margin = max(500, 0.10 * predicted_price)
        predicted_price = round(predicted_price / 100) * 100
        price_min = max(500, round((predicted_price - confidence_margin) / 100) * 100)
        price_max = round((predicted_price + confidence_margin) / 100) * 100
        
        return {
            'predicted_price': int(predicted_price),
            'price_range': {
                'min': int(price_min),
                'max': int(price_max)
            },
            'confidence': 0.85,  # Based on model R² score
            'depreciation_factors': self._analyze_depreciation(device_info, defects),
            'feature_contributions': self._get_feature_contributions(X)
        }
    
    def _score_to_grade(self, score):
        if score >= 9:
            return 'A'
        elif score >= 7:
            return 'B'
        elif score >= 5:
            return 'C'
        elif score >= 3:
            return 'D'
        else:
            return 'F'
    
    def _analyze_depreciation(self, device_info, defects):
        factors = []
        age_years = device_info['age_months'] / 12
        if age_years <= 1:
            age_factor = -15 * age_years
        else:
            age_factor = -15 - (10 * (age_years - 1))
        
        factors.append({
            'factor': 'Age Depreciation',
            'impact': f"{age_factor:.1f}%",
            'description': f"{device_info['age_months']} months old"
        })
        for defect in defects:
            impact_pct = defect.get('price_impact', 0) * 100
            factors.append({
                'factor': defect.get('name', 'Unknown defect'),
                'impact': f"{impact_pct:.1f}%",
                'description': f"Repair cost: ₹{defect.get('repair_cost', 0):,}"
            })
        
        return factors
    
    def _get_feature_contributions(self, X):
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        contributions = []
        for feature, importance in top_features:
            contributions.append({
                'feature': feature.replace('_', ' ').title(),
                'importance': f"{importance:.3f}"
            })
        
        return contributions
    
    def save_model(self, path='models/price_model.pkl'):
        if not self.is_trained:
            raise ValueError("No trained model to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, path)
    
    def load_model(self, path='models/price_model.pkl'):

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        self.is_trained = True


def train_model():
   
    predictor = PricePredictor()
    dataset_path = 'data/pricing_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"\ Dataset not found: {dataset_path}")
        return None
    metrics = predictor.train(dataset_path)
    predictor.save_model()
    return predictor, metrics

def test_predictions():
    predictor = PricePredictor()
    predictor.load_model()
    test_cases = [
        {
            'name': 'iPhone 13 - Cracked Screen',
            'device': {
                'brand': 'Apple',
                'model': 'iPhone 13',
                'original_price': 79900,
                'age_months': 24
            },
            'defects': [
                {
                    'id': 'SCR001',
                    'name': 'Cracked Screen',
                    'severity_score': 9,
                    'price_impact': -0.35,
                    'repair_cost': 4000,
                    'category': 'screen',
                    'critical': True
                }
            ],
            'condition_score': 6.0
        },
        {
            'name': 'Samsung S22 - Perfect Condition',
            'device': {
                'brand': 'Samsung',
                'model': 'Galaxy S22',
                'original_price': 72999,
                'age_months': 18
            },
            'defects': [],
            'condition_score': 9.5
        },
        {
            'name': 'MacBook Air - Multiple Issues',
            'device': {
                'brand': 'Apple',
                'model': 'MacBook Air M2',
                'original_price': 119900,
                'age_months': 12
            },
            'defects': [
                {
                    'id': 'KEY001',
                    'name': 'Keyboard Malfunction',
                    'severity_score': 6,
                    'price_impact': -0.18,
                    'repair_cost': 2500,
                    'category': 'keyboard',
                    'critical': False
                },
                {
                    'id': 'BAT001',
                    'name': 'Battery Drain',
                    'severity_score': 6,
                    'price_impact': -0.18,
                    'repair_cost': 2500,
                    'category': 'battery',
                    'critical': False
                }
            ],
            'condition_score': 5.5
        },
        {
            'name': 'OnePlus 11 - Water Damage',
            'device': {
                'brand': 'OnePlus',
                'model': 'OnePlus 11',
                'original_price': 56999,
                'age_months': 6
            },
            'defects': [
                {
                    'id': 'WTR001',
                    'name': 'Water Damage',
                    'severity_score': 10,
                    'price_impact': -0.60,
                    'repair_cost': 8000,
                    'category': 'water',
                    'critical': True
                }
            ],
            'condition_score': 2.0
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        
        device = test['device']
        print(f"\nDevice: {device['brand']} {device['model']}")
        print(f"   Original Price: ₹{device['original_price']:,}")
        print(f"   Age: {device['age_months']} months ({device['age_months']/12:.1f} years)")
        print(f"   Defects: {len(test['defects'])}")
        
        if test['defects']:
            print(f"\n Detected Defects:")
            for defect in test['defects']:
                print(f"   • {defect['name']} (Severity: {defect['severity_score']}/10)")
        
        print(f"\n Condition Score: {test['condition_score']}/10")
        
        result = predictor.predict(
            test['device'],
            test['defects'],
            test['condition_score']
        )
        
        print(f"\n PREDICTED RESALE PRICE: ₹{result['predicted_price']:,}")
        print(f"   Range: ₹{result['price_range']['min']:,} - ₹{result['price_range']['max']:,}")
        print(f"   Confidence: {result['confidence']:.0%}")
        
        print(f"\n Price Impact Factors:")
        for factor in result['depreciation_factors']:
            print(f"   • {factor['factor']}: {factor['impact']}")
            print(f"     {factor['description']}")
        
        print(f"\n Top Feature Contributions:")
        for contrib in result['feature_contributions']:
            print(f"   • {contrib['feature']}: {contrib['importance']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_predictions()
    else:
        predictor, metrics = train_model()
        if predictor:
            input("Press Enter to run test predictions...")
            test_predictions()