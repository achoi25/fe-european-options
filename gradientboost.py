import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def loadOptionData(filePath):
    df = pd.read_csv(filePath)
    
    dateColumns = ['QUOTE_UNIXTIME', 'QUOTE_READTIME', 'QUOTE_DATE', 'EXPIRE_DATE']
    for col in dateColumns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    if 'EXPIRE_UNIX' in df.columns and 'QUOTE_UNIXTIME' in df.columns:
        df['TIME_TO_EXPIRY'] = (df['EXPIRE_UNIX'] - df['QUOTE_UNIXTIME'].astype(int)) / (24 * 3600)
    
    if 'UNDERLYING_LAST' in df.columns and 'STRIKE' in df.columns:
        df['MONEYNESS'] = df['UNDERLYING_LAST'] / df['STRIKE']
        df['LOG_MONEYNESS'] = np.log(df['MONEYNESS'])
    
    return df

def prepareFeatures(df, targetColumn='C_LAST'):
    potentialFeatures = [
        'UNDERLYING_LAST', 'STRIKE', 'DTE', 'TIME_TO_EXPIRY', 
        'MONEYNESS', 'LOG_MONEYNESS', 'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT',
        'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV',
        'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 'P_RHO', 'P_IV'
    ]
    
    availableFeatures = [col for col in potentialFeatures if col in df.columns and not df[col].isnull().all()]
    
    X = df[availableFeatures].copy()
    
    if targetColumn not in df.columns:
        print(f"Target column '{targetColumn}' not found. Available columns: {list(df.columns)}")
        if 'C_LAST' in df.columns:
            targetColumn = 'C_LAST'
        elif 'P_LAST' in df.columns:
            targetColumn = 'P_LAST'
        else:
            priceColumns = [col for col in df.columns if 'LAST' in col or 'PRICE' in col]
            if priceColumns:
                targetColumn = priceColumns[0]
                print(f"Using {targetColumn} as target variable")
            else:
                print("Creating synthetic target variable for demonstration...")
                df['SYNTHETIC_PRICE'] = df['UNDERLYING_LAST'] * 0.1 + np.random.normal(0, 10, len(df))
                targetColumn = 'SYNTHETIC_PRICE'
    
    y = df[targetColumn].copy()
    
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    print(f"Using {len(availableFeatures)} features: {availableFeatures}")
    print(f"Target variable: {targetColumn}")
    print(f"Data shape: X {X.shape}, y {y.shape}")
    
    return X, y, availableFeatures

def trainXGBoostModel(X, y, testSize=0.2, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, random_state=randomState
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=randomState,
        objective='reg:squarederror',
        early_stopping_rounds=10,
        eval_metric='rmse'
    )
    
    print("Training XGBoost model...")
    
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=10
    )
    
    return model, X_train, X_test, y_train, y_test, scaler

def evaluateModel(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return y_pred, rmse, r2

def plotPredictions(y_test, y_pred, title="Actual vs Predicted Option Prices"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plotFeatureImportance(model, featureNames):
    importanceScores = model.feature_importances_
    featureImportanceDf = pd.DataFrame({
        'feature': featureNames,
        'importance': importanceScores
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(featureImportanceDf['feature'], featureImportanceDf['importance'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return featureImportanceDf

def plotTrainingHistory(model):
    if hasattr(model, 'evals_result_'):
        results = model.evals_result_
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
        plt.legend()
        plt.ylabel('RMSE')
        plt.xlabel('Epochs')
        plt.title('XGBoost Training History')
        plt.show()

if __name__ == "__main__":
    filePath = 'clean_data/cleaned_spx_eod_202301.csv'
    
    try:
        df = loadOptionData(filePath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        X, y, featureNames = prepareFeatures(df, targetColumn='C_LAST')
        
        model, X_train, X_test, y_train, y_test, scaler = trainXGBoostModel(X, y)
        
        y_pred, rmse, r2 = evaluateModel(model, X_test, y_test, scaler)
        
        plotPredictions(y_test, y_pred, "Actual vs Predicted Option Prices")
        plotTrainingHistory(model)
        
        featureImportanceDf = plotFeatureImportance(model, featureNames)
        print("\nTop 5 Most Important Features:")
        print(featureImportanceDf.head())
        
        print("\nExample Predictions:")
        for i in range(5):
            sampleIdx = i
            sampleFeatures = X_test.iloc[sampleIdx:sampleIdx+1]
            sampleFeaturesScaled = scaler.transform(sampleFeatures)
            prediction = model.predict(sampleFeaturesScaled)[0]
            actual = y_test.iloc[sampleIdx]
            
            print(f"Sample {i+1}: Predicted=${prediction:.2f}, Actual=${actual:.2f}, "
                  f"Error=${abs(prediction - actual):.2f} "
                  f"({abs(prediction - actual)/actual*100:.1f}%)")
        
        resultsDf = X_test.copy()
        resultsDf['ActualPrice'] = y_test.values
        resultsDf['PredictedPrice'] = y_pred
        resultsDf['AbsoluteError'] = np.abs(resultsDf['ActualPrice'] - resultsDf['PredictedPrice'])
        resultsDf['PercentageError'] = (resultsDf['AbsoluteError'] / resultsDf['ActualPrice']) * 100
        
        print(f"\nOverall Performance Summary:")
        print(f"Average Absolute Error: ${resultsDf['AbsoluteError'].mean():.2f}")
        print(f"Average Percentage Error: {resultsDf['PercentageError'].mean():.2f}%")
        print(f"Median Percentage Error: {resultsDf['PercentageError'].median():.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()