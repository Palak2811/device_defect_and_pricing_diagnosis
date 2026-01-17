
from price_predictor import train_model, test_predictions
import sys
import os

def main():
   
    dataset_path = 'data/pricing_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"\n Error: Dataset not found at {dataset_path}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--yes':
        proceed = True
    else:
        response = input("\nProceed with retraining? (yes/no): ").strip().lower()
        proceed = response in ['yes', 'y']
    
    if not proceed:
        print("\n Retraining cancelled.")
        return
    predictor, metrics = train_model()
    
    if predictor is None:
        print("\n Training failed.")
        return
    run_tests = input("Run test predictions? (yes/no): ").strip().lower()
    if run_tests in ['yes', 'y']:
        test_predictions()
    
if __name__ == "__main__":
    main()
