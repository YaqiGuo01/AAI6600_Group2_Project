import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MentalHealthCareClassifier:
    """
    Classifies mental health conversations into appropriate care types with confidence scores.
    Supports direct text input for real-time prediction.
    """
    
    def __init__(self, model_type='logistic', embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'logistic' or 'random_forest'
            embedding_model_name: Name of SentenceTransformer model for text embedding
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None  # Lazy load embedding model to save memory
        self.target_dim = 1536  # 固定目标维度为1536（与训练数据匹配）
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                multi_class='multinomial'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            )
        
        self.classes_ = None
    
    def _load_embedding_model(self):
        """Lazy load text embedding model (loads only when first used)"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"📥 Loading embedding model: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model
    
    def text_to_embedding(self, text):
        """Convert raw text to embedding vector and adjust to target dimension (1536)"""
        model = self._load_embedding_model()
        embedding = model.encode(text)
        
        # 自动调整维度至1536（不够补0，多余截断）
        if len(embedding) < self.target_dim:
            embedding = np.pad(embedding, (0, self.target_dim - len(embedding)), 'constant')
        else:
            embedding = embedding[:self.target_dim]
        return embedding
    
    def load_data(self, train_path, test_path):
        """Load training and test data from CSV files."""
        print("📁 Loading data...")
        
        # Load training data
        train_df = pd.read_csv(train_path)
        print(f"   Training samples: {len(train_df)}")
        print(f"   Columns in training data: {train_df.columns.tolist()[:5]}...")
        
        # Load test data
        test_df = pd.read_csv(test_path)
        print(f"   Test samples: {len(test_df)}")
        
        # Auto-detect embedding columns
        X_train = None
        X_test = None
        
        # Method 1: Check if there's a single 'embedding' column with array/list format
        if 'embedding' in train_df.columns:
            print("   ✓ Detected 'embedding' column")
            try:
                import ast
                # Parse string representation to array
                train_df['embedding_parsed'] = train_df['embedding'].apply(
                    lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
                )
                X_train = np.vstack(train_df['embedding_parsed'].values)
                
                # Same for test data
                if 'embedding' in test_df.columns:
                    test_df['embedding_parsed'] = test_df['embedding'].apply(
                        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
                    )
                    X_test = np.vstack(test_df['embedding_parsed'].values)
                print(f"   ✓ Successfully parsed embedding column")
            except Exception as e:
                print(f"   ✗ Could not parse 'embedding' column: {e}")
                X_train = None
        
        # Method 2: Look for columns with common embedding patterns
        if X_train is None:
            patterns = ['dim_', 'embedding_', 'emb_', 'feature_', 'e_', 'd_']
            embedding_cols = []
            
            for pattern in patterns:
                embedding_cols = [col for col in train_df.columns if col.startswith(pattern)]
                if embedding_cols:
                    print(f"   ✓ Found {len(embedding_cols)} columns with pattern '{pattern}'")
                    break
            
            if embedding_cols:
                X_train = train_df[embedding_cols].values
                X_test = test_df[embedding_cols].values
        
        # Method 3: Use all numeric columns except known label columns
        if X_train is None:
            print("   Trying to auto-detect numeric feature columns...")
            # Exclude common label/text column names
            exclude_cols = ['text', 'assistance_type', 'label', 'class', 'category', 
                          'type', 'id', 'index', 'Unnamed: 0']
            
            # Find numeric columns
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            embedding_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if embedding_cols:
                print(f"   ✓ Using {len(embedding_cols)} numeric columns as features")
                print(f"   Sample columns: {embedding_cols[:5]}")
                X_train = train_df[embedding_cols].values
                X_test = test_df[embedding_cols].values
        
        # If still no embeddings found, raise error
        if X_train is None:
            print("\n❌ ERROR: Could not detect embedding columns!")
            print(f"Available columns: {train_df.columns.tolist()}")
            raise ValueError(
                "Could not find embedding columns. Please ensure your CSV has either:\n"
                "  1. A column named 'embedding' with array values, OR\n"
                "  2. Multiple columns starting with 'dim_', 'embedding_', etc., OR\n"
                "  3. Numeric feature columns"
            )
        
        # Extract labels and texts
        # Check for different possible label column names
        label_col = None
        possible_labels = ['assistance_type', 'healthcare_label', 'label', 'class', 'category', 'type']
        for col in possible_labels:
            if col in train_df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError(f"Training data must have a label column. Expected one of: {possible_labels}")
        
        print(f"   Using '{label_col}' as label column")
        y_train = train_df[label_col].values
        test_texts = test_df['text'].values if 'text' in test_df.columns else [f"Sample {i+1}" for i in range(len(test_df))]
        
        print(f"   Feature dimensions: {X_train.shape[1]}")
        print(f"   Number of classes: {len(np.unique(y_train))}")
        print(f"   Sample classes: {list(np.unique(y_train))[:3]}...")
        
        return X_train, y_train, X_test, test_texts
    
    def train(self, X_train, y_train, validate=True):
        """
        Train the classifier with optional cross-validation.
        """
        print(f"\n🔧 Training {self.model_type} classifier...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Cross-validation
        if validate:
            print("   Performing 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='accuracy'
            )
            print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X_train_scaled, y_train)
        self.classes_ = self.model.classes_
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"   Training Accuracy: {train_acc:.4f}")
        
        return self
    
    def evaluate(self, X_train, y_train, split_ratio=0.2):
        """
        Evaluate model performance on a held-out validation set.
        """
        print(f"\n📊 Model Evaluation (Hold-out validation: {split_ratio*100:.0f}%)")
        print("=" * 70)
        
        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=split_ratio, random_state=42, stratify=y_train
        )
        
        # Scale features
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train on training split
        self.model.fit(X_tr_scaled, y_tr)
        self.classes_ = self.model.classes_
        
        # Predict on validation split
        y_pred = self.model.predict(X_val_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print("-" * 70)
        report = classification_report(y_val, y_pred, zero_division=0)
        print(report)
        
        # Confusion matrix summary
        cm = confusion_matrix(y_val, y_pred)
        print(f"\n🔢 Confusion Matrix Shape: {cm.shape}")
        print(f"   Total predictions: {cm.sum()}")
        print(f"   Correct predictions: {np.trace(cm)}")
        
        return accuracy
    
    def predict_with_confidence(self, X_test, text=None, top_k=3):
        """
        Predict care type with confidence scores.
        """
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        predictions = self.model.predict(X_test_scaled)
        
        # Get probability scores for all classes
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_test_scaled)
        else:
            # For models without predict_proba, use decision function
            decision_scores = self.model.decision_function(X_test_scaled)
            # Convert to pseudo-probabilities using softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        results = []
        for i in range(len(X_test)):
            # Get top k predictions
            top_indices = np.argsort(probabilities[i])[::-1][:top_k]
            top_classes = self.classes_[top_indices]
            top_probs = probabilities[i][top_indices]
            
            # Create top recommendations list
            top_recommendations = [
                {
                    'assistance_type': cls,
                    'confidence': prob
                }
                for cls, prob in zip(top_classes, top_probs)
            ]
            
            result = {
                'text': text[i] if text is not None else f"Sample {i+1}",
                'predicted_assistance_type': predictions[i],
                'confidence': probabilities[i][np.where(self.classes_ == predictions[i])[0][0]],
                'top_recommendations': top_recommendations,
                'all_probabilities': dict(zip(self.classes_, probabilities[i]))
            }
            results.append(result)
        
        return results
    
    def predict_text(self, text, top_k=3):
        """
        Predict care type directly from raw text.
        """
        # 1. Convert text to embedding vector (自动调整为1536维)
        embedding = self.text_to_embedding(text)
        # 2. Reshape to 2D array (required for model input)
        X_test = embedding.reshape(1, -1)
        # 3. Get prediction results
        results = self.predict_with_confidence(X_test, text=[text], top_k=top_k)
        return results[0]  # Return single result
    
    def display_predictions(self, results):
        """Display prediction results in a formatted way."""
        print("\n" + "="*70)
        print("🔮 PREDICTION RESULTS")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─'*70}")
            print(f"Sample #{i}: {result['text'][:80]}...")
            print(f"{'─'*70}")
            print(f"🎯 Recommended: {result['predicted_assistance_type']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"📋 Top 3 recommendations:")
            
            for j, rec in enumerate(result['top_recommendations'], 1):
                confidence_bar = "█" * int(rec['confidence'] * 20)
                print(f"   {j}. {rec['assistance_type']:<40} {rec['confidence']:>6.2%} {confidence_bar}")
            
            # Confidence interpretation
            conf = result['confidence']
            if conf >= 0.8:
                status = "🟢 High confidence"
            elif conf >= 0.5:
                status = "🟡 Medium confidence"
            else:
                status = "🔴 Low confidence - consider manual review"
            print(f"\n   {status}")


def main():
    """Main execution function with interactive text input support."""
    
    # File paths - 仅修改此处的TRAIN_FILE为新训练集路径
    TRAIN_FILE = 'training_data_embedding_1000.csv'  # 新训练集：training_data_embedding_1000.csv
    TEST_FILE = 'test_data_embedding.csv'  # 测试集路径保持不变
    
    print("🏥 Mental Health Care Classification System")
    print("="*70)
    
    # Initialize classifier
    classifier = MentalHealthCareClassifier(model_type='logistic')
    
    try:
        # Load data
        X_train, y_train, X_test, test_texts = classifier.load_data(TRAIN_FILE, TEST_FILE)
        
        # Evaluate model performance
        accuracy = classifier.evaluate(X_train, y_train, split_ratio=0.2)
        
        # Train on full dataset
        classifier.train(X_train, y_train, validate=True)
        
        # Make predictions on test data
        print(f"\n🔮 Making predictions on {len(X_test)} test samples...")
        results = classifier.predict_with_confidence(X_test, text=test_texts, top_k=3)
        
        # Display results
        classifier.display_predictions(results)
        
        # Summary statistics
        print("\n" + "="*70)
        print("📈 SUMMARY STATISTICS")
        print("="*70)
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"Average confidence: {avg_confidence:.2%}")
        
        high_conf = sum(1 for r in results if r['confidence'] >= 0.8)
        med_conf = sum(1 for r in results if 0.5 <= r['confidence'] < 0.8)
        low_conf = sum(1 for r in results if r['confidence'] < 0.5)
        
        print(f"High confidence predictions (≥80%): {high_conf}/{len(results)}")
        print(f"Medium confidence predictions (50-80%): {med_conf}/{len(results)}")
        print(f"Low confidence predictions (<50%): {low_conf}/{len(results)}")
        
    except Exception as e:
        print(f"\n⚠️ Warning: Error processing test data - {str(e)}")
        print("Proceeding directly to interactive text classification mode (model may be untrained)")
    
    # Interactive text input demo
    print("\n" + "="*70)
    print("💬 Interactive Text Classification Demo")
    print("="*70)
    print("Hint: Enter any text (e.g. 'I've been stressed lately and can't sleep'), it will be classified automatically; enter 'q' to quit")
    
    while True:
        try:
            user_input = input("\nEnter text: ")
            if user_input.strip().lower() in ['q', 'quit', 'exit']:
                print("👋 Exiting demo")
                break
            
            # Get prediction for user input
            result = classifier.predict_text(user_input)
            
            # Display result
            print("\n" + "─"*70)
            print(f"Input text: {user_input}")
            print(f"{'─'*70}")
            print(f"🎯 Recommended Category: {result['predicted_assistance_type']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print("\n📋 Top 3 Recommendations:")
            for i, rec in enumerate(result['top_recommendations'], 1):
                print(f"   {i}. {rec['assistance_type']} (Confidence: {rec['confidence']:.2%})")
            
            # Confidence status
            if result['confidence'] >= 0.8:
                print("\n🟢 High confidence: Classification result is highly reliable")
            elif result['confidence'] >= 0.5:
                print("\n🟡 Medium confidence: Result is reference-worthy")
            else:
                print("\n🔴 Low confidence: Manual review is recommended")
            print("─"*70)
        
        except Exception as e:
            print(f"❌ Error processing input: {str(e)}, please try again")
    
    return classifier


if __name__ == "__main__":
    classifier = main()