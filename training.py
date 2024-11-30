import numpy as np
from utilis import is_confident_state
from model import calculate_class_weights
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def train_xgboost_classifier(xgb_model, X_train, y_train, X_test, y_test):
    """
    训练XGBoost分类器并评估
    """
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    sample_weights = calculate_class_weights(y_train)

    xgb_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True,
    )

    return xgb_model


def evaluate_model(xgb_model, X_train, X_test, y_train, y_test):
    train_probs = xgb_model.predict_proba(X_train)
    train_preds = xgb_model.predict(X_train)
    train_conf, train_entropies = is_confident_state(train_probs)
    

    test_probs = xgb_model.predict_proba(X_test)
    test_preds = xgb_model.predict(X_test)
    test_conf, test_entropies = is_confident_state(test_probs)

    print("\nModel Evaluation:")
    print("Training Set:")
    print(f"Confident states ratio: {np.mean(train_conf):.2%}")
    print(f"Average entropy: {np.mean(train_entropies):.4f}")

    print("\nTest Set:")
    print(f"Confident states ratio: {np.mean(test_conf):.2%}")
    print(f"Average entropy: {np.mean(test_entropies):.4f}")


    n_classes = len(np.unique(y_train))

    print("\nAUC Scores per class:")
    for i in range(n_classes):
        try:
            train_auc = roc_auc_score(y_train == i, train_probs[:, i])
            test_auc = roc_auc_score(y_test == i, test_probs[:, i])
            print(f"Class {i}:")
            print(f"  Train AUC: {train_auc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")
        except:
            print(f"Could not compute AUC for class {i}")
            
    n_classes = len(np.unique(y_train))
    
    print("\nAccuracy per class:")
    for i in range(n_classes):
        train_class_mask = (y_train == i)
        train_class_acc = accuracy_score(y_train[train_class_mask], 
                                       train_preds[train_class_mask])
        train_class_correct = np.sum((y_train == i) & (train_preds == i))
        train_class_total = np.sum(y_train == i)
        
        test_class_mask = (y_test == i)
        test_class_acc = accuracy_score(y_test[test_class_mask], 
                                      test_preds[test_class_mask])
        test_class_correct = np.sum((y_test == i) & (test_preds == i))
        test_class_total = np.sum(y_test == i)
        
        print(f"\nClass {i}:")
        print(f"  Train Accuracy: {train_class_acc:.2%} ({train_class_correct}/{train_class_total})")
        print(f"  Test Accuracy: {test_class_acc:.2%} ({test_class_correct}/{test_class_total})")

      


# def evaluate_random_short_sequences(
#     X_test, y_test, lengths_test, model, n_samples, seq_length=5
# ):
#     """
#     从测试集随机选取n_samples个病人的短序列进行评估

#     参数:
#     X_test: 测试集特征 [total_sequences, hidden_dim]
#     y_test: 测试集标签 [total_sequences]
#     lengths_test: 每个病人的序列长度
#     model: 训练好的XGBoost模型
#     n_samples: 要选取的样本数量
#     seq_length: 短序列长度
#     """
#     # 计算每个样本的起始索引
#     start_indices = np.zeros(len(lengths_test), dtype=int)
#     for i in range(1, len(lengths_test)):
#         start_indices[i] = start_indices[i-1] + lengths_test[i-1]
    
#     # 找到序列长度大于seq_length的病人索引
#     valid_indices = np.where(lengths_test >= seq_length)[0]
    
#     if len(valid_indices) < n_samples:
#         print(f"Warning: Only {len(valid_indices)} valid sequences available. Using all of them.")
#         n_samples = len(valid_indices)
    
#     selected_indices = np.random.choice(valid_indices, size=n_samples, replace=False)

#     results = []

#     print(f"\nEvaluating {n_samples} random sequences of length {seq_length}:")

#     for i, idx in enumerate(selected_indices):
#         # 获取当前样本的起始位置
#         start_idx = start_indices[idx]
#         # 获取序列片段
#         patient_features = X_test[start_idx:start_idx + seq_length]
#         patient_labels = y_test[start_idx:start_idx + seq_length]

#         pred_probs = model.predict_proba(patient_features)
#         predictions = model.predict(patient_features)

#         eps = 1e-15
#         entropies = -np.sum(pred_probs * np.log(np.clip(pred_probs, eps, 1.0)), axis=1)

#         reference_probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
#         entropy_threshold = -np.sum(reference_probs * np.log(reference_probs))
#         is_confident = entropies < entropy_threshold

#         print(f"\nSequence {i+1}:")
#         print("Time step | True Label | Predicted | Confidence | Entropy | Confident?")
#         print("-" * 65)

#         for t in range(seq_length):
#             max_prob = np.max(pred_probs[t])
#             print(
#                 f"{t+1:^9} | {patient_labels[t]:^10} | {predictions[t]:^9} | "
#                 f"{max_prob:^10.4f} | {entropies[t]:^7.4f} | {is_confident[t]}"
#             )

#         results.append(
#             {
#                 "sequence_id": i + 1,
#                 "features": patient_features,
#                 "true_labels": patient_labels,
#                 "predictions": predictions,
#                 "pred_probs": pred_probs,
#                 "entropies": entropies,
#                 "is_confident": is_confident,
#             }
#         )

#         print("\nProbability distributions:")
#         print("Time step |", end=" ")
#         for c in range(5):
#             print(f"Class {c:^7}|", end=" ")
#         print()
#         print("-" * 65)

#         for t in range(seq_length):
#             print(f"{t+1:^9} |", end=" ")
#             for c in range(5):
#                 print(f"{pred_probs[t][c]:^7.4f}|", end=" ")
#             print()

#         print("\n" + "=" * 65)

#     return results

def evaluate_random_short_sequences(
    X_test, y_test, lengths_test, model, n_samples, seq_length=5
):
    """
    从测试集随机选取n_samples个病人的短序列进行评估

    参数:
    X_test: 测试集特征 [total_sequences, hidden_dim]
    y_test: 测试集标签 [total_sequences]
    lengths_test: 每个病人的序列长度
    model: 训练好的XGBoost模型
    n_samples: 要选取的样本数量
    seq_length: 短序列长度
    """
    ## select patients with time steps >= 5

    start_indices = np.zeros(len(lengths_test), dtype=int)
    for i in range(1, len(lengths_test)):
        start_indices[i] = start_indices[i-1] + lengths_test[i-1]
    
    valid_indices = np.where(lengths_test >= seq_length)[0]
    
    if len(valid_indices) < n_samples:
        print(f"Warning: Only {len(valid_indices)} valid sequences available. Using all of them.")
        n_samples = len(valid_indices)
    
    selected_indices = np.random.choice(valid_indices, size=n_samples, replace=False)


    ## Get the AUC and accuracy for each class 
    results = []
    all_predictions = []
    all_true_labels = []

    print(f"\nEvaluating {n_samples} random sequences of length {seq_length}:")

    for i, idx in enumerate(selected_indices):
        start_idx = start_indices[idx]
  
        patient_features = X_test[start_idx:start_idx + seq_length]
        patient_labels = y_test[start_idx:start_idx + seq_length]

        pred_probs = model.predict_proba(patient_features)
        predictions = model.predict(patient_features)

        all_predictions.extend(predictions)
        all_true_labels.extend(patient_labels)

        eps = 1e-15
        entropies = -np.sum(pred_probs * np.log(np.clip(pred_probs, eps, 1.0)), axis=1)

        reference_probs = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        entropy_threshold = -np.sum(reference_probs * np.log(reference_probs))
        is_confident = entropies < entropy_threshold

        print(f"\nSequence {i+1}:")
        print("Time step | True Label | Predicted | Confidence | Entropy | Confident?")
        print("-" * 65)

        for t in range(seq_length):
            max_prob = np.max(pred_probs[t])
            print(
                f"{t+1:^9} | {patient_labels[t]:^10} | {predictions[t]:^9} | "
                f"{max_prob:^10.4f} | {entropies[t]:^7.4f} | {is_confident[t]}"
            )

        results.append(
            {
                "sequence_id": i + 1,
                "features": patient_features,
                "true_labels": patient_labels,
                "predictions": predictions,
                "pred_probs": pred_probs,
                "entropies": entropies,
                "is_confident": is_confident,
            }
        )

        print("\nProbability distributions:")
        print("Time step |", end=" ")
        for c in range(5):
            print(f"Class {c:^7}|", end=" ")
        print()
        print("-" * 65)

        for t in range(seq_length):
            print(f"{t+1:^9} |", end=" ")
            for c in range(5):
                print(f"{pred_probs[t][c]:^7.4f}|", end=" ")
            print()

        print("\n" + "=" * 65)

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    n_classes = len(np.unique(all_true_labels))
    print("\nPer-class Accuracy:")
    print("-" * 30)
    
    for i in range(n_classes):
        class_mask = (all_true_labels == i)
        class_acc = accuracy_score(all_true_labels[class_mask], 
                                 all_predictions[class_mask])
        class_correct = np.sum((all_true_labels == i) & (all_predictions == i))
        class_total = np.sum(all_true_labels == i)
        
        print(f"Class {i}: {class_acc:.2%} ({class_correct}/{class_total})")
    
    overall_acc = accuracy_score(all_true_labels, all_predictions)
    overall_correct = np.sum(all_true_labels == all_predictions)
    overall_total = len(all_true_labels)
    print("-" * 30)
    print(f"Overall: {overall_acc:.2%} ({overall_correct}/{overall_total})")


    return results

    




