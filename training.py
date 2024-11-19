import numpy as np
from utilis import is_confident_state
from model import calculate_class_weights


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
    train_conf, _, train_entropies = is_confident_state(train_probs, X_train)

    test_probs = xgb_model.predict_proba(X_test)
    test_conf, _, test_entropies = is_confident_state(test_probs, X_test)

    print("\nModel Evaluation:")
    print("Training Set:")
    print(f"Confident states ratio: {np.mean(train_conf):.2%}")
    print(f"Average entropy: {np.mean(train_entropies):.4f}")

    print("\nTest Set:")
    print(f"Confident states ratio: {np.mean(test_conf):.2%}")
    print(f"Average entropy: {np.mean(test_entropies):.4f}")

    # 计算每个类别的AUC
    from sklearn.metrics import roc_auc_score

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


def evaluate_random_short_sequences(
    X_test, y_test, lengths_test, model, n_samples, seq_length=5
):
    """
    从测试集随机选取n_samples个病人的短序列进行评估

    参数:
    X_test: 测试集特征 [n_patients, max_length, hidden_dim]
    y_test: 测试集标签 [n_patients, max_length]
    lengths_test: 每个病人的序列长度
    model: 训练好的XGBoost模型
    n_samples: 要选取的样本数量
    seq_length: 短序列长度
    """
    # 找到序列长度大于seq_length的病人索引
    valid_indices = np.where(lengths_test >= seq_length)[0]

    selected_indices = np.random.choice(valid_indices, size=n_samples, replace=False)

    results = []

    print(f"\nEvaluating {n_samples} random sequences of length {seq_length}:")

    for i, idx in enumerate(selected_indices):
        patient_features = X_test[idx, :seq_length]
        patient_labels = y_test[idx, :seq_length]

        pred_probs = model.predict_proba(patient_features)
        predictions = model.predict(patient_features)

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

    return results
