import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def split_by_patient(states_with_action, bloctrain):
    """
    根据blocstrain的序号（1,2,3..）来分组病人数据
    当序号重新从1开始时，表示新的病人
    """
    patients_data = []
    current_patient_data = []

    if not isinstance(states_with_action, torch.Tensor):
        states_with_action = torch.tensor(states_with_action, dtype=torch.float)
    if not isinstance(bloctrain, torch.Tensor):
        bloctrain = torch.tensor(bloctrain, dtype=torch.float)

    for idx, step in enumerate(bloctrain):
        # 当前时间步是1，且不是第一个数据点时，说明是新病人的开始
        if step == 1 and idx > 0:
            if current_patient_data:
                patients_data.append(torch.stack(current_patient_data))
            current_patient_data = [states_with_action[idx]]
        else:
            current_patient_data.append(states_with_action[idx])

    # 添加最后一个病人的数据
    if current_patient_data:
        patients_data.append(torch.stack(current_patient_data))

    return patients_data


def prepare_batch_data(patients_data, actions, max_length):
    padded_sequences = []
    lengths = []
    action_sequences = []

    for i, patient_data in enumerate(patients_data):
        seq_len = patient_data.size(0)
        lengths.append(min(seq_len, max_length))

        if seq_len > max_length:
            patient_data = patient_data[:max_length]
            actions[i] = actions[i][:max_length]
        padded_sequences.append(patient_data)
        action_sequences.append(actions[i])

    padded_sequences = pad_sequence(padded_sequences, batch_first=True)
    action_sequences = pad_sequence(action_sequences, batch_first=True)

    return padded_sequences, lengths, action_sequences


def process_data(states_with_action, bloctrain, max_length):
    """
    处理状态和动作数据，确保维度匹配
    """
    # 分离状态和动作
    states = states_with_action[:, :-2]
    actions = states_with_action[:, -2:]

    states_tensor = (
        torch.tensor(states, dtype=torch.float)
        if not isinstance(states, torch.Tensor)
        else states
    )
    actions_tensor = (
        torch.tensor(actions, dtype=torch.float)
        if not isinstance(actions, torch.Tensor)
        else actions
    )
    bloctrain_tensor = (
        torch.tensor(bloctrain, dtype=torch.float)
        if not isinstance(bloctrain, torch.Tensor)
        else bloctrain
    )

    # 分离并处理状态和动作
    patients_states = split_by_patient(states_tensor, bloctrain_tensor)
    patients_actions = split_by_patient(actions_tensor, bloctrain_tensor)

    states_padded, lengths, action_nonflat = prepare_batch_data(
        patients_states, patients_actions, max_length=max_length
    )

    # 展平actions数据
    all_actions = []
    for i, patient_actions in enumerate(patients_actions):
        # 只取实际长度的action数据
        actual_length = lengths[i]
        all_actions.append(patient_actions[:actual_length])

    # 拼接所有病人的actions
    actions_flat = torch.cat(all_actions, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return states_padded, actions_flat, lengths_tensor, action_nonflat


# def prepare_xgboost_data(hidden_states, actions, test_size=0.2, random_state=42):
#     """
#     准备XGBoost的训练数据，并划分训练集和测试集
#     """
#     X = (
#         hidden_states.detach().cpu().numpy()
#         if torch.is_tensor(hidden_states)
#         else hidden_states
#     )
#     y = actions.detach().cpu().numpy() if torch.is_tensor(actions) else actions
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )

#     print(f"\nData shapes:")
#     print(f"X_train shape: {X_train.shape}")
#     print(f"X_test shape: {X_test.shape}")
#     print(f"y_train shape: {y_train.shape}")
#     print(f"y_test shape: {y_test.shape}")
#     print(f"Unique action values: {np.unique(y)}")

#     return X_train, X_test, y_train, y_test
def prepare_xgboost_data(hidden_states, actions, lengths=None, test_size=0.2, random_state=42):
    """
    准备XGBoost的训练数据，并划分训练集和测试集
    增加了lengths参数的处理
    """
    X = (
        hidden_states.detach().cpu().numpy()
        if torch.is_tensor(hidden_states)
        else hidden_states
    )
    y = actions.detach().cpu().numpy() if torch.is_tensor(actions) else actions
    
    indices = np.arange(len(lengths)) if lengths is not None else None
    
    # 如果提供了lengths，需要考虑样本的对应关系
    if lengths is not None:
        # 创建一个数组记录每个特征向量属于哪个原始样本
        sample_indices = []
        current_idx = 0
        for i, length in enumerate(lengths):
            sample_indices.extend([i] * length)
        sample_indices = np.array(sample_indices)
        
        # 确保sample_indices长度与特征数量相匹配
        assert len(sample_indices) == len(X)
        
        # 使用分层抽样确保每个原始样本的特征被分到同一个集合
        unique_indices = np.arange(len(lengths))
        train_indices, test_indices = train_test_split(
            unique_indices,
            test_size=test_size,
            random_state=random_state
        )
        
        # 根据原始样本的分割结果分配特征
        train_mask = np.isin(sample_indices, train_indices)
        test_mask = np.isin(sample_indices, test_indices)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        lengths_train = lengths[train_indices]
        lengths_test = lengths[test_indices]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        lengths_train = None
        lengths_test = None

    print(f"\nData shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Unique action values: {np.unique(y)}")
    
    if lengths is not None:
        print(f"lengths_train shape: {lengths_train.shape}")
        print(f"lengths_test shape: {lengths_test.shape}")

    return X_train, X_test, y_train, y_test, lengths_train, lengths_test
    
