import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def save_encoded_vectors(reduced_data, original_ids=None, output_dir="pca_output"):
    """
    保存所有数据的降维向量，将向量作为float数组存储在单个列中
    
    参数:
    reduced_data: PCA降维后的数据
    original_ids: 与输入数据对应的ID或索引(可选)
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 将降维向量转换为list of lists
    vector_list = reduced_data.tolist()
    
    # 创建DataFrame
    df = pd.DataFrame({'pca_vector': vector_list})
    
    # 如果有ID，添加到DataFrame
    if original_ids is not None:
        df.insert(0, 'id', original_ids)
    
    # 保存为numpy数组
    np_path = os.path.join(output_dir, "pca_vectors.npy")
    np.save(np_path, reduced_data)
    print(f"降维向量已保存为numpy数组: {np_path}")
    
    # 保存为CSV格式
    csv_path = os.path.join(output_dir, "pca_vectors.csv")
    df.to_csv(csv_path, index=False)
    print(f"降维向量已保存为CSV: {csv_path}")
    
    # 保存为parquet格式
    parquet_path = os.path.join(output_dir, "pca_vectors.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    print(f"降维向量已保存为parquet: {parquet_path}")
    
    print(f"降维向量形状: {reduced_data.shape}")

# 计算原始向量和降维向量之间的平均余弦相似度
def calculate_avg_cosine_similarity(original_data, reduced_data, n_samples=1000):
    """
    计算原始向量和降维向量之间的平均余弦相似度
    
    参数:
    original_data: 原始向量
    reduced_data: 降维后的向量
    n_samples: 采样数量，用于大数据集
    
    返回:
    平均余弦相似度
    """
    # 对于大数据集，随机抽样计算相似度
    if len(original_data) > n_samples:
        indices = np.random.choice(len(original_data), n_samples, replace=False)
        original_sample = original_data[indices]
        reduced_sample = reduced_data[indices]
    else:
        original_sample = original_data
        reduced_sample = reduced_data
    
    # 计算降维前后的余弦相似度
    similarities = []
    for i in range(len(original_sample)):
        # 归一化向量
        orig_vec = original_sample[i] / np.linalg.norm(original_sample[i])
        red_vec = reduced_sample[i] / np.linalg.norm(reduced_sample[i])
        # 计算余弦相似度
        similarity = np.dot(orig_vec, red_vec)
        similarities.append(similarity)
    
    return np.mean(similarities)

def main():
    # 加载parquet文件
    print("加载parquet文件...")
    input_path = "data/qwen_7B_vl_mean_L2_2_48.parquet"  # 请替换为实际的parquet文件路径
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(input_path)
        print(f"成功加载数据: {len(df)}行")
    except Exception as e:
        print(f"加载数据出错: {e}")
        return
    
    # 检查数据格式
    if 'item_id' not in df.columns or 'embedding' not in df.columns:
        print("错误: 数据格式不正确，需要'item_id'和'embedding'列")
        return
    
    # 提取id和向量
    item_ids = df['item_id'].values
    embeddings = np.array(df['embedding'].tolist())
    
    # 检查向量维度
    input_dim = embeddings.shape[1]
    print(f"原始向量维度: {input_dim}")
    
    # 设置PCA降维参数
    n_components = min(128, input_dim)  # 可以根据需要调整降维后的维度
    
    print(f"\n执行PCA降维 ({input_dim} -> {n_components})...")
    # 执行PCA降维
    pca_model = PCA(n_components=n_components, random_state=42)
    reduced_data = pca_model.fit_transform(embeddings)
    
    # 计算降维前后的平均余弦相似度
    print("计算降维前后的平均余弦相似度...")
    avg_cos_sim = calculate_avg_cosine_similarity(embeddings, pca_model.inverse_transform(reduced_data))
    
    # 保存降维后的向量
    print("\n保存降维后的向量:")
    save_encoded_vectors(reduced_data, original_ids=item_ids, output_dir="pca_all")
    
    # 额外保存参数信息
    params_info = {
        "model_type": "PCA",
        "input_dim": input_dim,
        "n_components": n_components,
        "compression_ratio": f"{n_components/input_dim:.2%}",
        "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "total_explained_variance": float(sum(pca_model.explained_variance_ratio_)),
        "avg_cosine_similarity": float(avg_cos_sim),
        "random_seed": 42
    }
    
    # 将参数信息保存为JSON
    import json
    with open("pca_all/model_info.json", "w") as f:
        json.dump(params_info, f, indent=4)
    
    print("\n任务完成!")
    print(f"原始维度: {input_dim} -> 降维维度: {n_components}")
    print(f"压缩率: {n_components/input_dim:.2%}")
    print(f"总解释方差比例: {sum(pca_model.explained_variance_ratio_):.4f}")
    print(f"平均余弦相似度: {avg_cos_sim:.6f}")
    print(f"PCA参数已保存到 pca_all/model_info.json")


if __name__ == "__main__":
    main()
