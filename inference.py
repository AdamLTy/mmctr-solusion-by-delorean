import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd

# 指定本地模型路径
model_path = "qwen2.5-vl-7B"  # 修改为您的本地模型路径

# 加载本地模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

# 如果您想启用flash_attention_2以获得更好的加速和内存节省
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# 加载本地处理器
processor = AutoProcessor.from_pretrained(model_path)


def get_embeddings(messages, pooling_method="mean"):
    """
    获取多模态输入的嵌入向量
    
    参数:
    messages - 包含文本和图像的消息列表
    pooling_method - 池化方法，可选 'mean'(平均池化)、'last'(最后一个token)或'cls'(第一个token)
    
    返回:
    嵌入向量
    """
    # 准备输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 处理本地图像路径
    for i, img in enumerate(image_inputs):
        if isinstance(img, str) and os.path.exists(img):
            from PIL import Image
            image_inputs[i] = Image.open(img).convert('RGB')
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # 使用模型获取嵌入向量(不生成文本)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 获取倒数第二层的隐藏状态
        last_hidden_states = outputs.hidden_states[-2]
        if pooling_method == "last":
            # 使用最后一个token的隐藏状态作为整个序列的表示
            embedding = last_hidden_states[0, -1, :]
            # 进行L2归一化
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
        elif pooling_method == "cls":
            # 使用第一个token(CLS token)的隐藏状态
            embedding = last_hidden_states[0, 0, :]
            # 进行L2归一化
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
        else:  # 默认使用mean
            # 使用所有token的平均隐藏状态
            embedding = torch.mean(last_hidden_states, dim=1).squeeze()
            # 进行L2归一化
            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
    
    return embedding

def process_items_and_generate_embeddings(parquet_file, images_folder, output_file, batch_size=100):
    """
    处理parquet文件，为每个item生成embedding并保存结果
    
    参数:
    - parquet_file: 输入的parquet文件路径
    - images_folder: 存储图片的文件夹路径
    - output_file: 输出的parquet文件路径
    - batch_size: 处理的批次大小，用于中间保存
    """
    
    # 读取parquet文件
    print(f"正在读取parquet文件: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # 检查列名并确保数据格式正确
    # 假设第一列是item_id, 第二列是文本内容
    cols = df.columns.tolist()
    id_col = cols[0]
    text_col = cols[1]
    
    # 创建新列用于存储embeddings
    df['embedding'] = None
    
    # 检查是否存在中间结果文件，如果存在则加载
    temp_file = f"{output_file}.temp.parquet"
    if os.path.exists(temp_file):
        print(f"发现中间结果文件，加载已处理的数据...")
        temp_df = pd.read_parquet(temp_file)
        
        # 获取已处理的item_ids
        processed_ids = set(temp_df[id_col].tolist())
        print(f"已处理的item数量: {len(processed_ids)}")
        
        # 更新主DataFrame中已处理的embeddings
        df = pd.concat([df[~df[id_col].isin(processed_ids)], temp_df])
    else:
        processed_ids = set()
    
    # 获取待处理的items
    items_to_process = df[~df[id_col].isin(processed_ids)].copy()
    
    if len(items_to_process) == 0:
        print("所有数据已处理完毕，无需再次处理")
        df.to_parquet(output_file)
        return
    
    print(f"待处理的item数量: {len(items_to_process)}")
    
    # 初始化进度条
    pbar = tqdm(total=len(items_to_process))
    
    # 批次处理
    batch_count = 0
    for idx, row in items_to_process.iterrows():
        item_id = row[id_col]
        text_content = row[text_col]
        image_path = os.path.join(images_folder, f"{item_id}.jpg")
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在 - {image_path}")
            df.loc[idx, 'embedding'] = np.zeros(1024).tolist()  # 使用零向量作为占位符，转换为Python列表
            continue
        
        try:
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Do not output the stem of multiple-choice questions. Check if you have completed all questions."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": f"""Please provide a detailed and comprehensive multi-dimensional analysis and description of the following video content:

Video title and tags: {text_content}

Please systematically analyze the video from the following dimensions, providing detailed descriptions for each:

1.Topic Analysis: What is the main content category and specific topic of this video? Which field does it belong to?

2.Visual Style: Analyze the visual elements of the cover, including color scheme, composition, main visual objects, aesthetic style, etc.

3.Emotional Tone: What is the main emotion and atmosphere conveyed by the video? Is it positive, negative, or neutral?

4.Target Audience: Based on the content and presentation, what type of viewers is this video most likely to attract?

5.Professional Level: How is the professional level and production quality of the video?

6.Content Depth: Is the video content easy to understand or deeply professional? How is the information density?

7.Timeliness Characteristics: Is the content evergreen or news-related? Are there any time-sensitive elements?

8.Interaction Potential: What kind of viewer reactions and interactions does this type of content typically generate?

9.Unique Selling Points: Compared to similar videos, what is unique about this video?

10.Style Positioning: Is the presentation style of the video formal/informal, professional/casual, educational/entertaining?

11.Technical Application: Does the video use specific editing techniques, animation effects, sound design, etc.? How do these techniques enhance the content's appeal?

12.Cultural Relevance: How does the video relate to current cultural, social trends, or hot topics? Does it include specific cultural symbols or references?

13.Business Potential: What is the commercial value of the video? Is it beneficial for brand promotion, product marketing, or other business purposes?

14.Narrative Structure: What kind of narrative approach does the video use? Is it linear, non-linear, or does it employ any special narrative structures?

15.Audio Elements: Analyze the use of audio in the video, including background music, sound effects, voiceovers, etc. How do these elements complement the visual content?

16.Engagement Level: How does the video content attract and maintain viewer attention? Are there specific strategies to increase viewer retention?

17.Educational Value: Does the video have educational significance? What can viewers learn from it?

18.Cross-platform Adaptability: Is this video content suitable for distribution across different social media platforms? What adjustments might be needed?

19.Innovativeness: What innovative aspects does the video present in terms of content, form, or technology? How does it break from tradition or convention?

20.Credibility: How credible is the video content? Does it provide reliable information sources or expert opinions?

21.Social Impact: What kind of impact might this video have on society? Does it address important social issues or values?

22.Long-tail Effect: Does the video content have enduring appeal? How does it continue to attract audiences after the initial buzz has faded?

23.Brand Consistency: If it is a branded video, is it consistent with the brand's overall image and messaging? How does it embody the brand's characteristics?

24.Audience Engagement Strategy: Does the video use specific strategies to encourage audience engagement, such as call-to-action phrases or interactive segments?

25.Spread Speed: How quickly does the video content spread on social media or other channels? Does it have the potential for rapid spread?

26.Community Building: Does the video help build an audience community or social circle? How does it foster interaction and discussion among viewers?

27.Ethics and Social Responsibility: Does the video content comply with ethical standards and social responsibility requirements? Does it promote positive social values?

28.Technical Adaptability："Does this video easily adapt to new technologies or platforms, such as virtual reality, augmented reality, or AI-generated content?"

29.User-generated Content Potential）："Does the video encourage the creation of user-generated content related to it? Does it have the potential to spark related challenges or creative activities?"

30.Data Analytics and Feedback："Does the video use audience feedback, interactive data, or analytical tools to improve content or performance? How are these data utilized to enhance the effectiveness of future videos?"

31.Interdisciplinary Application：Is the video content applicable across different disciplines or fields? How does it integrate multiple types of knowledge or methodologies?

32.Storytelling Ability：How does the video convey information or emotions through its narrative structure? Is it effective in engaging audiences with its story?

33.Memorability: How easily is the video content remembered by audiences? Are there specific elements or techniques that make it leave a lasting impression on viewers?

34.Psychological Impact Analysis: What is the psychological impact of the video on the audience? Are there specific psychological elements that enhance its impact?

35.Globalization Adaptability: Can the video content be adapted and disseminated across cultures and languages? Are there language or cultural barriers?

36.Continuous Improvement: Does the video content allow or encourage continuous updates and improvements? Are there plans to regularly update, expand, or improve the content?

37.Evolution Potential: Does the video content have the potential to evolve or be adapted into other forms, such as series videos, derivative works, courses, or books?

38.Intellectual Property Protection: How is the intellectual property of the video content protected? Are there any relevant copyright statements or protective measures?

39.Sustainability Impact: Is the video content related to sustainable development goals? Does it promote environmental protection, social equity, or economic sustainability?

40.Partnership and Collaboration: Does the video involve collaboration with other creators, brands, or organizations? How does the collaboration enhance the value of the video?

41.Scenic Application Potential: Is the video content suitable for use in specific scenarios, such as conferences, classrooms, exhibitions, etc.? How can it be adapted to meet the needs of different scenarios?

42.Cross-industry Application: Can the video content be applied across multiple industries or fields? How does it play a valuable role in different industries?

43.Risk Assessment and Management: Video content: Does it pose any risks or controversy? How can these risks be effectively managed to ensure the safety and compliance of the content?

44.echnological Enablement Efficiency: Does the video utilize technology to enhance production efficiency, reduce costs, or improve interactive experiences? How do these technologies function in practical application?

45.Assessment and Certification System: Can the video content be validated by a specific assessment or certification system to verify its quality, authenticity, or professionalism? How do these certifications enhance the credibility and persuasiveness of the content?

46.Cross-media Integration: Can the video content be integrated with other media forms (such as TV, film, audio programs)? How can multiple media platforms be utilized to expand its influence?

47.Virtual and Reality Fusion: Does the video incorporate virtual reality (VR) or augmented reality (AR) technology? How can these technologies be used to enhance audience experience and interaction?

48.Textual Summarization and Abstraction: Can the video content be effectively summarized or abstracted into text for quick understanding or reference in other media? How can the content be condensed and refined?

Please ensure your analysis is comprehensive, specific, and insightful, providing sufficiently detailed descriptions for each dimension."""
                        }
                    ],
                }
]

            
            # 获取嵌入向量
            embedding = get_embeddings(messages)
            
            # 将numpy数组转换为Python列表
            df.at[idx, 'embedding'] = embedding.tolist()
            
            # 更新已处理集合
            processed_ids.add(item_id)
            
            # 更新进度条
            pbar.update(1)
            
            # 批次计数
            batch_count += 1
            
            # 每处理batch_size个item保存一次中间结果
            if batch_count % batch_size == 0:
                # 保存中间结果
                temp_save_df = df[df[id_col].isin(processed_ids)].copy()
                temp_save_df.to_parquet(temp_file)
                print(f"\n已保存中间结果，当前处理了 {len(processed_ids)} 个items")
                
        except Exception as e:
            print(f"\n处理item {item_id}时出错: {str(e)}")
            # 继续处理下一个item
            continue
    
    # 关闭进度条
    pbar.close()
    
    # 保存最终结果
    print("处理完毕，保存最终结果...")
    
    # 将列表格式的embedding转换回numpy数组用于验证
    sample_idx = df.index[0]
    sample_embedding_list = df.at[sample_idx, 'embedding']
    sample_embedding = np.array(sample_embedding_list)
    print(f"样本embedding维度: {sample_embedding.shape}")
    
    # 保存结果
    df.to_parquet(output_file)
    
    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    # 参数
    parquet_file = "data/MicroLens_1M_x1/item_feature.parquet"  # 修改为您的输入文件路径
    images_folder = "data/item_images"  # 修改为您的图片文件夹路径
    output_file = "data/qwen_7B_vl_mean_L2_2_48.parquet"  # 输出文件路径
            
        
    process_items_and_generate_embeddings(
        parquet_file=parquet_file,
        images_folder=images_folder,
        output_file=output_file,
        batch_size=1000
    )
    
    # 测试读取结果
    test_df = pd.read_parquet(output_file)
    sample_embedding_list = test_df.iloc[0]['embedding']
    sample_embedding = np.array(sample_embedding_list)
    print(f"最终结果中的embedding维度: {sample_embedding.shape}")