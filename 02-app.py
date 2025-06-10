import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面配置
st.set_page_config(
    page_title="5A级景区推荐系统",
    page_icon="🏞️",
    layout="wide"
)

# 页面标题和介绍
st.title("全国5A级景区推荐系统")
st.markdown("基于用户评论的情感分析和内容相似度的景区推荐")

# 初始化状态变量
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False


# 自定义工具类
class ScenicRecommender:
    def __init__(self):
        self.df = None
        self.algo = None
        self.scenic_reviews = None
        self.similarity_matrix = None
        self.scenic_index = None
        self.training_time = 0

    def load_and_preprocess_data(self, scenic_file, comment_file):
        try:
            # 加载数据
            excel_file = pd.ExcelFile(scenic_file)
            df_scenic = excel_file.parse('5A')
            df_comment = pd.read_excel(comment_file)

            # 数据预处理
            df_scenic = df_scenic.dropna(subset=['dth_title'])
            df_comment = df_comment.dropna(subset=['景区名称'])
            self.df = pd.merge(df_scenic, df_comment, left_on='dth_title', right_on='景区名称', how='outer')

            # 提取关键词和情感分析
            def extract_keywords(text):
                if isinstance(text, float) and pd.isna(text):
                    return []
                return jieba.lcut(text)

            def get_sentiment_score(text):
                if isinstance(text, float) and pd.isna(text):
                    return None
                return SnowNLP(text).sentiments

            self.df['关键词'] = self.df['评论内容'].apply(extract_keywords)
            self.df['情感得分'] = self.df['评论内容'].apply(get_sentiment_score)
            self.df['映射评分'] = self.df['情感得分'] * 4 + 1  # 映射到1-5分

            st.session_state.data_ready = True
            return True
        except Exception as e:
            st.error(f"数据加载出错: {e}")
            return False

    def train_model(self):
        try:
            if self.df is None:
                st.error("请先加载数据")
                return False

            # 创建Reader和Dataset
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(self.df[['用户ID', '景区名称', '映射评分']], reader)

            # 划分训练集和测试集
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

            # 训练SVD模型
            start_time = time.time()
            self.algo = SVD(
                n_factors=50,
                n_epochs=20,
                lr_all=0.005,
                reg_all=0.02,
                random_state=42
            )
            self.algo.fit(trainset)
            self.training_time = time.time() - start_time

            # 内容推荐 - 基于TF-IDF
            self.scenic_reviews = self.df.groupby('景区名称')['评论内容'].agg(lambda x: ' '.join(x)).reset_index()

            # 自定义分词器
            def chinese_tokenizer(text):
                return jieba.lcut(text)

            vectorizer = TfidfVectorizer(
                tokenizer=chinese_tokenizer,
                stop_words=['的', '了', '和', '是', '在'],
                ngram_range=(1, 2),
                max_features=5000
            )

            tfidf_matrix = vectorizer.fit_transform(self.scenic_reviews['评论内容'])
            self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            self.scenic_index = {name: i for i, name in enumerate(self.scenic_reviews['景区名称'])}

            st.session_state.model_ready = True
            return True
        except Exception as e:
            st.error(f"模型训练出错: {e}")
            return False

    def get_similar_scenics(self, scenic_name, top_n=5):
        if scenic_name not in self.scenic_index:
            return pd.DataFrame({"景区名称": [f"抱歉，景区 '{scenic_name}' 不在数据集中。"], "相似度得分": [0]})

        idx = self.scenic_index[scenic_name]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_scenics = sim_scores[1:top_n + 1]

        result = []
        for i, score in top_scenics:
            result.append({
                '景区名称': self.scenic_reviews.iloc[i]['景区名称'],
                '相似度得分': score
            })

        return pd.DataFrame(result)

    def hybrid_recommend(self, user_id, scenic_name, top_n=5, content_weight=0.7, collab_weight=0.3):
        # 内容推荐
        content_rec = self.get_similar_scenics(scenic_name, top_n * 2)

        # 协同过滤推荐
        items = self.df['景区名称'].unique()
        user_items = self.df[self.df['用户ID'] == user_id]['景区名称'].tolist()
        unrated_items = [item for item in items if item not in user_items]

        predictions = []
        for item in unrated_items:
            pred = self.algo.predict(user_id, item)
            predictions.append((item, pred.est))

        collab_rec = pd.DataFrame(predictions, columns=['景区名称', '预测评分'])
        collab_rec = collab_rec.sort_values('预测评分', ascending=False).head(top_n * 2)

        # 混合推荐
        merged_rec = pd.merge(content_rec, collab_rec, on='景区名称', how='outer')
        merged_rec['综合得分'] = (content_weight * merged_rec['相似度得分'].fillna(0) +
                                  collab_weight * merged_rec['预测评分'].fillna(0))

        return merged_rec.sort_values('综合得分', ascending=False).head(top_n)

    def get_scenic_details(self, scenic_name):
        return self.df[self.df['景区名称'] == scenic_name].iloc[0]

    def get_user_recommendations(self, user_id, top_n=5):
        items = self.df['景区名称'].unique()
        user_items = self.df[self.df['用户ID'] == user_id]['景区名称'].tolist()
        unrated_items = [item for item in items if item not in user_items]

        predictions = []
        for item in unrated_items:
            pred = self.algo.predict(user_id, item)
            predictions.append((item, pred.est))

        recommendations = pd.DataFrame(predictions, columns=['景区名称', '预测评分'])
        return recommendations.sort_values('预测评分', ascending=False).head(top_n)

    def get_user_reviewed_scenics(self, user_id):
        return self.df[self.df['用户ID'] == user_id]['景区名称'].unique()

    def get_scenic_comments(self, scenic_name, top_n=3):
        return self.df[self.df['景区名称'] == scenic_name]['评论内容'].dropna().head(top_n)

    def get_scenic_keywords(self, scenic_name, top_n=10):
        keywords = self.get_scenic_details(scenic_name).get('关键词', [])
        return keywords[:top_n] if keywords else []

    def get_popular_scenics(self, top_n=5):
        return self.df.groupby('景区名称')['用户ID'].count().sort_values(ascending=False).head(top_n)

    def get_regional_distribution(self):
        return self.df.groupby('省份')['景区名称'].nunique().sort_values(ascending=False)

    def get_average_ratings(self):
        return self.df.groupby('景区名称')['映射评分'].mean().sort_values(ascending=False)


# 主程序流程
recommender = ScenicRecommender()

with st.spinner("正在加载数据和训练模型..."):
    success = recommender.load_and_preprocess_data('全国5A级景区.xlsx', '景区评论数据集.xlsx')

    if success:
        success = recommender.train_model()

        if success:
            st.success(f"数据加载和模型训练完成！训练耗时: {recommender.training_time:.2f}秒")

            # 显示数据统计信息
            st.subheader("数据统计信息")
            col1, col2, col3 = st.columns(3)
            col1.metric("景区数量", len(recommender.df['景区名称'].unique()))
            col2.metric("用户评论数量", len(recommender.df))
            col3.metric("用户数量", len(recommender.df['用户ID'].unique()))

            # 数据可视化
            st.subheader("数据可视化")
            tab1, tab2, tab3 = st.tabs(["景区分布", "评分分布", "热门景区"])

            with tab1:
                regional_distribution = recommender.get_regional_distribution()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=regional_distribution.index, y=regional_distribution.values, ax=ax)
                plt.xticks(rotation=45)
                plt.title("各省5A级景区数量分布")
                st.pyplot(fig)

            with tab2:
                ratings = recommender.get_average_ratings()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(ratings.values, bins=20, kde=True, ax=ax)
                plt.title("景区评分分布")
                st.pyplot(fig)

            with tab3:
                popular_scenics = recommender.get_popular_scenics(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=popular_scenics.values, y=popular_scenics.index, ax=ax)
                plt.title("热门景区（评论数量）")
                st.pyplot(fig)

            # 推荐系统界面
            st.subheader("景区推荐")
            recommendation_type = st.selectbox(
                "推荐类型",
                ["基于内容的推荐", "基于用户的推荐", "混合推荐"]
            )

            if recommendation_type == "基于内容的推荐":
                selected_scenic = st.selectbox(
                    "选择景区",
                    sorted(recommender.df['景区名称'].unique())
                )
                top_n = st.slider("推荐数量", 1, 10, 5)

                if st.button("获取推荐"):
                    with st.spinner("正在生成推荐..."):
                        recommendations = recommender.get_similar_scenics(selected_scenic, top_n)
                        st.write(f"与 **{selected_scenic}** 相似的景区:")
                        st.dataframe(recommendations.style.format({"相似度得分": "{:.2f}"}))

            elif recommendation_type == "基于用户的推荐":
                user_id = st.text_input("输入用户ID", "11111")
                top_n = st.slider("推荐数量", 1, 10, 5)

                if st.button("获取推荐"):
                    with st.spinner("正在生成推荐..."):
                        # 获取用户已经评价过的景区
                        user_scenics = recommender.get_user_reviewed_scenics(user_id)

                        if len(user_scenics) == 0:
                            st.warning(f"用户 {user_id} 没有评价记录，无法进行个性化推荐。")
                        else:
                            st.write(f"用户 {user_id} 已评价的景区:")
                            st.write(", ".join(user_scenics))

                            # 生成推荐
                            recommendations = recommender.get_user_recommendations(user_id, top_n)

                            st.write(f"为用户 {user_id} 推荐的景区:")
                            st.dataframe(recommendations.style.format({"预测评分": "{:.2f}"}))

            else:  # 混合推荐
                user_id = st.text_input("输入用户ID", "11111")
                selected_scenic = st.selectbox(
                    "选择参考景区",
                    sorted(recommender.df['景区名称'].unique())
                )
                top_n = st.slider("推荐数量", 1, 10, 5)
                content_weight = st.slider("内容推荐权重", 0.0, 1.0, 0.7, 0.1)
                collab_weight = st.slider("协同过滤权重", 0.0, 1.0, 0.3, 0.1)

                if st.button("获取推荐"):
                    with st.spinner("正在生成混合推荐..."):
                        recommendations = recommender.hybrid_recommend(
                            user_id,
                            selected_scenic,
                            top_n,
                            content_weight,
                            collab_weight
                        )

                        st.write(f"混合推荐结果 (用户 {user_id} 可能喜欢的类似 **{selected_scenic}** 的景区):")
                        st.dataframe(recommendations[['景区名称', '相似度得分', '预测评分', '综合得分']].style.format({
                            "相似度得分": "{:.2f}",
                            "预测评分": "{:.2f}",
                            "综合得分": "{:.2f}"
                        }))

            # 景区详情查看
            st.subheader("景区详情")
            selected_scenic_detail = st.selectbox(
                "选择景区查看详情",
                sorted(recommender.df['景区名称'].unique())
            )

            if st.button("查看详情"):
                with st.spinner("正在加载景区详情..."):
                    try:
                        scenic_details = recommender.get_scenic_details(selected_scenic_detail)
                        st.write(f"### {scenic_details['景区名称']}")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("平均评分", f"{scenic_details['映射评分']:.1f}/5.0")
                        col1.metric("评论数量", len(recommender.get_scenic_comments(selected_scenic_detail)))
                        col2.metric("省份", scenic_details.get('省份', '未知'))
                        col2.metric("城市", scenic_details.get('城市', '未知'))

                        st.write("**景区简介**")
                        st.write(scenic_details.get('景区简介', '暂无简介'))

                        st.write("**热门评论**")
                        comments = recommender.get_scenic_comments(selected_scenic_detail)
                        for i, comment in enumerate(comments):
                            st.markdown(f"> {comment}")

                        st.write("**关键词**")
                        keywords = recommender.get_scenic_keywords(selected_scenic_detail)
                        if keywords:
                            st.markdown(" ".join([f"#{word}" for word in keywords]))

                        # 评论情感分析
                        st.write("**评论情感分析**")
                        scenic_comments = recommender.df[recommender.df['景区名称'] == selected_scenic_detail]
                        sentiment_scores = scenic_comments['情感得分'].dropna()

                        if len(sentiment_scores) > 0:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.histplot(sentiment_scores, bins=20, kde=True, ax=ax)
                            plt.title("评论情感得分分布")
                            plt.xlabel("情感得分 (0=负面, 1=正面)")
                            st.pyplot(fig)

                            positive_ratio = len(sentiment_scores[sentiment_scores > 0.7]) / len(sentiment_scores)
                            neutral_ratio = len(
                                sentiment_scores[(sentiment_scores >= 0.3) & (sentiment_scores <= 0.7)]) / len(
                                sentiment_scores)
                            negative_ratio = len(sentiment_scores[sentiment_scores < 0.3]) / len(sentiment_scores)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("积极评论比例", f"{positive_ratio:.2%}")
                            col2.metric("中性评论比例", f"{neutral_ratio:.2%}")
                            col3.metric("消极评论比例", f"{negative_ratio:.2%}")
                    except Exception as e:
                        st.error(f"获取景区详情失败: {e}")
else:
st.error("数据加载或模型训练失败，请检查数据文件和依赖库。")