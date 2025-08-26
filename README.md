
# Sepsis–ARDS Gut Dysbiosis Risk Predictor (Streamlit)

这是一个基于 Python/Streamlit 的在线预测工具模板。将您的训练模型 `final_model.pkl` 放在同一目录，
即可本地运行或一键部署到 Streamlit Cloud。

## 一、本地运行
1. 安装依赖：`pip install -r requirements.txt`
2. 启动应用：`streamlit run app.py`
3. 浏览器打开 http://localhost:8501

> 若模型中包含 LightGBM/CatBoost/XGBoost，请确保 `requirements.txt` 与训练环境版本相容。

## 二、部署到 Streamlit Cloud（推荐）
1. 新建 GitHub 仓库，上传以下文件：
   - `app.py`
   - `requirements.txt`
   - `final_model.pkl`（您的已训练模型）
2. 登录 https://streamlit.io/cloud → “New app” → 选择该仓库 → 指定入口文件为 `app.py` → 部署
3. 部署完成后即可获得公开链接，分享给临床团队使用。

## 三、重要提示（特征顺序与预处理）
- **最稳妥做法**：在训练阶段将 **预处理（标准化/独热编码等）与模型一起封装为 sklearn `Pipeline`**，并 `pickle` 为 `final_model.pkl`。
  这样推理阶段只需要传入与训练一致的列名/DataFrame，即可直接 `predict_proba`。
- 如果您的 `final_model.pkl` 仅包含最终分类器（不是 Pipeline），请：
  1) 在 `app.py` 中手动复现与训练完全一致的特征工程；
  2) 确保 **特征列顺序、类型、单位** 与训练保持一致。
- 默认示例使用以下特征（按顺序）：
  `["age", "spo2_max", "ph", "absolute_monocytes", "vent", "hr_min"]`
  请根据您的训练数据实际情况在 `app.py` 中修改。

## 四、可解释性（SHAP）
- 若要显示 SHAP 值：
  1) 在训练阶段保留 **特征名**（`feature_names_in_`）并使用与推理一致的 DataFrame；
  2) 在 `app.py` 的 “可解释性” 区域中接入 `shap.Explainer`。
- 如果云端安装 `shap` 失败，可先移除 `requirements.txt` 中的 `shap`，或改为显示模型自带的 `feature_importances_`。

## 五、常见问题
- **加载模型报错**：请确认 `final_model.pkl` 与 `scikit-learn` 版本兼容；若版本不同，建议在训练环境中 `joblib.dump` 并记下版本。
- **预测维度不匹配**：确认输入特征顺序与训练一致；若训练包含类别特征（如 Race），请在 Pipeline 中包含 `OneHotEncoder` 并在推理时传入字符串类别。
- **外部验证性能下降**：这是多中心迁移常见现象，建议收集新中心数据进行再校准（比如 Platt scaling/Isotonic regression）或轻量微调。

## 六、肠-肺轴临床背景（写给用户看的）
脓毒症导致的肠屏障破坏和菌群紊乱可通过“肠-肺轴”加剧肺部炎症，影响 ARDS 的严重程度与预后。
模型的早期预测可帮助医生提前识别高风险个体，尽早实施益生菌/营养支持/密切监测等干预。

---
如需将本模版改造成 Flask/FastAPI 或者需要对接院内 EMR 系统，欢迎在 `app.py` 基础上扩展。
