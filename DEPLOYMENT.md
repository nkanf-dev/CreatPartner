# CreatPartner 部署指南

## 本地开发部署

### 1. 环境准备
```bash
# Python 3.12+
python --version

# 安装依赖
python install.py

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件添加API密钥
```

### 2. 启动MongoDB
```bash
# 本地MongoDB
mongod --dbpath /path/to/data

# 或使用Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 3. 运行应用
```bash
# 命令行版本
python main.py

# Web界面版本
streamlit run app.py
```

## 生产环境部署

### 使用Docker

1. **创建Dockerfile**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY . .

# 安装Python依赖
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **创建docker-compose.yml**
```yaml
version: '3.8'

services:
  creatpartner:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGODB_URI=mongodb://mongodb:27017
      - JINA_API_KEY=${JINA_API_KEY}
    depends_on:
      - mongodb
    
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    
volumes:
  mongodb_data:
```

3. **部署命令**
```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 使用云服务

#### MongoDB Atlas
1. 创建Atlas集群
2. 获取连接字符串
3. 配置网络访问白名单
4. 更新MONGODB_URI环境变量

#### Heroku部署
1. **创建Procfile**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **部署命令**
```bash
# 初始化Git
git init
git add .
git commit -m "Initial commit"

# 创建Heroku应用
heroku create your-app-name

# 设置环境变量
heroku config:set OPENAI_API_KEY=your_key
heroku config:set MONGODB_URI=your_mongodb_uri

# 部署
git push heroku main
```

#### Vercel部署
1. **创建vercel.json**
```json
{
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/app.py"
    }
  ]
}
```

2. **部署命令**
```bash
# 安装Vercel CLI
npm i -g vercel

# 部署
vercel --prod
```

## 环境变量配置

### 必需变量
- `OPENAI_API_KEY`: OpenAI API密钥
- `MONGODB_URI`: MongoDB连接字符串

### 可选变量
- `JINA_API_KEY`: Jina搜索API密钥
- `DEFAULT_MODEL`: AI模型选择
- `MAX_SEARCH_RESULTS`: 搜索结果数量
- `DEBUG`: 调试模式

## 监控和维护

### 日志管理
```bash
# 查看应用日志
docker-compose logs creatpartner

# 查看MongoDB日志
docker-compose logs mongodb
```

### 数据备份
```bash
# MongoDB备份
mongodump --uri="your_mongodb_uri" --out=backup/

# 恢复数据
mongorestore --uri="your_mongodb_uri" backup/
```

### 性能监控
- 使用Prometheus + Grafana监控系统指标
- 配置alerting规则
- 监控API调用频率和成本

## 安全建议

1. **API密钥管理**
   - 使用环境变量存储敏感信息
   - 定期轮换API密钥
   - 限制API密钥权限

2. **网络安全**
   - 使用HTTPS
   - 配置防火墙规则
   - 限制数据库访问

3. **数据保护**
   - 加密敏感数据
   - 定期备份
   - 实施访问控制

## 故障排除

### 常见问题

1. **连接MongoDB失败**
```bash
# 检查MongoDB服务状态
systemctl status mongod

# 检查网络连接
ping your-mongodb-host
```

2. **API调用失败**
```bash
# 检查API密钥
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

3. **依赖安装失败**
```bash
# 清理缓存
pip cache purge

# 重新安装
pip install -r requirements.txt --force-reinstall
```

### 调试模式
```bash
# 启用调试模式
export DEBUG=true

# 查看详细日志
python main.py --verbose
```
