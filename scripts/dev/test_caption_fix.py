import asyncio
import sys
import os

# 确保导入根目录的 videorag
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vidgraphrag._llm import dashscope_caption_complete

async def test_caption():
    """测试 DashScope Caption 函数"""
    
    # 模拟配置
    api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-your-key")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    print(f" Testing dashscope_caption_complete...")
    print(f"   API Key: {api_key[:10]}..." if api_key else "    No API Key")
    print(f"   Base URL: {base_url}")
    
    # 构造测试内容（模拟视频帧）
    content_list = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,test"}},
        {"type": "text", "text": "描述这个视频片段的内容"}
    ]
    
    try:
        result = await dashscope_caption_complete(
            model_name="qwen-vl-max",
            content_list=content_list,
            ali_dashscope_api_key=api_key,
            ali_dashscope_base_url=base_url
        )
        print(f" Caption result: {result[:200] if result else 'None'}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_caption())
    sys.exit(0 if success else 1)
