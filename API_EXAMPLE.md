# API使用示例

## Python示例

```python
import requests

# 上传图片并去除水印
url = "http://localhost:8000/api/remove-watermark"

with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    data = {'watermark_ratio': 0.15}
    response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open('result.jpg', 'wb') as f:
        f.write(response.content)
    print("处理成功！")
else:
    print(f"错误: {response.json()}")
```

## JavaScript示例

```javascript
// 使用FormData上传
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('watermark_ratio', 0.15);

fetch('http://localhost:8000/api/remove-watermark', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'processed_image.jpg';
    a.click();
});
```

## curl示例

```bash
# 基本用法
curl -X POST "http://localhost:8000/api/remove-watermark" \
  -F "file=@test.jpg" \
  -F "watermark_ratio=0.15" \
  --output result.jpg

# 并发测试
for i in {1..10}; do
  curl -X POST "http://localhost:8000/api/remove-watermark" \
    -F "file=@test.jpg" \
    -F "watermark_ratio=0.15" \
    --output "result_$i.jpg" &
done
wait
```

## 并发测试脚本

创建 `test_concurrent.py`:

```python
import asyncio
import aiohttp
import time

async def process_image(session, url, file_path, index):
    with open(file_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename='test.jpg')
        data.add_field('watermark_ratio', '0.15')
        
        start_time = time.time()
        async with session.post(url, data=data) as response:
            if response.status == 200:
                result = await response.read()
                with open(f'result_{index}.jpg', 'wb') as out:
                    out.write(result)
                elapsed = time.time() - start_time
                print(f"任务 {index} 完成，耗时: {elapsed:.2f}秒")
            else:
                print(f"任务 {index} 失败: {response.status}")

async def main():
    url = "http://localhost:8000/api/remove-watermark"
    file_path = "test.jpg"
    num_concurrent = 10
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_image(session, url, file_path, i)
            for i in range(num_concurrent)
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

运行并发测试：

```bash
pip install aiohttp
python test_concurrent.py
```

