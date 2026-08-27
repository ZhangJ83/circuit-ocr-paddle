import os, time
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ['HF_HOME'] = 'F:/hf_cache'
os.makedirs('F:/hf_cache/xet/logs', exist_ok=True)

from huggingface_hub import HfApi, upload_file
api = HfApi()
repo = 'yingchu83/CircuitOCR'

# Fix: demo.launch() -> demo.launch(server_name="0.0.0.0", server_port=7860)
app_path = r'g:\mimo_project\circuit_ocr\hf_space\app.py'
content = open(app_path, encoding='utf-8').read()
old = 'demo.launch()'
new = 'demo.launch(server_name="0.0.0.0", server_port=7860)'
if old in content:
    content = content.replace(old, new)
    open(app_path, 'w', encoding='utf-8').write(content)
    print('Fixed launch() call')
else:
    print('launch() already fixed or not found')

# Upload
upload_file(path_or_fileobj=app_path, path_in_repo='app.py',
    repo_id=repo, repo_type='space',
    commit_message='Fix: demo.launch(server_name, server_port) for HF Spaces')
print('Uploaded')

# Wait for running
for i in range(6):
    time.sleep(20)
    info = api.get_space_runtime(repo)
    print(f'[{i+1}] {info.stage}')
    if info.stage == 'RUNNING':
        print('SUCCESS! https://huggingface.co/spaces/yingchu83/CircuitOCR')
        break
    elif info.stage in ('RUNTIME_ERROR', 'BUILD_ERROR'):
        err = info.raw.get('errorMessage', '')
        if 'ValueError' in err:
            print(err[err.find('ValueError'):][:300])
        elif 'TypeError' in err:
            print(err[err.find('TypeError'):][:300])
        else:
            print(err[:400])
        break
