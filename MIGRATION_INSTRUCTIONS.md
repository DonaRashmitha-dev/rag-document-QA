OpenAI -> Google Gemini Migration Instructions
What You Need to Do
1. Download These Files
All files are generated and ready. Download each one from the links below.
2. Replace Files in Your Project
Copy these files into your `rag_project` folder, overwriting the old ones:
Download	Replace This File in Your Project
pyproject.toml	`pyproject.toml`
config.py	`app/core/config.py`
embedder.py	`app/core/embedder.py`
llm.py	`app/core/llm.py`
vector_store.py	`app/core/vector_store.py`
.env.example	`.env.example`
conftest.py	`tests/conftest.py`
test_embedder.py	`tests/unit/test_embedder.py`
3. Delete Old FAISS Index
The dimension changed from 1536 (OpenAI) to 768 (Gemini). Old index is incompatible.
In PowerShell:
```powershell
cd "C:\Users\donar\OneDrive\Desktop\rag_project"
Remove-Item -Recurse -Force .\data\faiss_index
```
4. Reinstall Dependencies
```powershell
pip install -e ".[dev]"
```
If this fails, try:
```powershell
pip install langchain-google-genai==2.1.9
pip install -e ".[dev]"
```
5. Set Your Real API Key
Create/edit `.env` file in your project root:
```
GOOGLE_API_KEY=AIza...your-real-key...
JWT_SECRET=any-random-secret-string-here
```
6. Run Tests
```powershell
$env:GOOGLE_API_KEY="test-key"
$env:JWT_SECRET="test-secret"
pytest tests/ -v --tb=long
```
Should show: 22 passed
7. Run the Real Server
```powershell
$env:GOOGLE_API_KEY="AIza...your-real-key..."
$env:JWT_SECRET="your-real-secret"
flask --app app:create_app run --port 8000
```
8. Test with curl
In another terminal:
Get JWT token:
```powershell
cd "C:\Users\donar\OneDrive\Desktop\rag_project"
$env:JWT_SECRET="your-real-secret"
python -c "from app.middleware.auth import create_access_token; print(create_access_token())"
```
Ingest a document:
```powershell
curl -X POST http://localhost:8000/ingest `
  -H "Authorization: Bearer <paste-token>" `
  -F "file=@test_doc.txt"
```
Query it:
```powershell
curl -X POST http://localhost:8000/query `
  -H "Authorization: Bearer <paste-token>" `
  -H "Content-Type: application/json" `
  -d '{"query": "How long do refunds take?", "top_k": 3}'
```
Important Notes
Free tier limits: Gemini 1.5 Flash = 1,500 requests/day, 1M tokens/day. More than enough for testing.
If tests fail: Make sure `GOOGLE_API_KEY` environment variable is set before running pytest.
If you see import errors: Make sure `langchain-google-genai==2.1.9` installed successfully.
Cursor is locked until your monthly reset. Use these files directly -- no AI needed for this migration.