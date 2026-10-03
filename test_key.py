"""Проверка ключа: запрашивает список моделей и сохраняет его в data/models.json."""
import json, urllib.request
from pathlib import Path

env = {}
for line in Path(__file__).with_name(".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

req = urllib.request.Request(env["LLM_BASE_URL"].rstrip("/") + "/models",
                             headers={"Authorization": f"Bearer {env['LLM_API_KEY']}"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
except Exception as e:
    raise SystemExit(f"Ошибка: {e}")

models = data.get("data", data)
out = Path(__file__).parent / "data" / "models.json"
out.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Ключ работает. Моделей: {len(models)}. Список сохранён в {out}")
