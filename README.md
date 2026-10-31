# GitHub Issue Triage AI

GitHub issue'larını otomatik sınıflandırmak için geliştirilen bir makine öğrenmesi projesi.

## Veri toplama

Kolektör; FastAPI, MLflow, scikit-learn, pandas ve HTTPX depolarından issue başlıklarını, açıklamalarını ve mevcut etiketlerini alır.

## Kurulum

```powershell
poetry install

Issue toplama (poweshell)
poetry run python -m src.data.collect
Ham veriler data/raw/github_issues.jsonl dosyasına kaydedilir. GitHub token'ını kod veya dosyalara eklemeyin.
Planlanan çalışmalar
- Veriyi temizleme ve etiketleri dört sınıfa eşleme
- TF-IDF ve Logistic Regression modeli eğitme
- FastAPI ile tahmin servisi oluşturma
- MLflow, Docker ve izleme araçlarını ekleme