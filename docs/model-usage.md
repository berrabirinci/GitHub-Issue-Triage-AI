# Issue kategori modeli

## Dört kategori

Model `Documentation`, `Bug`, `Question` ve `Feature Request` kategorilerini tahmin eder. Eğitimde yalnızca bu kategorilerden biri açıkça atanmış issue'lar kullanılır. Labelsız, belirsiz ve eşleşmeyen kayıtlar eğitim dışında kalır.

## Veri genişletme

Bug dışındaki sınıflar için dengeli örnek toplamak üzere:

```powershell
python src/data/collect_category_expansion.py
python src/data/preprocess.py
``` 

Toplanan örnekler ayrı bir DVC izli ham dosyada tutulur ve ana eğitim verisine eklenir.

## Eğitme

Proje klasöründe:

```powershell
python src/models/train.py
```

Eğitim betiği başlık ve issue açıklamasını kullanır. Aynı 419 issue'luk sabit test kümesiyle temel ve genişletilmiş veri karşılaştırılır; kaydedilen son model ise tüm güvenilir etiketli kayıtlarla yeniden eğitilir. Ayrı bir doğrulama kümesi henüz eklenmedi.

Mevcut sınıflandırıcı TF-IDF özellikleriyle `LinearSVC` kullanır. Sunumdaki Logistic Regression temel modeliyle karşılaştırma henüz yapılmadı.

- Eğitilmiş model: `models/issue_category_model.joblib`
- Test ölçümleri: `data/processed/model_evaluation.json`

## Bağımsız dış test

Eğitim kaynaklarında bulunmayan issue'lar `data/test/raw/github_issues_external_test.csv` dosyasında ayrı tutulur. Puanlanabilir kayıtlar `data/test/processed/github_issues_external_test.csv`, eşleşmeyen veya belirsiz kayıtlar ise `data/test/processed/github_issues_external_test_review.csv` dosyasındadır.

Son modeli bu bağımsız veriyle yeniden ölçmek için:

```powershell
python src/models/evaluate_external_test.py
```

Sonuç `data/test/processed/model_external_test_evaluation.json` dosyasına kaydedilir. Yalnızca tek ve güvenilir bir kategoriye eşleşen kayıtlar puanlanır. Dış depoların etiket kullanımı ve kategori dağılımı farklı olabileceğinden, kategori başına örnek sayılarını ve macro F1 değerini doğrulukla birlikte değerlendirin.

## Yeni issue için tahmin

```powershell
python src/models/predict.py --title "Issue başlığı" --body "Issue açıklaması"
```

Etiket ve kategori sütunları tahmin girdisi olarak kullanılmaz; bunlar modelin öğrenmesi gereken hedef cevaplardır.


