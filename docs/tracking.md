# Veri ve deney takibi

Bu proje DVC ile veri/model dosyalarını, MLflow ile eğitim sonuçlarını takip eder.

## DVC

DVC'nin çalışma ortamı `.dvc-venv` içindedir. PowerShell terminalinde proje klasöründeyken DVC komutlarını şöyle çalıştır:

```powershell
.\dvc-local.ps1 status
```

Yeni bir veri veya model dosyasını DVC'ye eklerken:

```powershell
.\dvc-local.ps1 add data\raw\YENI_DOSYA.csv
git add data\raw\YENI_DOSYA.csv.dvc data\raw\.gitignore
```

CSV dosyaları proje klasöründe normal dosya olarak kalır. `.dvc` dosyaları bu dosyaların sürüm kimliklerini taşır; `.dvc/cache` ise yerel kopyaları tutar. DVC önbellek kopyalama kipinde olduğu için OneDrive ile dosya bağlantısı kaynaklı sorun çıkarmaz. Şu anda DVC uzak yedeği tanımlı değil; bilgisayar dışına yedeklemek için ayrıca bir DVC remote ayarlanmalı.

## MLflow

Eğitim kodu deneyleri `mlflow/tracking.db` veritabanına kaydeder. Önceki denemelerin veritabanı `mlflow/legacy.db` olarak korunur. MLflow, model ve ölçüm dosyalarını `mlruns/` altında tutar; bu klasör gereklidir ama VS Code dosya ağacında gizlenmiştir.

Arayüz için tam MLflow paketi, Python 3.12 kullanan `.mlflow-venv` ortamına kuruldu. Proje klasöründe arayüzü açmak için:

```powershell
.\start-mlflow.ps1
```

Sonra tarayıcıda `http://127.0.0.1:5000` adresini aç. Durdurmak için terminalde `Ctrl+C` tuşla. UI bağımlılıkları Python 3.15 kullanan ana `.venv` ortamına yüklenmez; eksiksiz paketler `requirements/mlflow-ui.txt` içinde listelenmiştir.

Bu arayüz ortamını başka bir bilgisayarda yeniden kurarken Python 3.12 ile `.mlflow-venv` oluştur ve `.mlflow-venv\Scripts\python.exe -m pip install -r requirements\mlflow-ui.txt` komutunu çalıştır.

## İlk sonuçlar

Karşılaştırma aynı sabit 419 issue'luk test kümesinde yapıldı:

- Eski 3.000 satırlık temel veri: doğruluk 0.947, macro F1 0.934.
- Genişletilmiş veri: doğruluk 0.952, macro F1 0.943.

Bu sonuçlar aynı test örnekleri üzerinde adil karşılaştırmadır; daha fazla veri tek başına başarı garantisi vermez. Etiket kalitesi ve kategori dengesi de önemlidir.
