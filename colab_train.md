# Training on free Google Colab / Kaggle GPU

```bash
!git clone https://github.com/<you>/pothole-defect-detector.git
%cd pothole-defect-detector
!pip install -r requirements.txt
import os; os.environ["ROBOFLOW_API_KEY"] = "<paste key here, do not commit>"
!python scripts/download_dataset.py
!python scripts/train.py --epochs 50
!python scripts/find_failures.py
```

Download `models/best.pt`, `runs/pothole_v8n/*.png` (confusion matrix, PR curve, results)
and `failure_cases/` from the Colab file browser.
