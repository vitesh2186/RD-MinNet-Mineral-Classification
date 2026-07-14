# Mineral Classifier — Streamlit App

Classifies mineral photos into 7 classes (bornite, quartz, malachite, pyrite,
muscovite, biotite, chrysocolla) using whichever model you trained in the
`YOLOv8_CLS_Mineral_Classification` notebook.

## Setup

1. Put `app.py`, `requirements.txt`, and the `.streamlit/` folder in the same
   directory as your trained weight files:
   - `yolov8x_cls_best.pt`
   - `resnet50_minerals.pth`
   - `resnet101_minerals.pth`
   - `rd_minnet120.pth`

   (The app also checks a `minerals_results/weights/` subfolder automatically,
   in case you copy that whole folder over from Colab as-is.)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```

## Notes

- The sidebar shows a checklist of which weight files were found so you can
  tell at a glance if something's missing.
- Only the models whose weight file is present are usable — you don't need
  all four to run the app, but a model without its weights on disk will show
  an error if selected.
- RD-MinNet additionally reports predicted luster and Mohs hardness, since it
  was trained with those as auxiliary tasks.
- GPU is used automatically if available (`torch.cuda.is_available()`),
  otherwise it falls back to CPU.
