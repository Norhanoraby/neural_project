# Pneumonia Detection from Chest X-rays (Vision Transformer)

Fine-tuning a CXR-pretrained Vision Transformer
(`nickmuchi/vit-finetuned-chest-xray-pneumonia`) to classify chest X-rays
as **Pneumonia** vs. **Non-Pneumonia**, using the MIMIC-CXR-JPG-LITE dataset.
The project also produces Grad-CAM heatmaps to show which regions of the
image the model relies on.

## Dataset

MIMIC-CXR-JPG-LITE (Kaggle). 
/kaggle/input/datasets/phuong20052/mimic-cxr-jpg-lite


## Project structure

```
pneumonia-vit/
├── README.md
├── requirements.txt
├── config.py              # all paths & hyperparameters
├── main.py                # runs the full pipeline end-to-end
├── src/
│   ├── data_loading.py    # find CSVs, load tables, merge, label, attach images
│   ├── preprocessing.py   # clean (frontal + noise), balance, split
│   ├── dataset.py         # PyTorch Dataset + DataLoaders (resize + normalize)
│   ├── model.py           # load pretrained ViT + Grad-CAM wrapper
│   ├── train.py           # training loop (best model by val AUC)
│   ├── evaluate.py        # accuracy / precision / recall / F1 / AUC
│   └── visualize.py       # confusion matrix, prediction grid, Grad-CAM
└── notebooks/
    └──  notebook93345d525e.ipynb  # original exploratory notebook
```

## How it works

1. **Data loading** (`src/data_loading.py`) — finds the CSV tables, merges
   metadata + CheXpert + split, builds a binary pneumonia label, and matches
   each row to its image file.
2. **Preprocessing** (`src/preprocessing.py`) — filters to frontal (AP/PA)
   views, removes label noise, builds a balanced subset, and creates a
   stratified 70/15/15 split. `prepare_data()` chains loading + preprocessing.
2. **Model** (`src/model.py`) — loads the pretrained ViT and processor.
3. **Training** (`src/train.py`) — fine-tunes for 5 epochs, saving the
   checkpoint with the best validation AUC.
4. **Evaluation** (`src/evaluate.py`) — reports test metrics.
5. **Visualization** (`src/visualize.py`) — confusion matrix, sample
   prediction grid, and Grad-CAM overlays.

## Two experiments

- **1000 images** — balanced 500/500, basic preprocessing.
- **3000 images (clean)** — frontal-only + label-noise filtering, with
  weight decay added during training.

Toggle these in `main.py`.

## Running

```bash
pip install -r requirements.txt
python main.py
```

## Notes

- This is a progress submission; some pieces are still being refined.
- Trained `.pth` checkpoints and generated images are excluded via
  `.gitignore` to keep the repo light.
