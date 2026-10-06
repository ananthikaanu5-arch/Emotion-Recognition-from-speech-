# 🎙️ Emotion Recognition from Speech

Recognize human emotions (angry, disgust, fear, happy, neutral, sad, surprise) from speech audio using **speech signal processing** and **deep learning**.

## Approach

```
 audio (.wav) ──► trim silence / pad to 3 s ──► MFCC (40 coeffs) ──► normalise ──► CNN / RNN / LSTM / CNN-LSTM ──► emotion
```

- **Features:** Mel-Frequency Cepstral Coefficients (MFCCs) extracted with `librosa`
- **Models:** 1D-CNN, GRU-based RNN, Bidirectional LSTM, and a hybrid CNN-LSTM
- **Datasets:** [RAVDESS](https://zenodo.org/record/1188976), [TESS](https://tspace.library.utoronto.ca/handle/1807/24487), [EMO-DB](http://emodb.bilderbar.info/) - used individually or combined
- **Extras:** optional data augmentation (noise, time shift, pitch shift), early stopping, LR scheduling, confusion matrix and classification report

## Project structure

```
├── src/
│   ├── config.py      # constants (sample rate, MFCC settings, labels)
│   ├── datasets.py    # label parsing for RAVDESS / TESS / EMO-DB
│   ├── features.py    # audio loading, MFCC extraction, augmentation
│   ├── models.py      # CNN, RNN, LSTM, CNN-LSTM architectures
│   ├── train.py       # training pipeline
│   ├── evaluate.py    # metrics & plots
│   └── predict.py     # inference on a single file
├── tests/             # unit tests for label parsing
├── data/              # put datasets here (see data/README.md)
├── models/            # saved models (git-ignored)
├── results/           # plots & reports (git-ignored)
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/speech-emotion-recognition.git
cd speech-emotion-recognition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Download the datasets and place them in `data/` (see [`data/README.md`](data/README.md)).
2. Train:

```bash
python -m src.train --model cnn_lstm --epochs 60 --augment
# other options: --model cnn | rnn | lstm   --datasets ravdess tess   --batch_size 32
```

3. Predict on a new recording:

```bash
python -m src.predict --audio path/to/sample.wav --model_path models/cnn_lstm.keras
```

Outputs (model, `labels.json`, `scaler.npz`, training curves, confusion matrix, classification report) are written to `models/` and `results/`.

## Emotion label mapping

| Unified label | RAVDESS | TESS | EMO-DB |
|---|---|---|---|
| angry | angry | angry | Wut |
| disgust | disgust | disgust | Ekel |
| fear | fearful | fear | Angst |
| happy | happy | happy | Freude |
| neutral | neutral, calm | neutral | neutral |
| sad | sad | sad | Trauer |
| surprise | surprised | pleasant surprise | - |

EMO-DB's *boredom* class is skipped.

## Results

Add your own numbers here after training, e.g.:

| Model | Datasets | Test accuracy |
|---|---|---|
| CNN | RAVDESS + TESS + EMO-DB | _xx.x%_ |
| LSTM | RAVDESS + TESS + EMO-DB | _xx.x%_ |
| CNN-LSTM | RAVDESS + TESS + EMO-DB | _xx.x%_ |

> **Note:** the train/test split is random per file, so the same speaker can appear in both sets, which can inflate accuracy. For a stricter evaluation, split by speaker.

## Tech stack

Python · TensorFlow/Keras · librosa · scikit-learn · NumPy · Matplotlib · Seaborn

## License

MIT
