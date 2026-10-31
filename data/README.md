# Dataset

Due to GitHub's file size limits and dataset licenses, no raw data is hosted in this repository. Everything in `data/` except this file is ignored by git.

## Kinetics-400 (pretrained model evaluation)

Used for the SlowFast dry-run evaluation in `result/slowfast_summary.md`.

1. Download `part_0.tar.gz` of the Kinetics-400 validation split and `val.csv` from the [CVDF mirror](https://github.com/cvdfoundation/kinetics-dataset).
2. Extract the videos into `data/kinetics400/videos/`.
3. Create `data/kinetics400_50/test.csv` listing the first 50 videos, one per line, in PySlowFast format `<path_to_video> <label_index>`:

   ```
   /path/to/data/kinetics400/videos/--07WQ2iBlw_000001_000011.mp4 166
   ```

4. Pass the folder to PySlowFast with `DATA.PATH_TO_DATA_DIR data/kinetics400_50`.

## State Farm Distracted Driver Detection

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data).
2. Extract the contents into `data/statefarm/`.
3. The expected structure is:
   - `data/statefarm/imgs/train/c0/` ... `data/statefarm/imgs/train/c9/`
   - `data/statefarm/imgs/test/`
   - `data/statefarm/driver_imgs_list.csv`

## AUC Distracted Driver Dataset (AUCDD)

The dataset is available on request from its authors ([Abouelnaga et al., 2017](https://arxiv.org/abs/1706.09498)). Extract it into `data/aucdd/` and keep the original folder structure.

## Expected layout

```
data/
├── README.md
├── kinetics400/
│   └── videos/
├── kinetics400_50/
│   └── test.csv
├── statefarm/
│   ├── imgs/
│   └── driver_imgs_list.csv
└── aucdd/
```
