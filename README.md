# Player Re-Identification in Sports Footage

This project implements two solutions for player re-identification in sports analytics:

- **Task 1: Cross-Camera Player Mapping**
- **Task 2: Single-Feed Player Re-Identification**

## Folder Structure

- `src/` — Source code
- `videos/` — Input videos (populated automatically)
- `models/` — Model weights (populated automatically)
- `output/` — All output videos/results
- `Data/` — (Optional) Downloaded Google Drive folder (auto-copied to correct locations)
- `report.md` — Project report and methodology

## Setup
1. **Set up environment:**
 ``` bash 
   python -m venv venv
   venv/Scripts/Activate # For Windows 
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download all required files and prepare folders:**
   ```bash
   python src/main.py --download
   ```
   This will:
   - Download all files to the correct folders using Info.txt
   - Copy all `.mp4` files to `videos/`
   - Copy the model file (e.g., `best.pt`) to `models/model.pt`
   - Create the `output/` directory

## Running the Tasks

### Task 1: Cross-Camera Player Mapping

- **Input:** `videos/broadcast.mp4`, `videos/tacticam.mp4`, `models/model.pt`
- **Output:**
  - `output/cross_camera_broadcast_output.mp4` (broadcast video with IDs)
  - `output/cross_camera_tacticam_output.mp4` (tactical video with IDs)

Run:
```bash
python src/main.py --task 1
```

You can also specify custom paths:
```bash
python src/main.py --task 1 --broadcast videos/broadcast.mp4 --tacticam videos/tacticam.mp4 --model models/model.pt --out_broadcast output/cross_camera_broadcast_output.mp4 --out_tacticam output/cross_camera_tacticam_output.mp4
```

### Task 2: Single-Feed Player Re-Identification

- **Input:** `videos/15sec_input_720p.mp4`, `models/model.pt`
- **Output:**
  - `output/single_feed_output.mp4` (single feed video with IDs)

**You can specify both `--input` and `--output` arguments for Task 2.**
Run:
```bash
python src/main.py --task 2
```
You can also run to specify path:
```bash
python src/main.py --task 2 --input videos/15sec_input_720.mp4 --model models/model.pt --output output/single_feed_output.mp4
```

> **Note:** If you do not specify both `--input` and `--output`, the script will show an error and exit.

## Notes
- No GUI windows are shown; all results are saved as videos in `output/`.
- The scripts will automatically copy files from `Data/` to the correct locations if needed.
- The code is fully automated for reproducible results.
- See `report.md` for methodology and findings.
- See `src/utils.py` for helper functions.
- See `src/detector.py` for YOLOv11 inference wrapper. 