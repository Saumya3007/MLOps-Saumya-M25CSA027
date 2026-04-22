# CityScape Image Segmentation Pipeline

## Model Performance
**Question2: mIOU: 0.4895 and mDICE: 0.5528**

---

## My Approach
1. **Model Architecture**: Used a **UNet** based encoder-decoder architecture. 
   - **Encoder**: 4 levels of Double Convolution blocks followed by Max Pooling.
   - **Bottleneck**: Deepest layer with 1024 filters.
   - **Decoder**: 4 levels of Transposed Convolutions (Upsampling) concatenated with skip connections from the encoder to preserve spatial details.
2. **Dataset**: 
   - Downloaded CityScape dataset using `gdown`.
   - Classes: 23 semantic classes.
   - **Split**: 80% Training, 20% Testing (Seed 42).
   - Preprocessing: Correctly extracted class IDs from the **Red channel** of the RGBA masks. Resized to 256x256.
3. **Training**:
   - Optimizer: Adam (Learning Rate: 1e-3).
   - Loss Function: Cross Entropy Loss.
   - Duration: 20 epochs on Tesla V100 GPU.
4. **Deployment**: Built a Streamlit app with two main pages:
   - **Training Dashboard**: Visualizes Loss, mIOU, and mDICE curves.
   - **Prediction Demo**: Allows uploading images from the test set to see Ground Truth vs Model Prediction.

---

## Steps to Run the Pipeline

### 1. Build the Docker Environment
```bash
docker build -t q2_seg .
```

### 2. Run the Streamlit App (Deployment)
To launch the deployment dashboard on port **7860**:
```bash
docker run -p 7860:7860 -v $(pwd):/app q2_seg streamlit run app.py --server.port 7860 --server.address 0.0.0.0
```
Then open `http://localhost:7860` in your browser.

---

## Folder Structure
- `train.py`: Training script with UNet definition, data loading, and evaluation.
- `app.py`: Streamlit application for deployment.
- `Dockerfile`: Environment setup (exposed on 7860).
- `outputs/`: 
  - `unet_cityscape.pth`: Final trained weights.
  - `metrics.json`: Final mIOU/mDICE scores.
  - `training_plots.png`: Training history curves.
  - `test_results.log`: Text log of final evaluation scores.
