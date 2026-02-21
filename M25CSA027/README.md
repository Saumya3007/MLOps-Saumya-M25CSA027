/usr/local/lib/python3.12/dist-packages/notebook/notebookapp.py:191: SyntaxWarning: invalid escape sequence '\/'
  | |_| | '_ \/ _` / _` |  _/ -_)
wandb: WARNING If you're specifying your api key in code, ensure this code is not shared publicly.
wandb: WARNING Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
wandb: [wandb.login()] Using explicit session credentials for https://api.wandb.ai.
wandb: No netrc file found, creating one.
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Currently logged in as: pancholisaumya (pancholisaumya-iit) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
Device: cuda
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
README.md: 
 1.18k/? [00:00<00:00, 108kB/s]
data/train-00000-of-00001.parquet: 100%
 11.4M/11.4M [00:01<00:00, 7.52MB/s]
data/validation-00000-of-00001.parquet: 100%
 1.15M/1.15M [00:00<00:00, 1.96MB/s]
data/test-00000-of-00001.parquet: 100%
 2.29M/2.29M [00:00<00:00, 4.29MB/s]
Generating train split: 100%
 5000/5000 [00:00<00:00, 66309.96 examples/s]
Generating validation split: 100%
 500/500 [00:00<00:00, 20467.21 examples/s]
Generating test split: 100%
 1000/1000 [00:00<00:00, 31679.03 examples/s]
Train: 5000, Val: 500, Test: 1000
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|██████████| 44.7M/44.7M [00:00<00:00, 211MB/s]
Tracking run with wandb version 0.25.0
Run data is saved locally in /content/wandb/run-20260221_102638-2e4drb62
Syncing run copper-shadow-2 to Weights & Biases (docs)
View project at https://wandb.ai/pancholisaumya-iit/cifar10_resnet18
View run at https://wandb.ai/pancholisaumya-iit/cifar10_resnet18/runs/2e4drb62
Epoch 01/15 | Train 0.9930/0.6588 | Val 1.0472/0.6740
  ✅ Best model saved  val_acc=0.6740
Epoch 02/15 | Train 0.6206/0.7928 | Val 0.7914/0.7360
  ✅ Best model saved  val_acc=0.7360
Epoch 03/15 | Train 0.4865/0.8332 | Val 1.0474/0.7040
Epoch 04/15 | Train 0.4227/0.8592 | Val 0.6011/0.7980
  ✅ Best model saved  val_acc=0.7980
Epoch 05/15 | Train 0.2919/0.9024 | Val 0.7115/0.7680
Epoch 06/15 | Train 0.1580/0.9474 | Val 0.4990/0.8380
  ✅ Best model saved  val_acc=0.8380
Epoch 07/15 | Train 0.0711/0.9786 | Val 0.4621/0.8500
  ✅ Best model saved  val_acc=0.8500
Epoch 08/15 | Train 0.0445/0.9892 | Val 0.4484/0.8740
  ✅ Best model saved  val_acc=0.8740
Epoch 09/15 | Train 0.0801/0.9768 | Val 0.5553/0.8320
Epoch 10/15 | Train 0.1085/0.9634 | Val 0.5076/0.8620
Epoch 11/15 | Train 0.0337/0.9916 | Val 0.4459/0.8680
Epoch 12/15 | Train 0.0165/0.9974 | Val 0.4464/0.8740
Epoch 13/15 | Train 0.0100/0.9986 | Val 0.4209/0.8820
  ✅ Best model saved  val_acc=0.8820
Epoch 14/15 | Train 0.0116/0.9982 | Val 0.4431/0.8820
Epoch 15/15 | Train 0.0314/0.9910 | Val 0.4583/0.8860
  ✅ Best model saved  val_acc=0.8860
/usr/local/lib/python3.12/dist-packages/huggingface_hub/hf_api.py:9786: UserWarning: Warnings while validating metadata in README.md:
- empty or missing yaml metadata in repo card
  warnings.warn(f"Warnings while validating metadata in README.md:\n{message}")
Processing Files (1 / 1)      : 100%
 44.8MB / 44.8MB, 7.00MB/s  
New Data Upload               : 100%
 44.8MB / 44.8MB, 7.00MB/s  
  ..._upload/pytorch_model.bin: 100%
 44.8MB / 44.8MB            
✅ Model pushed → https://huggingface.co/Saumya3007/cifar10-resnet18
pytorch_model.bin: 100%
 44.8M/44.8M [00:05<00:00, 7.62MB/s]
✅ Model loaded from HuggingFace

📊 Test Accuracy: 0.8720
Class-wise Accuracy:
  Class 0 (airplane    ): 0.8500
  Class 1 (automobile  ): 0.9400
  Class 2 (bird        ): 0.7300
  Class 3 (cat         ): 0.7300
  Class 4 (deer        ): 0.8800
  Class 5 (dog         ): 0.8500
  Class 6 (frog        ): 0.9500
  Class 7 (horse       ): 0.9200
  Class 8 (ship        ): 0.9100
  Class 9 (truck       ): 0.9600
✅ Confusion matrix logged
✅ Bar plot logged
✅ 20 test samples logged


Run history:

epoch	▁▁▂▃▃▃▄▅▅▅▆▇▇▇█
train_acc	▁▄▅▅▆▇███▇█████
train_loss	█▅▄▄▃▂▁▁▁▂▁▁▁▁▁
val_acc	▁▃▂▅▄▆▇█▆▇▇████
val_loss	█▅█▃▄▂▁▁▃▂▁▁▁▁▁

Run summary:

acc_airplane	0.85
acc_automobile	0.94
acc_bird	0.73
acc_cat	0.73
acc_deer	0.88
acc_dog	0.85
acc_frog	0.95
acc_horse	0.92
acc_ship	0.91
acc_truck	0.96
+6	...

View run copper-shadow-2 at: https://wandb.ai/pancholisaumya-iit/cifar10_resnet18/runs/2e4drb62
View project at: https://wandb.ai/pancholisaumya-iit/cifar10_resnet18
Synced 5 W&B file(s), 3 media file(s), 26 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20260221_102638-2e4drb62/logs

==================================================
✅ FINAL TEST ACCURACY : 0.8720
==================================================
Class-wise Accuracy for exam sheet:
  Class 0 (airplane    ): 0.8500
  Class 1 (automobile  ): 0.9400
  Class 2 (bird        ): 0.7300
  Class 3 (cat         ): 0.7300
  Class 4 (deer        ): 0.8800
  Class 5 (dog         ): 0.8500
  Class 6 (frog        ): 0.9500
  Class 7 (horse       ): 0.9200
  Class 8 (ship        ): 0.9100
  Class 9 (truck       ): 0.9600
