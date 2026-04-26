python train_csv.py \
  --csv_files nmo_dataset/all_log.csv \
  --ablation M \
  --epochs 10 --batch 8 --lr 2e-5 \
  --input_mode nmo \            # nmo or msg
  --outdir outputs_M_nmo \
  --device cuda
