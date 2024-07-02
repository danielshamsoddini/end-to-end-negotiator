
Setup:

create a conda environment,
required packages are 
pytorch, visdom and
pandas, pyarrow (if parsing casino dataset again)




Training, Stage 1:
what i ran:
python train.py  --data data/casino --bsz 32  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 50  --min_lr 0.01  --momentum 0.1  --nembed_ctx 128  --nembed_word 512  --nesterov  --nhid_attn 512  --nhid_ctx 128  --nhid_lang 256  --nhid_sel 512  --nhid_strat 256  --sel_weight 0.5  --model_file logs/casino/supervised_casino.pt
original version:
python train.py  --data data/casino --bsz 16  --clip 0.5  --decay_every 1  --decay_rate 5.0  --dropout 0.5  --init_range 0.1  --lr 1  --max_epoch 30  --min_lr 0.01  --momentum 0.1  --nembed_ctx 64  --nembed_word 256  --nesterov  --nhid_attn 256  --nhid_ctx 64  --nhid_lang 128  --nhid_sel 256  --nhid_strat 128  --sel_weight 0.5  --model_file logs/casino/sv_model.pt

First Reinforce, Stage 2:
 python reinforce.py --data data/casino --bsz 16 --clip 1 --context_file data/casino/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/casino/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model logs/casino/supervised_casino.pt --bob_model logs/casino/supervised_casino.pt --rw_type utility --output_model_file logs/casino/rl_model.pt

Second Reinforce, Stage 3:
Supervised against selfish:
 python reinforce.py --data data/casino --bsz 16 --clip 1 --context_file data/casino/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/casino/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model logs/casino/supervised_casino.pt --bob_model logs/casino/rl_model_rw_utility_1_0_0_0.pt --rw_type utility --output_model_file logs/casino/stage_3_selfish.pt
Supervised against fair:
python reinforce.py --data data/casino --bsz 16 --clip 1 --context_file data/casino/selfplay.txt --eps 0.0 --gamma 0.95 --lr 0.5 --momentum 0.1 --nepoch 4 --nesterov --ref_text data/casino/train.txt --rl_clip 1 --rl_lr 0.2 --score_threshold 6 --sv_train_freq 4 --temperature 0.5 --alice_model logs/casino/supervised_casino.pt --bob_model logs/casino/rl_model_rw_utility_1_0_-0.75_-0.75.pt --rw_type utility --output_model_file logs/casino/stage_3_fair.pt

Results:

Stage 3 Selfish:
Selfish Partner:
final: train_loss 3.020 train_ppl 20.490
final: train_select_loss 0.891 train_select_ppl 2.438
final: valid_loss 3.162 valid_ppl 23.609
final: valid_select_loss 1.121 valid_select_ppl 3.067
final: dialog_len=14.33 sent_len=12.26 agree=58.33% advantage=2.57 time=1.206s comb_rew=36.00 alice_rew=18.75 alice_sel=34.53% alice_unique=407 bob_rew=17.25 bob_sel=65.47% bob_unique=347 full_match=0.32
Fair Partner:
final: train_loss 3.027 train_ppl 20.636
final: train_select_loss 0.801 train_select_ppl 2.227
final: valid_loss 3.169 valid_ppl 23.789
final: valid_select_loss 1.011 valid_select_ppl 2.750
final: dialog_len=8.10 sent_len=11.89 agree=90.00% advantage=7.75 time=0.728s comb_rew=45.45 alice_rew=26.21 alice_sel=46.48% alice_unique=374 bob_rew=19.24 bob_sel=53.52% bob_unique=342 full_match=0.33
Stage 3 Fair:
Selfish Partner:
final: train_loss 3.023 train_ppl 20.545
final: train_select_loss 0.981 train_select_ppl 2.667
final: valid_loss 3.166 valid_ppl 23.701
final: valid_select_loss 1.147 valid_select_ppl 3.150
final: dialog_len=10.81 sent_len=16.12 agree=81.25% advantage=2.08 time=1.195s comb_rew=50.06 alice_rew=25.88 alice_sel=51.06% alice_unique=464 bob_rew=24.19 bob_sel=48.94% bob_unique=428 full_match=0.26
Fair Partner:
final: train_loss 3.031 train_ppl 20.717
final: train_select_loss 0.830 train_select_ppl 2.292
final: valid_loss 3.173 valid_ppl 23.889
final: valid_select_loss 1.033 valid_select_ppl 2.809
final: dialog_len=9.59 sent_len=14.43 agree=81.82% advantage=3.39 time=0.789s comb_rew=38.86 alice_rew=20.90 alice_sel=42.55% alice_unique=436 bob_rew=17.97 bob_sel=56.74% bob_unique=411 full_match=0.26