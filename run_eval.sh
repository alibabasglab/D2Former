test_dir='/mnt/nas_sg/mit_sg/shengkui.zhao/DEMAND_SE_data/data/16kHz/test'
model_dir=ckpt
save_dir=saved_tracks_best

list=models_to_test.txt
cd $model_dir
ls D2Former* >../$list
cd ..
for model in $(cat ./$list)
do
  model_path=${model_dir}/${model}
  CUDA_VISIBLE_DEVICES='3' python3 evaluation.py --test_dir $test_dir --model_path $model_path --save_dir $save_dir
done
