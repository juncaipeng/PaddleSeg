export CUDA_VISIBLE_DEVICES=0,1
model=mobileseg_mv3_deepglobe_1024x1024_80k
tag=test_1

save_dir="output/${model}/${tag}"
mkdir -p ${save_dir}
cd ${save_dir}
rm -rf log_dir
rm -rf log_train.txt
rm -rf *.log
cd -

echo "cuda: ${CUDA_VISIBLE_DEVICES}"
echo "model: ${model}"
echo "tag: ${tag}"
echo "save_dir: ${save_dir}"

nohup python -m paddle.distributed.launch --log_dir ${save_dir}/log_dir train.py\
	--config configs/road_seg/${model}.yml \
	--save_dir ${save_dir} \
	--save_interval 200 \
	--num_workers 3 \
	--do_eval \
	--use_vdl \
	--log_iters 10 \
	2>&1 >${save_dir}/log_train.txt &

