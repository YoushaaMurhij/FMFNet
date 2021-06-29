for i in {15..15}
do
   echo "Evaluating epoch #$i:"
   python tools/dist_test.py /home/trainer/fmf/FMF/configs/waymo/pp/waymo_fmf_pp_two_pfn_stride1_3x.py \
    --work_dir waymo_exp/FMF-PP-New-$i \
    --checkpoint waymo_exp/FMF-PP-New/epoch_$i.pth  \
    --speed_test \
    --testset \
    --gpus 1
    echo "===================="
done