for ptm in kac kcr khib kmal ksucc kla
do
    # sed '1d' /home/zhqin/project/2024/01.SLAM/RicePTM_result/ratio_all_transfer/SLAM_seq/${ptm}_logits_results.txt |awk -v OFS='\t' '{print $2,$1} '> results/logits/${ptm}.txt
    ./roc.out -i results/logits/${ptm}.txt -o results/logits/${ptm}_roc.txt
done

#  && ./roc.out results/logits/${ptm}.txt -o results/logits/${ptm}_roc.txt