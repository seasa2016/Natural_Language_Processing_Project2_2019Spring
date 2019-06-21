num=8
model='siamese'
word=('all_no' 'part_no')
#word=('all_no')
pred=('two_class' 'focal_two_class' 'three_class')
#pred=('focal_two_class')

for((i=0;$i<${#word[@]};i=i+1))
do
    for((j=0;$j<${#pred[@]};j=j+1))
    do
		echo ${model}_${word[i]}_${pred[j]}
        mkdir ./saved_models/${model}_${word[i]}_${pred[j]}_total
        python train.py --model ${model} --save ${model}_${word[i]}_${pred[j]}_total --pred ${pred[j]} --data ./data/${word[i]}_embedding --batch_size 128 --learning_rate 0.0005 > saved_models/${model}_${word[i]}_${pred[j]}_total/log &
    done
	wait
done
