num=8
model='attnlstm'
word=('all_no' 'part_no')
#word=('all_no')
pred=('two_class' 'focal_two_class' 'three_class')
#pred=('two_class')


for((i=0;$i<${#word[@]};i=i+1))
do
    for((j=0;$j<${#pred[@]};j=j+1))
    do
		echo ${model}_${word[i]}_${pred[j]}_total
        python test.py --save ${model}_${word[i]}_${pred[j]}_total --data ./data/${word[i]}_embedding/test.csv --out ./saved_models/${model}_${word[i]}_${pred[j]}_total/pred
    done
done
