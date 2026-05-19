MONTH_YEAR=$(date +%m%Y -d 'last month')

echo $MONTH_YEAR
sed -i "/MONTH_YEAR/s/=.*/=$MONTH_YEAR/g" config_fidji.ini
month=$(date +%B -d 'last month')
#for TD in 9
for TD in 3 5 7 9
do
echo "DRIFT_TIME : $TD"
sed -i "/DRIFT_TIME/s/=.*/=$TD/g" config_fidji.ini
sed -i "/EXP_NAME/s/=.*/=fidji_$month$TD/g" config_fidji.ini
echo "Launch Lagrangian_tool_traj_compute_fidji.py : "
#python Lagrangian_tool_traj_compute_fidji.py
#python Lagrangian_tool_lagrangian_scores_fidji.py
#python Lagrangian_tool_lagrangian_scores_more_fidji.py
python Lagrangian_tool_lagrangian_diffusion_scores_fidji.py

done



