# schnet_qm9_noise

(if not run first) rm -rf \*db

python gen_q1q2q3.py

python add_noise.py -n 3 -i Q1.db -o Q13.db

python add_noise.py -n 1 -i Q2.db -o Q21.db

python merge_2_db.py -f Q1.db -s Q2.db -o Q1Q2.db

python merge_2_db.py -f Q13.db -s Q21.db -o Q13Q21.db

-------------------------------------------------------

python train.py -t Q1Q2.db

python evaluate.py -t Q3.db -f Q1Q2db

-------------------------------------------------------

python train.py -t Q13Q21.db

python evaluate.py -t Q3.db -f Q13Q21db

--------------------------------------------------------

python train.py -t Q13.db

python train.py -t Q21.db -p Q13db

python calibration.py -i Q13.db -o Q13E.db -f Q21db

python merge_2_db.py -f Q13E.db -s Q21.db -o Q13EQ21.db

python train.py -t Q13EQ21.db

python evaluate.py -t Q3.db -f Q13EQ21db

