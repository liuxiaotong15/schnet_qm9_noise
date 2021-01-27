# schnet_qm9_noise

python gen_q1q2q3.py

python add_noise.py -n 3 -i Q1.db -o Q13.db

python add_noise.py -n 1 -i Q2.db -o Q21.db

python merge_2_db.py -f Q1.db -s Q2.db -o Q1Q2.db

python merge_2_db.py -f Q13.db -s Q21.db -o Q13Q21.db
