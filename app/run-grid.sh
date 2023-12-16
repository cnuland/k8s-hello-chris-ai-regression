cat saves_to_record.txt | xargs -I {} sh -c "python3 run-grid-pretrained.py {}"
