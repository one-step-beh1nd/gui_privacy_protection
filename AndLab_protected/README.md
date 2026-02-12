python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml

python eval.py -n paper_xml -c ./configs/gpt-4o-linux-XML.yaml -p 3

python eval.py -n paper_som -c ./configs/gpt-4o-linux-SoM.yaml

python eval.py -n [log directory name] -c [path to config file]

python eval.py -n gemini_xml -c ./configs/gemini-linux-XML.yaml --task_id bluecoins_1,calendar_9,cantook_2,cantook_5,clock_13


python generate_result.py \
  --input_folder ./logs/evaluation \
  --output_folder ./outputs \
  --output_excel ./outputs/[output file name].xlsx \
  --judge_model [model] \
  --api_key  [api key] \
  --api_base  [base url] \
  --target_dirs [target directory name in logs]