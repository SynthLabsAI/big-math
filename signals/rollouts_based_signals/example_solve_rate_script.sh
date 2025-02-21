# define some variables

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# path to the dataset to generate rollouts on
EVAL_DATASET_PATH="SynthLabsAI/Big-Math-RL-Verified"

# path to save the updated dataset to: change this to a valid path
SAVE_DATASET_PATH="<your dataset path here>"

# local path to save intermediate results to
INTERMEDIATE_STEPS_SAVE_DIR="testing_inference_eval_pipeline"

# column to save the models responses to
RESPONSE_COLUMN_NAME="llama8b_response"

# set the number of processes to use
HF_NUM_PROC=16

# make sure the user changed the SAVE_DATASET_PATH
if [ "${SAVE_DATASET_PATH}" == "<your dataset path here>" ]; then
    echo "Please change the SAVE_DATASET_PATH variable to a valid path"
    exit 1
fi

# first, we need to run the inference script
echo "Running inference on ${MODEL} on ${EVAL_DATASET_PATH} and saving to ${SAVE_DATASET_PATH}"
mkdir -p ${INTERMEDIATE_STEPS_SAVE_DIR}
cd inference
python3 sample_from_model.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name ${EVAL_DATASET_PATH} --dataset_name_outputs ${SAVE_DATASET_PATH} --response_column_name ${RESPONSE_COLUMN_NAME} --save_folder ${INTERMEDIATE_STEPS_SAVE_DIR} --save_name llama8b_inference --max_tokens 2048 --n 64

# then, we evaluate the response correctness
echo "Evaluating responses"
cd ..
python3 evaluate_responses.py --predictions_dataset_name ${SAVE_DATASET_PATH} --response_column_name ${RESPONSE_COLUMN_NAME} --ground_truth_answer_column_name final_answer --num_proc ${HF_NUM_PROC} --save_folder ${INTERMEDIATE_STEPS_SAVE_DIR} --save_name llama8b_eval_responses

echo "Done!"