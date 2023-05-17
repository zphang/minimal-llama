set -e
export PYTHONPATH=/fsx/zphang/code/transformers/src:${PYTHONPATH}
export PYTHONPATH=/fsx/zphang/code/minimal-llama:${PYTHONPATH}
mkdir -p ${2}
aws s3 cp ${1} ${2} --recursive
cd cd /home/zp489/code/eleutherai/minimal-llama
python minimal_llama/utils/zero_to_torch.py --fp16 ${1} ${2}
rm -r ${2}