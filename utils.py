import os, sys

def kmer2seq(kmers):
    """
    Convert kmers to original sequence
    
    Arguments:
    kmers -- str, kmers separated by space.
    
    Returns:
    seq -- str, original sequence.
    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def data2train(file_path, k):
    """
    Convert CNN+GRU model data to DNABERT pre-training input

    Arguments:
    path -- file path

    Returns: void
    Outputs: pre-training input file for DNABERT
    """
    result = ""

    file = open(file_path, "r")
    while True:
        line = file.readline() # skip >EP~~~
        line = file.readline()
        if not line:
            break
        result += seq2kmer(line.strip(), k) + "\n"
    file.close()
    
    return result.strip()
    


def data2tune(file_path, k, label):
    """
    Convert CNN+GRU model data to DNABERT finetuning input

    Arguments:
    path -- file path

    Returns: void
    Outputs: finetuning input file for DNABERT
    """
    result = "sequence	label\n"

    file = open(file_path, "r")
    while True:
        line = file.readline() # skip >EP~~~
        line = file.readline()
        if not line:
            break
        result += seq2kmer(line.strip(), k) + " " + label + "\n"
    file.close()
    
    return result.strip()
    
    return result

if __name__ == "__main__":
    if not len(sys.argv) in range (4, 6):
        print("wrong arguments")
        exit(0)
    
    job = sys.argv[1] # train or tune
    data_name = sys.argv[2] # data file name (without .txt)
    k = sys.argv[3] # k for kmer
    if(job == "tune"):
        val = sys.argv[4]

    data_dir = "./gru_input"
    save_dir = "./bert_input"

    if job == "train":
        res_lines = data2train(os.path.join(data_dir, f"{data_name}.txt"), int(k))
        res_file = open(os.path.join(save_dir, f"{data_name}_train.txt"), "w")
        res_file.write(res_lines)
        res_file.close()
    elif job == "tune":
        res_lines = data2tune(os.path.join(data_dir, f"{data_name}.txt"), int(k), val)
        res_file = open(os.path.join(save_dir, f"{data_name}_tune.txt"), "w")
        res_file.write(res_lines)
        res_file.close()
    else:
        raise ValueError
