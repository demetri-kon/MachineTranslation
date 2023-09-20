import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# Initialize translation probabilities uniformly
translation_prob = defaultdict(lambda: 1.0 / len(opts.french))

sys.stderr.write("Training with IBM Model 1...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

# Initialize count dictionaries
count_e_given_f = defaultdict(float)
total_f = defaultdict(float)

# EM algorithm
for iteration in range(23):
    sys.stderr.write(f"Iteration {iteration + 1}...")
    
    # Initialize count dictionaries
    count_e_given_f.clear()
    total_f.clear()

    for (n, (f, e)) in enumerate(bitext):
        # Calculate normalization factor
        for e_i in e:
            total_s = 0.0
            for f_j in f:
                total_s += translation_prob[(e_i, f_j)]
            
            # Update counts
            for f_j in f:
                count = translation_prob[(e_i, f_j)] / total_s
                count_e_given_f[(e_i, f_j)] += count
                total_f[f_j] += count
        
    # Estimate translation probabilities
    for (e_i, f_j) in count_e_given_f.keys():
        translation_prob[(e_i, f_j)] = count_e_given_f[(e_i, f_j)] / total_f[f_j]
        
# Reverse alignment probabilities (swap e and f)
reverse_translation_prob = defaultdict(lambda: 1.0 / len(opts.english))
for (e_i, f_j) in translation_prob.keys():
    reverse_translation_prob[(f_j, e_i)] = translation_prob[(e_i, f_j)]

# Combine forward and reverse alignments
symmetric_translation_prob = defaultdict(float)
for (e_i, f_j) in translation_prob.keys():
    forward_prob = translation_prob[(e_i, f_j)]
    reverse_prob = reverse_translation_prob[(e_i, f_j)]
    symmetric_translation_prob[(e_i, f_j)] = (forward_prob + reverse_prob) / 2.0

# Output word alignment probabilities using symmetric_translation_prob
for (n, (f, e)) in enumerate(bitext):
    for (i, f_i) in enumerate(f):
        best_prob = 0.0
        best_j = -1
        for (j, e_j) in enumerate(e):
            if symmetric_translation_prob[(e_j, f_i)] > best_prob:
                best_prob = symmetric_translation_prob[(e_j, f_i)]
                best_j = j
        if best_j >= 0:
            sys.stdout.write(f"{i}-{best_j} ")
    sys.stdout.write("\n")
