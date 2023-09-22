import optparse
import sys
from collections import defaultdict

def add_one_smoothing(probabilities, smoothing_value):
    smoothed_probabilities = defaultdict(lambda: smoothing_value)
    for key, value in probabilities.items():
        smoothed_probabilities[key] = (value + smoothing_value) / (1.0 + smoothing_value * len(probabilities))
    return smoothed_probabilities

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# Initialize translation probabilities with add-one smoothing
smoothing_value = 5  # Adjust this value!!!!!
translation_prob = defaultdict(lambda: smoothing_value)

sys.stderr.write("Training with IBM Model 1 (with add-one smoothing)...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

# Initialize count dictionaries
count_e_given_f = defaultdict(float)
total_f = defaultdict(float)

# EM algorithm
for iteration in range(20):
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
        
    # Estimate translation probabilities with add-one smoothing
    translation_prob = add_one_smoothing(count_e_given_f, smoothing_value)

# Output word alignment probabilities
for (n, (f, e)) in enumerate(bitext):
    for (i, f_i) in enumerate(f):
        best_prob = 0.0
        best_j = -1
        for (j, e_j) in enumerate(e):
            if translation_prob[(e_j, f_i)] > best_prob:
                best_prob = translation_prob[(e_j, f_i)]
                best_j = j
        if best_j >= 0:
            sys.stdout.write(f"{i}-{best_j} ")
    sys.stdout.write("\n")
