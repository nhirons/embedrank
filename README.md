# EmbedRank
An exploratory word importance ranking method on an embedded space by:

1. Creating a similarity matrix of words from an embedded space (eg. from an RBF kernel with parameter gamma)
2. Optionally thresholding the similarities to enforce sparsity (NOTE: yet to be implemented)
3. Softmaxing by non-zero row elements with some temperature tau, while zeroing out the diagonals, to imitate a transition matrix of a directed word graph
4. running PageRank on this "transition matrix" to establish the most important words.

The results of word importance are highly sensitive to gamma and tau, and these can be tuned to explore the tradeoff between the importance of strong neighborhood proximity and proximity to more distant points.