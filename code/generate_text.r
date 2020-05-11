require(progress)
require(stringr)
require(yaml)

# Bring in any parameters we need  -----------------------------------------------
FLAGS <- yaml.load_file('./generate_text.yml')

# Bring helper functions  --------------------------------------------------------
source('load_corpus.r')
source('dnn_model.r')
source('utils.r')

# Load our corpus and create our model -------------------------------------------
corpus <- load_corpus('../data')
# override the coupus length because they are crazy
corpus$max_sentence_length <- FLAGS$max_sentence_length
model <- create_dnn(corpus$max_sentence_length, length(corpus$vocab))

# Training -----------------------------------------------------------------------
for(document in corpus$documents) {
  
  samples <- make_samples(document, corpus$max_sentence_length + 1)
  sz = dim(samples)

  # The `train_on_batch()` function is the most granular training.
  # In our lab, we like that level of control.
  # In your case just using `fit()` on the whole thing may make sense
  for(i in 1:FLAGS$epochs) {
    batch_i <- 1
    pb <-
      progress_bar$new(
        format = 'batch :current/:total [:bar] eta: :eta',
        total = ceiling(sz[1]/FLAGS$batch_size))
    pb$tick(0)
    while(batch_i <= sz[1]) {
      batch <- make_batch(samples, batch_i, FLAGS$batch_size)
      one_hot <- one_hot_batch(batch, corpus$vocab)
      loss = train_on_batch(model, one_hot$x, one_hot$y)
      rm(batch, one_hot)
      batch_i <- batch_i + FLAGS$batch_size
      pb$tick()
    }
    print(sprintf('Epoch: %i, Loss: %f', i, loss))
  }
}
