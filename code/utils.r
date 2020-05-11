
make_samples <- function(tokens, sample_length) {
  samples <- vector(mode = 'list', length = length(tokens) - sample_length)
  for(j in 1:length(samples)) {
    samples[[j]] <- tokens[j:(j + sample_length - 1)]
  }
  samples
}

make_batch <- function(samples, batch_start, batch_size) {
  sz = dim(samples)
  batch <- batch_start:min(sz[1], batch_start + batch_size - 1)
  batch_x <- samples[batch,-sz[2]]
  batch_y <- samples[batch, sz[2]]
  list(x = batch_x, y = batch_y)
}

one_hot_batch <- function(batch, unique_tokens) {
  one_hot_x <- array(0, dim = c(dim(batch$x), length(unique_tokens)))
  one_hot_y <- array(0, dim = c(length(batch$y), length(unique_tokens)))
  sz = dim(batch$x)
  for(i in 1:sz[1]) {
    for(j in 1:sz[2]) {
      one_hot_x[i,j,] <- as.integer(unique_tokens == batch$x[i,j])
    }
    one_hot_y[i,] <- as.integer(unique_tokens == batch$y[i])
  }
  list(x = one_hot_x, y = one_hot_y)
}
